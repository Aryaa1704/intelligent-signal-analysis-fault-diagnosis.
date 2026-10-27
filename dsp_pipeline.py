"""
IntelliSignal — DSP Pipeline
FFT | STFT | Wavelet Transform | Spectrogram Generation
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import pywt
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class SignalData:
    samples: np.ndarray          # raw signal samples
    sample_rate: float           # Hz
    label: str = "unknown"       # e.g. "Tone", "Chirp", "Pulse", "Normal"
    source_path: Optional[str] = None


@dataclass
class DSPResult:
    # FFT
    fft_freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    fft_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))

    # STFT
    stft_freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    stft_times: np.ndarray = field(default_factory=lambda: np.array([]))
    stft_Zxx: np.ndarray = field(default_factory=lambda: np.array([]))   # complex STFT output
    stft_magnitude: np.ndarray = field(default_factory=lambda: np.array([]))  # |Zxx| in dB

    # Wavelet
    wavelet_coeffs: list = field(default_factory=list)   # [cA_n, cD_n, ..., cD_1]
    wavelet_name: str = "db4"
    wavelet_level: int = 5

    # Spectrogram (400×400 — for model input, matches WASD/Kim et al. 2025)
    spectrogram_image: Optional[np.ndarray] = None   # shape (400, 400)


# ─────────────────────────────────────────────
# Loaderss
# ─────────────────────────────────────────────

class SignalLoader:
    """Load signals from CSV, WAV, or IQ binary files."""

    @staticmethod
    def from_csv(path: str, sample_rate: float = 1e6, col: int = 0) -> SignalData:
        """
        CSV with one or more columns of float samples.
        col: which column to use (default 0).
        """
        data = np.loadtxt(path, delimiter=",")
        if data.ndim > 1:
            samples = data[:, col].astype(np.float32)
        else:
            samples = data.astype(np.float32)
        return SignalData(samples=samples, sample_rate=sample_rate,
                         source_path=path)

    @staticmethod
    def from_wav(path: str) -> SignalData:
        """Standard WAV file (mono or first channel)."""
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        samples = data.astype(np.float32)
        # Normalise to [-1, 1]
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples /= peak
        return SignalData(samples=samples, sample_rate=float(sr),
                         source_path=path)

    @staticmethod
    def from_iq(path: str, sample_rate: float = 2.048e6,
                dtype: np.dtype = np.float32) -> SignalData:
        """
        RTL-SDR / HackRF style interleaved IQ binary.
        Pairs of (I, Q) floats → complex signal → magnitude.
        """
        raw = np.fromfile(path, dtype=dtype)
        iq = raw[0::2] + 1j * raw[1::2]
        samples = np.abs(iq).astype(np.float32)
        return SignalData(samples=samples, sample_rate=sample_rate,
                         source_path=path)

    @staticmethod
    def from_oscilloscope_csv(path: str, time_col: int = 0,
                               voltage_col: int = 1) -> SignalData:
        """
        Oscilloscope CSV export: header rows skipped automatically.
        Columns: time (s), voltage (V).
        """
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        times = data[:, time_col]
        samples = data[:, voltage_col].astype(np.float32)
        dt = np.mean(np.diff(times))
        sample_rate = 1.0 / dt if dt > 0 else 1e6
        return SignalData(samples=samples, sample_rate=sample_rate,
                         source_path=path)


# ─────────────────────────────────────────────
# DSP Pipeline
# ─────────────────────────────────────────────

class DSPPipeline:
    """
    Unified DSP pipeline for IntelliSignal.
    Runs FFT → STFT → Wavelet → 400×400 spectrogram.
    """

    def __init__(
        self,
        fft_n: int = 4096,
        stft_nperseg: int = 3200,
        stft_noverlap: int = 1600,
        stft_window: str = "blackmanharris",   # matches Kim et al. 2025
        wavelet_name: str = "db4",
        wavelet_level: int = 5,
        spectrogram_size: int = 400,           # 400×400 for model input
    ):
        self.fft_n = fft_n
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap
        self.stft_window = stft_window
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.spectrogram_size = spectrogram_size

    # ── FFT ─────────────────────────────────

    def compute_fft(self, sig: SignalData) -> tuple[np.ndarray, np.ndarray]:
        """
        One-sided FFT magnitude spectrum.
        Returns: (freqs_Hz, magnitudes_dB)
        """
        n = len(sig.samples)
        window = np.hanning(n)
        spectrum = np.fft.rfft(sig.samples * window, n=self.fft_n)
        magnitudes = np.abs(spectrum)
        magnitudes_db = 20 * np.log10(magnitudes + 1e-12)
        freqs = np.fft.rfftfreq(self.fft_n, d=1.0 / sig.sample_rate)
        return freqs, magnitudes_db

    # ── STFT ────────────────────────────────

    def compute_stft(
        self, sig: SignalData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Short-Time Fourier Transform using Blackman-Harris window.
        Returns: (freqs, times, Zxx_complex, magnitude_dB)
        """
        freqs, times, Zxx = signal.stft(
            sig.samples,
            fs=sig.sample_rate,
            window=self.stft_window,
            nperseg=self.stft_nperseg,
            noverlap=self.stft_noverlap,
        )
        magnitude = np.abs(Zxx)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        return freqs, times, Zxx, magnitude_db

    # ── Wavelet ─────────────────────────────

    def compute_wavelet(self, sig: SignalData) -> list:
        """
        Discrete Wavelet Transform (multi-level decomposition).
        Returns: list of coefficients [cA_n, cD_n, ..., cD_1]
        """
        coeffs = pywt.wavedec(
            sig.samples,
            wavelet=self.wavelet_name,
            level=self.wavelet_level,
        )
        return coeffs

    # ── Spectrogram → 400×400 ───────────────

    def make_spectrogram_image(
        self, magnitude_db: np.ndarray
    ) -> np.ndarray:
        """
        Resize STFT magnitude (dB) to 400×400 grayscale array.
        Matches WASD dataset format used in Kim et al. 2025.
        """
        from PIL import Image  # lazy import

        # Normalise to 0-255
        vmin, vmax = magnitude_db.min(), magnitude_db.max()
        norm = (magnitude_db - vmin) / (vmax - vmin + 1e-12)
        img_uint8 = (norm * 255).astype(np.uint8)

        pil_img = Image.fromarray(img_uint8, mode="L")
        pil_img = pil_img.resize(
            (self.spectrogram_size, self.spectrogram_size),
            resample=Image.LANCZOS,
        )
        return np.array(pil_img)

    # ── Full Run ────────────────────────────

    def run(self, sig: SignalData) -> DSPResult:
        """
        Run the full DSP pipeline on a SignalData object.
        Returns a DSPResult with all transforms populated.
        """
        result = DSPResult(
            wavelet_name=self.wavelet_name,
            wavelet_level=self.wavelet_level,
        )

        # 1. FFT
        result.fft_freqs, result.fft_magnitudes = self.compute_fft(sig)

        # 2. STFT
        (result.stft_freqs,
         result.stft_times,
         result.stft_Zxx,
         result.stft_magnitude) = self.compute_stft(sig)

        # 3. Wavelet
        result.wavelet_coeffs = self.compute_wavelet(sig)

        # 4. 400×400 spectrogram image (for model input)
        result.spectrogram_image = self.make_spectrogram_image(result.stft_magnitude)

        return result


# ─────────────────────────────────────────────
# Visualiser (optional — for dashboard/debug)
# ─────────────────────────────────────────────

class DSPVisualiser:
    """Render DSP results as base64-encoded PNGs for API/frontend."""

    @staticmethod
    def _fig_to_b64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64

    @staticmethod
    def plot_fft(result: DSPResult, sig_label: str = "") -> str:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result.fft_freqs / 1e6, result.fft_magnitudes, linewidth=0.8)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"FFT Spectrum — {sig_label}")
        ax.grid(True, alpha=0.3)
        return DSPVisualiser._fig_to_b64(fig)

    @staticmethod
    def plot_stft(result: DSPResult, sig_label: str = "") -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.pcolormesh(
            result.stft_times * 1e3,
            result.stft_freqs / 1e6,
            result.stft_magnitude,
            shading="gouraud",
            cmap="viridis",
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(f"STFT Spectrogram — {sig_label}")
        return DSPVisualiser._fig_to_b64(fig)

    @staticmethod
    def plot_wavelet(result: DSPResult, sig_label: str = "") -> str:
        n_plots = len(result.wavelet_coeffs)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots))
        labels = ["Approx"] + [f"Detail L{i}" for i in range(
            result.wavelet_level, 0, -1)]
        for ax, coeff, lbl in zip(axes, result.wavelet_coeffs, labels):
            ax.plot(coeff, linewidth=0.6)
            ax.set_title(f"{lbl} — {sig_label}", fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return DSPVisualiser._fig_to_b64(fig)

    @staticmethod
    def plot_spectrogram_400(result: DSPResult, sig_label: str = "") -> str:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(result.spectrogram_image, cmap="gray", aspect="auto",
                  origin="lower")
        ax.set_title(f"400×400 Spectrogram — {sig_label}")
        ax.axis("off")
        return DSPVisualiser._fig_to_b64(fig)


# ─────────────────────────────────────────────
# Quick test / standalone demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("IntelliSignal DSP Pipeline — Synthetic Signal Test\n")

    # --- Generate a synthetic chirp signal (like base paper) ---
    FS = 2.048e6          # 2.048 MHz sample rate
    T = 1_300_000 / FS    # same length as WASD dataset
    t = np.linspace(0, T, int(T * FS), endpoint=False)

    # Chirp: frequency sweeps 100 kHz → 900 kHz
    from scipy.signal import chirp as sp_chirp
    samples = sp_chirp(t, f0=1e5, f1=9e5, t1=T, method="linear").astype(np.float32)

    sig = SignalData(samples=samples, sample_rate=FS, label="Chirp")

    # --- Run pipeline ---
    pipeline = DSPPipeline()
    result = pipeline.run(sig)

    print(f"FFT      : {len(result.fft_freqs)} frequency bins")
    print(f"STFT     : {result.stft_magnitude.shape} (freq × time)")
    print(f"Wavelet  : {len(result.wavelet_coeffs)} levels ({result.wavelet_name})")
    print(f"Spectro  : {result.spectrogram_image.shape} (ready for model input)")
    print("\nDSP pipeline completed successfully.")

    # --- Optional: save spectrogram to disk ---
    from PIL import Image
    img = Image.fromarray(result.spectrogram_image, mode="L")
    img.save("test_spectrogram_400x400.png")
    print("Saved: test_spectrogram_400x400.png")
