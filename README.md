WASD: Wireless Anomaly Signal Dataset
WASD is a wireless anomaly dataset for spectrum-level anomaly detection in fixed urban environments. It combines measured LTE/5G signals with synthetic anomaly overlays (tone, chirp, pulse) for detection and localization tasks.

Paper
IEEE Access (2024): https://ieeexplore.ieee.org/document/10813361
DOI: 10.1109/ACCESS.2024.3521946
Related Works
Kim et al., "Spectrum Anomaly Detection Using Deep Neural Networks: A Wireless Signal Perspective", IEEE Access (2025): https://doi.org/10.1109/ACCESS.2025.3603216
Kim et al., "Anomaly Detection for Wireless Cellular Communication based on Synthetic Anomaly", IEEE Access (2025): https://doi.org/10.1109/ACCESS.2025.3584113
Highlights
Real-world measurements over 19 LTE/5G bands.
Synthetic anomalies: tone, chirp, pulse.
Channel/fading-aware synthesis following practical assumptions.
85,500 spectrogram samples total (19 bands x 4,500 samples/band).
Bounding-box labels for anomaly localization.
Public interactive viewer: https://remilab.ai/wasd/
Band Examples Data Synthesis

Quick Visualization (Local)
Generate baseline figures from local sample assets (5G IQ zip, LTE IQ bins, anomaly npy + labels):

python3 scripts/wasd_basic_visualization.py
The basic 5G/LTE STFT figures use the same spectrum crop as the REMI Lab viewer:

Full FFT: 2048 bins (±30.72 MHz)
Guard band removed: 224 bins (6.72 MHz) on each side
Visible spectrum after crop: 1600 bins (48.00 MHz, about ±24.00 MHz)
Generated files:

images/basic_visualization/wasd_5g_iq_stft_basic.png
images/basic_visualization/wasd_lte_iq_stft_basic.png
images/basic_visualization/wasd_anomaly_overlay_basic.png
images/basic_visualization/wasd_label_summary_basic.png
images/basic_visualization/basic_visualization_manifest.json
5G IQ STFT (Basic) LTE IQ STFT (Basic) Generated Anomaly Overlay (Basic) Dataset Label Summary (Basic)

Interactive Viewer (REMI Lab)
Main viewer: https://remilab.ai/wasd/
Large viewer: https://remilab.ai/wasd/viewer-large/
Repository Contents
.
|-- WASD_data_loader.ipynb
|-- WASD_abnormal_signal_generation.ipynb
|-- images/
|   `-- basic_visualization/
|       |-- wasd_5g_iq_stft_basic.png
|       |-- wasd_lte_iq_stft_basic.png
|       |-- wasd_anomaly_overlay_basic.png
|       |-- wasd_label_summary_basic.png
|       `-- basic_visualization_manifest.json
|-- scripts/
|   `-- wasd_basic_visualization.py
|-- LICENSE
`-- README.md
Dataset Packaging (Full Release)
Dataset/
|-- IQ data.zip
|   `-- <band_name>/
|       |-- bin/
|       |   `-- IQ_<timestamp>.bin
|       `-- IQ_<timestamp>.json
`-- npy data.zip
    |-- Abnormal/
    |   |-- Abnormal_spectrogram.npy
    |   `-- label/Spectrum_label.csv
    `-- Normal/
        `-- Normal_spectrogram.npy
Data Notes
Each .npy file stores spectrogram matrices.
Spectrum_label.csv can contain multiple bounding boxes per spectrogram.
The number of bounding boxes can exceed the number of spectrogram samples.
IQ Calibration Note
iq_level_offset: correction offset applied to raw ADC-domain IQ power.
reference_level: reference level used for absolute calibration.
Typical conversion path: apply iq_level_offset to convert scale, then convert to dBm as defined by measurement metadata.
Notebooks
WASD_data_loader.ipynb
Example loading of IQ binary, spectrogram .npy, and label .csv.
WASD_abnormal_signal_generation.ipynb
Example synthesis of tone/chirp/pulse anomaly signals and time-frequency visualization.
Access Links
IEEE DataPort (split dataset for quick evaluation):

https://ieee-dataport.org/open-access/wasd-wireless-anomaly-signal-dataset
Google Drive (full dataset chunks):

Chunk 1
Chunk 2
Chunk 3
Chunk 4
Chunk 5
Chunk 6
Chunk 7
Chunk 8
Chunk 9
Chunk 10
Chunk 11
Chunk 12
Chunk 13
Chunk 14
Chunk 15
Chunk 16
Authors
Jinha Kim: jinha.kim@o.cnu.ac.kr
Hyeongwoo Kim: hyeongwoo.kim@o.cnu.ac.kr
Byungkwan Kim: byungkwan.kim@cnu.ac.kr
Acknowledgement
This work was supported by MSIT/IITP through the ICT R&D program (RS-2023-00229541).
