# BrainComputerInterface

Framework for multimodal Brain-Computer Interface (BCI), focusing on EEG-based emotion recognition using CNN and LSTM architectures. Developed at the [BCI Lab, Nazarbayev University](https://nu.edu.kz) under the supervision of Prof. Min-Ho Lee.

This repository implements the signal processing and classification pipeline used in our research, built on the [EAV dataset](https://github.com/nubcico/EAV) — a multimodal EEG-Audio-Video corpus for emotion recognition in conversational contexts.

---

## Overview

The project addresses the challenge of decoding emotional states from neurophysiological signals. Starting from raw EEG recordings, the pipeline covers noise reduction, neural component filtering, feature extraction, and classification via deep learning models trained in MATLAB.

---

## Repository Structure

```
BrainComputerInterface/
├── PreProcessing/       # EEG denoising, bandpass filtering, ICA-based artifact removal,
│                        # neural component extraction, feature construction
├── CNN/                 # Convolutional neural network models and 3D input procedures
│                        # for spatial-spectral EEG feature classification
└── LSTM/                # LSTM and CNN-LSTM models for temporal sequence modeling
                         # and multimodal fusion classification
```

---

## Dataset

This codebase is designed to work with the **EAV Dataset** — a 30-channel EEG + Audio + Video dataset collected from 42 participants across 5 emotion classes (Neutral, Anger, Happiness, Sadness, Calmness).

**→ [nubcico/EAV: EEG-Audio-Video Dataset for Emotion Recognition in Conversations](https://github.com/nubcico/EAV)**

The dataset contains:
- **EEG**: 200 trials × 10,000 time points × 30 channels per subject (`.mat`)
- **Audio**: 100 speaking-task clips per subject, 20s each (`.wav`)
- **Video**: 200 clips per subject, 20s at 30fps (`.mp4`)

Download the raw data and pre-extracted features from [Zenodo](https://doi.org/10.5281/zenodo.10205702).

---

## Getting Started

### Requirements

- MATLAB R2021b or later
- Signal Processing Toolbox
- Deep Learning Toolbox

### Installation

```bash
git clone https://github.com/nubcico/BrainComputerInterface.git
cd BrainComputerInterface
```

Download the EAV dataset from Zenodo and place it in a local directory. Update the data path variables in the preprocessing scripts accordingly.

### Pipeline

1. **PreProcessing** — Run the scripts in `PreProcessing/` to denoise raw EEG signals, apply bandpass filtering, extract neural components, and construct feature representations.
2. **CNN** — Use scripts in `CNN/` to train convolutional models on the extracted 3D feature inputs.
3. **LSTM** — Use scripts in `LSTM/` to train sequence models or the combined CNN-LSTM architecture for temporal and multimodal classification.

---

## Publications

Work produced using this framework:

1.	S. Bralina; A. Yazici; C. Guan; M. H. Lee. “Adaptive Bottleneck Transformer for Multimodal EEG, Audio, and Vision Fusion.” Expert Systems with Applications, 2026, 312, Article 131487. DOI: 10.1016/j.eswa.2026.131487. Indexing information: Scopus Q1, Top 10%. Published.
2.	Baimukanova, Z.; Saparbekov, Y.; Ha, H.; Min-Ho Lee*. “Evidence-Grounded LLM Summarization for Actionable Student Feedback Analysis.” Information, 2026, 17(4), Article 351. DOI: 10.3390/info17040351. Indexing information: Scopus Q1. Published.
3.	Vadim Atlassov; Isabella Schlattner; Yelzhas Omarov; Hyun-Jin Ju; Min-Ho Lee*. “Controllable Diffusion-Based Image Generation for Failure Diagnosis of Reinforced Concrete Beam–Column Joints.” Journal of Building Engineering, 2026. DOI: N/A. Indexing information: Scopus Q1, Top 1%, Accepted.
4.	Z. Kabidenova; B. Abibullaev; A. Yazici; M.-H. Lee. “EEG2Face: Neural–Peripheral EEG Face Reconstruction.” Expert Systems with Applications, 2026. DOI: N/A. Indexing information: Scopus Q1, Top 10%. Under revision.
5.	M. Kudaibergenova; A. Yazici; A. Shomanov; M.-H. Lee. “Multimodal Joint Representations of EEG and Audio-Vision for Zero-Shot Learning.” Biomedical Signal Processing and Control, 2026. DOI: N/A. Indexing information: Scopus Q1, Top 10%. Under revision.


---

## Contact

**Prof. Min-Ho Lee** — [minho.lee@nu.edu.kz](mailto:minho.lee@nu.edu.kz)  
BCI Lab, School of Engineering and Digital Sciences, Nazarbayev University

---

## License

This project is licensed under the GPL-3.0 License. See [`LICENSE`](https://github.com/nubcico/BrainComputerInterface/blob/main/LICENSE) for details.
