# 🎙️ Hidden Markov Model in Speech Recognition | Interactive Visualizer

An interactive mathematical simulation and real-time audio visualization web application demonstrating **Hidden Markov Models (HMM)**, **Viterbi Trellis Decoding**, **Baum-Welch Expectation-Maximization Training**, and **MFCC Feature Extraction** for speech recognition.

🚀 **Live Demo:** https://Mithileshkose27.github.io/Hidden-Markov-Model/

---

## 📌 Overview

Speech recognition systems rely on probabilistic sequence modeling to convert acoustic waveforms into textual sequences. This web application provides a full interactive visual toolkit that bridges theoretical mathematics with real-time audio signal processing.

Users can record live speech or simulate acoustic inputs, extract Mel-Frequency Cepstral Coefficients (MFCC), inspect Hidden Markov Model topologies, trace the Viterbi decoding path through dynamic programming trellis matrices, and visualize Baum-Welch model parameter updates.

---

## ✨ Features

- 🎤 **Real-Time Audio Capture & Spectrogram**: Capture live audio via standard Web Audio API and generate live frequency spectrograms.
- 🎛️ **MFCC Feature Extraction**: Extract 13 Mel-Frequency Cepstral Coefficients representing the spectral envelope of human vocal speech.
- 🕸️ **Interactive HMM Graph Topology**: Visualize hidden states, transition probabilities \( A_{ij} \), and continuous Gaussian emission distributions \( B_j(x) \).
- 🧩 **Viterbi Trellis Decoder**: Compute the optimal state sequence (maximum likelihood path) using dynamic programming trellis matrices.
- 🔁 **Baum-Welch Model Trainer**: Interactive Expectation-Maximization (EM) algorithm trainer visualizing forward-backward variables (\(\alpha\), \(\beta\)) and parameter convergence.
- 🏆 **Speech Word Recognition**: Evaluate acoustic likelihoods across multiple trained word models (e.g., *"Yes"*, *"No"*, *"Stop"*, *"Go"*) to classify spoken utterances.
- 🐍 **Python Reference Implementation**: Includes clean Python backend modules (`python_src/`) featuring NumPy/SciPy algorithms for offline evaluation and benchmark comparisons.

---

## 🧮 Mathematical Foundations

### 1. Hidden Markov Model Formulation
An HMM is defined by the tuple \(\lambda = (A, B, \pi)\):
- **State Transition Matrix \(A\)**: Probability of transitioning from state \(i\) to state \(j\), \(a_{ij} = P(q_t = S_j \mid q_{t-1} = S_i)\).
- **Emission Probability Density Function \(B\)**: Continuous multivariate Gaussian probability density for state \(j\):
  $$b_j(O_t) = \mathcal{N}(O_t; \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{D/2}|\Sigma_j|^{1/2}} \exp\left(-\frac{1}{2}(O_t - \mu_j)^T \Sigma_j^{-1} (O_t - \mu_j)\right)$$
- **Initial State Distribution \(\pi\)**: \(\pi_i = P(q_1 = S_i)\).

### 2. Viterbi Path Search (Decoding)
The maximum likelihood state sequence \(V_{t}(j)\) at frame \(t\) for state \(j\) is defined recursively:
$$V_{t}(j) = \max_{i} \left[ V_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(O_t)$$
Taking the natural logarithm prevents numerical underflow during long temporal sequences:
$$v_t(j) = \max_i \left[ v_{t-1}(i) + \ln a_{ij} \right] + \ln b_j(O_t)$$

### 3. Baum-Welch Expectation-Maximization (Training)
Model parameters are re-estimated using forward probabilities \(\alpha_t(i)\) and backward probabilities \(\beta_t(i)\):
- **Forward Variable**: \(\alpha_t(i) = P(O_1, O_2, \dots, O_t, q_t = S_i \mid \lambda)\)
- **Backward Variable**: \(\beta_t(i) = P(O_{t+1}, O_{t+2}, \dots, O_T \mid q_t = S_i, \lambda)\)
- **Posterior Probability**:
  $$\gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^{N} \alpha_t(j)\beta_t(j)}$$

---

## 🏗️ Tech Stack

- **Frontend Core**: React 18, Vite 5, JavaScript (ES2023)
- **Styling & UI**: Modern Vanilla CSS, Glassmorphism, CSS Custom Properties, Dark Mode
- **Icons**: Lucide React
- **Audio Processing**: Web Audio API (AudioContext, AnalyserNode, FFT)
- **Graphing & Visuals**: HTML5 Canvas API, SVG vector paths
- **Deployment**: GitHub Actions, GitHub Pages

---

## 📂 Repository Structure

```
Hidden-Markov-Model/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Pages automated CI/CD pipeline
├── python_src/
│   ├── demo.py                 # CLI demonstration script
│   ├── hmm_core.py             # Python HMM implementation (NumPy)
│   ├── mfcc_extractor.py       # MFCC feature extraction pipeline
│   ├── speech_classifier.py    # Speech classification module
│   └── synthetic_speech_generator.py # Synthetic audio dataset generator
├── src/
│   ├── components/
│   │   ├── AudioRecorder.jsx    # Audio capture component
│   │   ├── BaumWelchTrainer.jsx # Interactive EM trainer view
│   │   ├── HmmGraph.jsx         # Interactive state transition graph
│   │   ├── RecognitionResult.jsx# Classification results card
│   │   ├── SpectrogramView.jsx  # Spectrogram & MFCC canvas
│   │   └── ViterbiTrellis.jsx   # Dynamic programming trellis grid
│   ├── utils/
│   │   ├── webAudio.js          # Web Audio API & MFCC calculation
│   │   └── webHmm.js            # In-browser HMM & Viterbi engine
│   ├── App.jsx                  # Main application component
│   ├── index.css                # Design system & CSS variables
│   └── main.jsx                 # Application entry point
├── index.html
├── package.json
└── vite.config.js
```

---

## 🚀 Quick Start (Local Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mithileshkose27/Hidden-Markov-Model.git
   cd Hidden-Markov-Model
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start local development server**:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000/` in your web browser.

4. **Build for production**:
   ```bash
   npm run build
   ```

---

## 📄 Deployment to GitHub Pages

This repository includes a pre-configured GitHub Actions workflow (`.github/workflows/deploy.yml`). 

To enable deployment:
1. Navigate to your repository **Settings** on GitHub.
2. Select **Pages** from the sidebar menu.
3. Under **Build and deployment -> Source**, select **GitHub Actions**.
4. Push your changes to the `main` branch. GitHub Actions will automatically build and publish the site.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
