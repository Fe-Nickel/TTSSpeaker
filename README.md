# TTSSpeaker

A GUI frontend for IndexTTS2 with voice cloning, emotion control, and virtual audio device output.

## Features

- 🎤 Voice cloning
- 😊 Emotion control
- 🔊 Virtual audio device output
- 🎵 Voice extraction from audio files
- 🖥️ Easy-to-use GUI built with CustomTkinter

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/Fe-Nickel/TTSSpeaker.git
cd TTSSpeaker
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Install the appropriate version of PyTorch according to your CUDA or CPU environment.

### 3. Clone the IndexTTS repository

```bash
git clone https://github.com/index-tts/index-tts.git index-tts-main
```

Place the repository in the project root directory, then install it:

```bash
pip install -e index-tts-main
```

### 4. Replace the patched file

Copy

```
patches/infer_v2.py
```

to

```
index-tts-main/indextts/infer_v2.py
```

and overwrite the original file.

### 5. Download the IndexTTS-2 model

Follow the official IndexTTS instructions to download the model and place it in the appropriate directory.

### 6. Run

```bash
python src/main.py
```

---

## Notes

This project only provides the graphical user interface and related functionality.

IndexTTS source code and model files are **not included** in this repository and must be downloaded separately.
