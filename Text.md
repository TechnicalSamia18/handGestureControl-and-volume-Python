# Hand Gesture Control – Mouse & System Volume

Real-time hand gesture control for mouse and system volume using Python, OpenCV, MediaPipe and pycaw.

- **Left hand** → virtual mouse (move + click)  
- **Right hand** → system volume (mute, unmute, volume up/down)

This project turns your webcam into a simple AI-powered input device using deep-learning–based hand tracking from MediaPipe.

---

## ✨ Features

### 🖱️ Left Hand – Virtual Mouse

- Move your left hand → moves the cursor  
  - Uses the **middle finger MCP joint** as the cursor anchor.
- Bend your **index finger** down (tip goes below PIP joint) → **left-click**
- Debounced click logic:  
  - One bend = **one** click  
  - Holding your finger bent does **not** spam clicks

### 🔊 Right Hand – System Volume Control

- **Open palm (5 fingers extended)** → **Mute**
- **Fist (fingers closed)** → **Unmute**
- **Pinch (thumb + index finger)** → Adjusts volume  
  - Pinch distance is mapped to **0–100% volume**  
  - Smoothing so tiny hand shakes don’t make the volume jump

---

## 🧠 How It Works (Short)

- Uses **MediaPipe Hands** to detect 3D landmarks of each hand in real time.
- Simple geometric rules (distances, relative positions) convert landmarks into gestures:
  - Bent finger → click  
  - Palm vs fist → mute / unmute  
  - Thumb–index distance → volume level
- Uses:
  - `pyautogui` for mouse movement & click
  - `pycaw` for Windows audio endpoint control

This is an **AI-powered HCI project** built on top of a pre-trained deep learning model (MediaPipe).

---

## 🧩 Tech Stack

- **Language:** Python 3.11 (recommended)
- **Libraries:**
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `pyautogui`
  - `pycaw`
  - `comtypes` (dependency of pycaw)

> ⚠️ Note: `pycaw` is Windows-only, so the full project (with volume control) currently targets **Windows**.

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
