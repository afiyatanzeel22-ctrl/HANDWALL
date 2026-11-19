# ✋ HandWall – Virtual Neon Handwriting

HandWall is a computer vision-based virtual handwriting application that allows users to write in the air using simple hand gestures through a webcam.

Built using **Python, OpenCV, NumPy, and MediaPipe**, the application tracks hand landmarks in real time and converts index finger movements into glowing neon strokes on the screen.

## ✨ Features

- **Gesture-Based Writing:** Pinch your thumb and index finger to start writing.
- **Index Finger Tracking:** Move your index finger to draw on the screen.
- **Gesture-Based Control:** Release the pinch to stop writing.
- **Neon Glow Effect:** Creates attractive pink neon strokes with a white inner highlight.
- **Smooth Drawing:** Uses finger movement smoothing to reduce unwanted shaking.
- **Real-Time Hand Tracking:** Detects hand landmarks using MediaPipe.
- **Clear Canvas:** Press `C` to erase all drawings.
- **Easy Exit:** Press `Q` to close the application.

## 🛠️ Technologies Used

- Python 3.13
- OpenCV
- MediaPipe Tasks API
- NumPy
- Math

## 🚀 How It Works

1. The webcam captures real-time video.
2. MediaPipe detects and tracks hand landmarks.
3. The distance between the thumb tip and index fingertip is calculated.
4. Pinching the two fingers activates the virtual pen.
5. Moving the index finger creates neon strokes.
6. Releasing the pinch stops drawing while preserving previous strokes.

## ⚙️ Installation

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy
```

Download the MediaPipe hand landmark model:

[Hand Landmarker Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

Place `hand_landmarker.task` in the same folder as `handwall.py`.

## ▶️ Run the Project

```bash
python handwall.py
```

Allow webcam access when prompted.

## 🎮 Controls

| Gesture / Key | Action |
|---|---|
| Pinch thumb and index | Start writing |
| Move index finger | Draw |
| Release pinch | Stop writing |
| C | Clear canvas |
| Q | Quit application |

## 📌 Future Improvements

- Multiple neon colours.
- Adjustable brush thickness.
- Save handwritten drawings as images.
- Gesture-based colour selection.
- Support for multiple hand gestures.

## 👩‍💻 Project Overview

HandWall demonstrates the practical application of computer vision and hand gesture recognition to create a touch-free digital writing experience.

**Write in the air. Create without touching the screen!**
