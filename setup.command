#!/bin/bash
# Double-click this once to install everything Wave to Elon needs.
cd "$(dirname "$0")" || exit 1

echo "=== Wave to Elon setup ==="

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Get it from https://www.python.org/downloads/ then run this again."
  read -r -p "Press Enter to close..."
  exit 1
fi

echo "-> Creating virtual environment (.venv)"
python3 -m venv .venv || { read -r -p "Failed. Press Enter to close..."; exit 1; }

echo "-> Installing libraries (MediaPipe, OpenCV, PySerial, NumPy)"
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt || { read -r -p "Install failed. Press Enter to close..."; exit 1; }

echo "-> Downloading the hand-tracking model"
mkdir -p models
if [ ! -s models/gesture_recognizer.task ]; then
  curl -fL -o models/gesture_recognizer.task \
    https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task \
    || { read -r -p "Download failed. Press Enter to close..."; exit 1; }
fi

chmod +x run.command
echo
echo "All set! Upload arduino/wave_servo/wave_servo.ino to the Uno with the Arduino IDE,"
echo "then double-click run.command to start."
read -r -p "Press Enter to close..."
