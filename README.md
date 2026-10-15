# Wave to Elon

Wave at the Mac webcam → MediaPipe detects the wave → Python sends `W` over USB → Arduino Uno waves a cardboard hand on a servo.

## Setup (once)
Double-click **`setup.command`**. It installs all the Python libraries and downloads the hand-tracking model.
(If macOS blocks it: right-click → Open.)

Then open `arduino/wave_servo/wave_servo.ino` in the Arduino IDE, pick **Arduino Uno** + its port, and Upload.
Close the IDE's Serial Monitor afterwards (it blocks Python from using the port).

## Run
Double-click **`run.command`**.
Keys: `f` fullscreen, `t` test the servo, `q`/Esc quit.

## Gestures (up to 4 hands at once)
| Gesture | Effect |
|---|---|
| 👋 Wave (open hand, left-right-left) | Servo waves + hearts + "Elon is waving back!" |
| 🫶 Two-hand heart | Floating hearts + "RMC LOVES YOU TOO" |
| 👍 Thumbs up | Confetti rain |
| ✌️ Peace | Sparkles |
| ☝️ Point up | Rocket launch |
| 🤘 Rock on  | Rainbow burst + confetti + "ROCK ON!" |

Gestures are read by MediaPipe's trained Gesture Recognizer and must be held steady for a moment
before they fire. If they're too picky or too loose, change `MIN_SCORE` / `VOTE_NEEDED` at the top of `wave_to_elon.py`.

## Hardware
- Servo brown/black → GND, red → 5V, orange/yellow → pin 9
- Tape/glue the cardboard hand to the servo horn

Tuning is at the top of `wave_to_elon.py` (`SWING_DISTANCE`, `SWINGS_NEEDED`, `COOLDOWN`)
and at the top of the `.ino` (angles, number of waves, speed).
