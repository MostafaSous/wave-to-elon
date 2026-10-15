"""Wave to Elon: wave at the webcam and a cardboard hand on a servo waves back.

Run:  ./run.command                                  (or double-click it in Finder)
      .venv/bin/python wave_to_elon.py --port /dev/cu.usbmodem1101
      .venv/bin/python wave_to_elon.py --no-serial   (camera only, no Arduino)
Keys: q / Esc = quit, f = fullscreen, t = test the servo without waving

Gestures (up to 4 hands at once):
  wave (open hand, left-right-left)  -> servo waves + hearts
  two-hand heart                     -> floating hearts
  rock on / Spider-Man (🤘 or 🤟)      -> colour burst
  thumbs up                          -> confetti rain
  peace sign                         -> sparkles
  point up (index finger)            -> rocket launch
"""

import argparse
import glob
import math
import random
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.components.processors import ClassifierOptions

MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"
MAX_HANDS = 4

# Wave detection tuning
SWING_DISTANCE = 0.6    # how far the hand must move before a direction change counts, in palm lengths
SWINGS_NEEDED = 3       # left-right-left = a wave
SWING_WINDOW = 1.5      # seconds the swings must happen within
COOLDOWN = 4.0          # seconds between servo waves (servo needs time to finish)

# Gesture accuracy tuning
MIN_SCORE = 0.65        # how sure the model must be about a gesture in one frame
VOTE_FRAMES = 8         # look at the last this-many frames of each hand...
VOTE_NEEDED = 6         # ...and only accept a gesture seen in at least this many of them
GESTURE_GRACE = 0.3     # a gesture can drop out for this long without "ending"

# MediaPipe's built-in gesture names -> ours (Closed_Fist / Thumb_Down are ignored on purpose)
MODEL_GESTURES = {
    "ILoveYou": "rock",   # 🤟 Spider-Man sign
    "Thumb_Up": "thumbs_up",
    "Victory": "peace",
    "Pointing_Up": "point",
}

# Colours are BGR
WHITE = (255, 255, 255)
SKY = (255, 190, 60)
PINKS = [(180, 105, 255), (147, 20, 255), (203, 192, 255), (60, 20, 220)]
GOLDS = [(0, 215, 255), (80, 240, 255), (255, 255, 255)]
RAINBOW = [(255, 80, 200), (255, 255, 0), (0, 215, 255), (180, 105, 255), (0, 140, 255), (255, 190, 60)]

GESTURE_LABELS = {
    "heart": ("RMC LOVES YOU TOO", (180, 105, 255)),
    "thumbs_up": ("THUMBS UP!", (0, 215, 255)),
    "peace": ("PEACE OUT", (255, 255, 0)),
    "point": ("TO MARS!", (0, 140, 255)),
    "rock": ("ROCK ON!", (255, 80, 200)),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ---------------------------------------------------------------- detection

class WaveDetector:
    """Counts left/right direction changes of the hand; enough of them quickly = a wave."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = 0
        self.extreme = None
        self.swings = deque()

    def update(self, x, now, threshold):
        if self.extreme is None:
            self.extreme = x
        elif self.direction == 0:
            if abs(x - self.extreme) > threshold:
                self.direction = 1 if x > self.extreme else -1
                self.extreme = x
        elif (x - self.extreme) * self.direction > 0:
            self.extreme = x  # still moving the same way
        elif abs(x - self.extreme) > threshold:
            self.swings.append(now)  # reversed direction
            self.direction = -self.direction
            self.extreme = x

        while self.swings and now - self.swings[0] > SWING_WINDOW:
            self.swings.popleft()
        return len(self.swings) >= SWINGS_NEEDED

    @property
    def progress(self):
        return min(len(self.swings) / SWINGS_NEEDED, 1.0)


def palm_size(P):
    return np.linalg.norm(P[9] - P[0])


def fingers_extended(P):
    """[index, middle, ring, pinky]: True if the fingertip is clearly farther from the wrist than its middle joint."""
    def d(a, b):
        return np.linalg.norm(P[a] - P[b])
    return [d(0, tip) > d(0, pip) * 1.1 for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]


def is_waving_pose(P):
    """Open hand held up: at least 3 fingers stretched out and the fingers above the wrist."""
    return sum(fingers_extended(P)) >= 3 and P[12][1] < P[0][1] - 0.5 * palm_size(P)


def is_rock_sign(P):
    """🤘 index and pinky up, middle and ring folded (MediaPipe's model has no class for this one)."""
    return fingers_extended(P) == [True, False, False, True] and P[8][1] < P[0][1] and P[20][1] < P[0][1]


def gesture_anchor(name, P):
    """Where on the hand a gesture's effect should come from."""
    if name == "point":
        return P[8]
    if name == "thumbs_up":
        return P[4]
    if name == "peace":
        return (P[8] + P[12]) / 2
    return P[9]


def two_hand_heart(A, B):
    """Index tips touching, thumb tips touching, with a gap between them = a heart. Returns its centre or None."""
    size = (palm_size(A) + palm_size(B)) / 2
    index_gap = np.linalg.norm(A[8] - B[8])
    thumb_gap = np.linalg.norm(A[4] - B[4])
    index_y = (A[8][1] + B[8][1]) / 2
    thumb_y = (A[4][1] + B[4][1]) / 2
    if index_gap < 0.45 * size and thumb_gap < 0.45 * size and thumb_y - index_y > 0.6 * size:
        return (A[8] + B[8] + A[4] + B[4]) / 4
    return None


class HandTrack:
    """One hand followed across frames, so each hand gets its own wave detector and gesture votes."""
    _next_id = 0

    def __init__(self, pos, now):
        self.id = HandTrack._next_id
        HandTrack._next_id += 1
        self.pos = pos
        self.last_seen = now
        self.wave = WaveDetector()
        self.votes = deque(maxlen=VOTE_FRAMES)

    def stable_gesture(self):
        """A gesture only counts if the model agreed on it in most of the recent frames."""
        if len(self.votes) < VOTE_NEEDED:
            return None
        name, count = Counter(self.votes).most_common(1)[0]
        return name if name and count >= VOTE_NEEDED else None


def match_tracks(tracks, palms, now, max_dist):
    """Pair this frame's hands with existing tracks by nearest position; start new tracks for new hands."""
    pairs = sorted((np.linalg.norm(t.pos - p), ti, pi) for ti, t in enumerate(tracks) for pi, p in enumerate(palms))
    used_t, matched = set(), [None] * len(palms)
    for dist, ti, pi in pairs:
        if dist > max_dist:
            break
        if ti in used_t or matched[pi] is not None:
            continue
        used_t.add(ti)
        matched[pi] = tracks[ti]
    for pi, palm in enumerate(palms):
        if matched[pi] is None:
            matched[pi] = HandTrack(palm, now)
            tracks.append(matched[pi])
        matched[pi].pos = palm
        matched[pi].last_seen = now
    tracks[:] = [t for t in tracks if now - t.last_seen < 0.5]
    return matched


class GestureTracker:
    """Keeps gestures alive through brief dropouts; returns the active ones and which just started.
    Keys are (gesture name, hand id) so several hands can do the same gesture at once."""

    def __init__(self):
        self.last = {}
        self.started = {}

    def update(self, detected, now):
        for key in detected:
            self.last[key] = now
        new = set()
        for key in detected:
            if key not in self.started:
                self.started[key] = now
                new.add(key)
        for key in list(self.started):
            if now - self.last[key] > GESTURE_GRACE:
                del self.started[key]
        return new

    def current_label(self):
        """The most recently started gesture that's still going, with its start time."""
        if not self.started:
            return None, None
        key = max(self.started, key=self.started.get)
        return key[0], self.started[key]


# ---------------------------------------------------------------- effects

_t = np.linspace(0, 2 * np.pi, 40)
HEART_SHAPE = np.stack([16 * np.sin(_t) ** 3,
                        -(13 * np.cos(_t) - 5 * np.cos(2 * _t) - 2 * np.cos(3 * _t) - np.cos(4 * _t))], 1) / 16


def draw_heart(img, x, y, size, color):
    pts = (HEART_SHAPE * size + (x, y)).astype(np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.circle(img, (int(x - size * 0.45), int(y - size * 0.35)), max(1, int(size * 0.15)), WHITE, -1, cv2.LINE_AA)


def draw_star(img, x, y, size, angle, color):
    a = angle + np.arange(10) * np.pi / 5
    r = np.where(np.arange(10) % 2 == 0, size, size * 0.45)
    pts = np.stack([x + r * np.sin(a), y - r * np.cos(a)], 1).astype(np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def draw_confetti(img, x, y, size, angle, color):
    box = cv2.boxPoints(((x, y), (size, size * 0.5), math.degrees(angle))).astype(np.int32)
    cv2.fillPoly(img, [box], color, cv2.LINE_AA)


def draw_rocket(img, x, y, s, t):
    def poly(points, color):
        cv2.fillPoly(img, [np.array([(x + px * s, y + py * s) for px, py in points], np.int32)], color, cv2.LINE_AA)

    flick = 1 + 0.35 * math.sin(t * 40)
    poly([(-0.7, 1.5), (0.7, 1.5), (0, 1.5 + 2.6 * flick)], (0, 120, 255))       # flame
    poly([(-0.4, 1.5), (0.4, 1.5), (0, 1.5 + 1.5 * flick)], (80, 240, 255))
    poly([(-1, 0.2), (-1.9, 2), (-1, 1.5)], (40, 40, 220))                       # fins
    poly([(1, 0.2), (1.9, 2), (1, 1.5)], (40, 40, 220))
    poly([(0, -3.2), (0.6, -2.4), (1, -1.2), (1, 1.5), (-1, 1.5), (-1, -1.2), (-0.6, -2.4)], (235, 235, 235))
    poly([(0, -3.2), (0.6, -2.4), (0.8, -1.9), (-0.8, -1.9), (-0.6, -2.4)], (40, 40, 220))  # nose
    cv2.circle(img, (int(x), int(y - 0.5 * s)), int(0.5 * s), (90, 90, 90), -1, cv2.LINE_AA)   # window
    cv2.circle(img, (int(x), int(y - 0.5 * s)), int(0.35 * s), (230, 170, 60), -1, cv2.LINE_AA)


class Particle:
    def __init__(self, kind, x, y, vx, vy, life, size, color, gravity=0.0, spin=0.0, wobble=0.0):
        self.kind, self.x, self.y, self.vx, self.vy = kind, x, y, vx, vy
        self.life, self.size, self.color = life, size, color
        self.gravity, self.spin, self.wobble = gravity, spin, wobble
        self.age = 0.0
        self.angle = random.uniform(0, 2 * math.pi)
        self.phase = random.uniform(0, 2 * math.pi)


class Effects:
    MAX_PARTICLES = 900

    def __init__(self):
        self.particles = []

    def add(self, *args, **kwargs):
        if len(self.particles) < self.MAX_PARTICLES:
            self.particles.append(Particle(*args, **kwargs))

    def hearts(self, x, y, n):
        for _ in range(n):
            ang = random.uniform(-math.pi * 0.85, -math.pi * 0.15)
            speed = random.uniform(120, 340)
            self.add("heart", x, y, math.cos(ang) * speed, math.sin(ang) * speed, random.uniform(1.3, 2.3),
                     random.uniform(14, 34), random.choice(PINKS), gravity=-30, wobble=random.uniform(8, 25))

    def stars(self, x, y, n, colors):
        for _ in range(n):
            ang = random.uniform(0, 2 * math.pi)
            speed = random.uniform(150, 420)
            self.add("star", x, y, math.cos(ang) * speed, math.sin(ang) * speed, random.uniform(0.7, 1.4),
                     random.uniform(8, 20), random.choice(colors), gravity=250, spin=random.uniform(-6, 6))

    def confetti_rain(self, width, n):
        for _ in range(n):
            self.add("confetti", random.uniform(0, width), random.uniform(-120, 0), random.uniform(-60, 60),
                     random.uniform(150, 350), 4.0, random.uniform(10, 18), random.choice(RAINBOW + GOLDS),
                     gravity=80, spin=random.uniform(-8, 8), wobble=random.uniform(5, 25))

    def confetti_burst(self, x, y, n):
        for _ in range(n):
            ang = random.uniform(-math.pi * 0.9, -math.pi * 0.1)
            speed = random.uniform(300, 700)
            self.add("confetti", x, y, math.cos(ang) * speed, math.sin(ang) * speed, 2.5,
                     random.uniform(10, 18), random.choice(RAINBOW), gravity=700, spin=random.uniform(-10, 10))

    def rocket(self, x, y):
        self.add("rocket", x, y, random.uniform(-20, 20), -60, 4.0, random.uniform(14, 20), None, gravity=-550)

    def update(self, dt, height):
        smoke = []
        alive = []
        for p in self.particles:
            p.age += dt
            p.vy += p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            if p.kind == "rocket":
                for _ in range(2):
                    smoke.append(Particle("smoke", p.x + random.uniform(-p.size * 0.4, p.size * 0.4),
                                          p.y + p.size * 3.5, random.uniform(-40, 40), random.uniform(40, 120),
                                          random.uniform(0.5, 0.9), p.size * 0.6, None))
            if p.age < p.life and -300 < p.y < height + 300:
                alive.append(p)
        self.particles = (smoke + alive)[-self.MAX_PARTICLES:]

    def draw(self, img, now):
        for p in self.particles:
            f = p.age / p.life
            x = p.x + math.sin(p.age * 4 + p.phase) * p.wobble
            pop = min(1.0, p.age / 0.15)
            if p.kind == "heart":
                draw_heart(img, x, p.y, p.size * pop * (1 - f ** 3), p.color)
            elif p.kind == "star":
                draw_star(img, x, p.y, p.size * (1 - f ** 2), p.angle, p.color)
            elif p.kind == "confetti":
                draw_confetti(img, x, p.y, p.size, p.angle, p.color)
            elif p.kind == "smoke":
                shade = int(170 + 60 * f)
                cv2.circle(img, (int(x), int(p.y)), max(1, int(p.size * (1 - f))), (shade, shade, shade), -1,
                           cv2.LINE_AA)
            elif p.kind == "rocket":
                draw_rocket(img, p.x, p.y, p.size, now)


def emit_count(rate, dt):
    return int(rate * dt + random.random())


# ---------------------------------------------------------------- arduino

def find_arduino_port():
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


class Arduino:
    def __init__(self, port):
        self.ser = None
        self.status = "No Arduino (camera only)"
        if port is None:
            return
        try:
            import serial
            self.ser = serial.Serial(port, 9600, timeout=0)
            time.sleep(2)  # the Uno resets when the port opens
            self.status = f"Arduino: {port}"
            print(f"Connected to Arduino on {port}")
        except Exception as e:
            self.status = "Arduino not connected"
            print(f"Could not open {port}: {e}")

    def wave(self):
        print("Wave detected -> sending W")
        if self.ser:
            try:
                self.ser.write(b"W")
            except Exception as e:
                print(f"Serial write failed: {e}")
                self.ser = None
                self.status = "Arduino disconnected"

    def poll(self):
        if self.ser and self.ser.in_waiting:
            print("Arduino:", self.ser.read(self.ser.in_waiting).decode(errors="ignore").strip())

    def close(self):
        if self.ser:
            self.ser.close()


# ---------------------------------------------------------------- drawing

def draw_text_centered(img, text, y, scale, color, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - w) // 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hand(img, P, color):
    pts = [tuple(int(v) for v in p) for p in P]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 4, WHITE, -1, cv2.LINE_AA)


def fit_to_window(img, window):
    """Letterbox the frame to the window size so the whole camera image is always visible, never stretched or cropped."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect(window)
    except cv2.error:
        return img
    h, w = img.shape[:2]
    if ww <= 0 or wh <= 0 or (ww, wh) == (w, h):
        return img
    s = min(ww / w, wh / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    canvas = np.zeros((wh, ww, 3), np.uint8)
    x0, y0 = (ww - nw) // 2, (wh - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Wave to Elon")
    parser.add_argument("--port", help="Arduino serial port (default: auto-detect)")
    parser.add_argument("--no-serial", action="store_true", help="run without an Arduino")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    arduino = Arduino(None if args.no_serial else (args.port or find_arduino_port()))

    recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        canned_gesture_classifier_options=ClassifierOptions(score_threshold=0.5),
    ))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Allow camera access for your terminal in "
                         "System Settings > Privacy & Security > Camera.")

    tracks = []
    tracker = GestureTracker()
    effects = Effects()
    last_wave = -COOLDOWN
    last_rocket = {}
    start = prev = time.monotonic()
    fullscreen = False
    window = "Wave to Elon"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            h, w = frame.shape[:2]
            sc = w / 1280
            now = time.monotonic()
            dt = min(now - prev, 0.1)
            prev = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - start) * 1000))

            # --- read the hands ---
            hands = [np.array([[p.x * w, p.y * h] for p in lm]) for lm in result.hand_landmarks]
            hand_tracks = match_tracks(tracks, [P[9] for P in hands], now, max_dist=0.15 * w)
            detected = {}
            wave_at = None
            for P, track, guesses in zip(hands, hand_tracks, result.gestures):
                top = guesses[0] if guesses else None
                vote = MODEL_GESTURES.get(top.category_name) if top and top.score >= MIN_SCORE else None
                if vote is None and is_rock_sign(P):
                    vote = "rock"
                track.votes.append(vote)

                waving_pose = is_waving_pose(P)
                draw_hand(frame, P, SKY if waving_pose else (0, 160, 255))
                if waving_pose:
                    if track.wave.update(P[9][0], now, SWING_DISTANCE * palm_size(P)):
                        wave_at = P[9]
                else:
                    track.wave.reset()

                name = track.stable_gesture()
                if name:
                    detected[(name, track.id)] = gesture_anchor(name, P)

            for i in range(len(hands)):
                for j in range(i + 1, len(hands)):
                    centre = two_hand_heart(hands[i], hands[j])
                    if centre is not None:
                        detected[("heart", (hand_tracks[i].id, hand_tracks[j].id))] = centre

            # --- wave: the main event ---
            if wave_at is not None and now - last_wave > COOLDOWN:
                last_wave = now
                for track in tracks:
                    track.wave.reset()
                arduino.wave()
                effects.hearts(*wave_at, 25)

            # --- other gestures ---
            for key in tracker.update(detected, now):
                name, pos = key[0], detected[key]
                if name == "heart":
                    effects.hearts(*pos, 25)
                elif name == "thumbs_up":
                    effects.confetti_rain(w, 120)
                elif name == "peace":
                    effects.stars(*pos, 30, GOLDS)
                elif name == "rock":
                    effects.stars(*pos, 30, RAINBOW)
                    effects.confetti_burst(*pos, 40)
            for key, (x, y) in detected.items():
                name = key[0]
                if name == "heart":
                    effects.hearts(x, y, emit_count(12, dt))
                elif name == "thumbs_up":
                    effects.confetti_rain(w, emit_count(40, dt))
                elif name == "peace":
                    effects.stars(x, y, emit_count(12, dt), GOLDS)
                elif name == "rock":
                    effects.stars(x, y, emit_count(15, dt), RAINBOW)
                elif name == "point" and now - last_rocket.get(key, -10.0) > 1.2:
                    last_rocket[key] = now
                    effects.rocket(x, y)

            effects.update(dt, h)
            effects.draw(frame, now)

            # --- text ---
            draw_text_centered(frame, "WAVE TO ELON!", int(85 * sc), 2.4 * sc, WHITE, max(2, int(5 * sc)))
            if now - last_wave < 2.5:
                pulse = 1 + 0.06 * math.sin(now * 10)
                draw_text_centered(frame, "Elon is waving back!", int(140 * sc), 1.1 * sc * pulse, SKY,
                                   max(1, int(2 * sc)))

            name, started = tracker.current_label()
            if name:
                text, color = GESTURE_LABELS[name]
                pop = 1 + 0.35 * max(0.0, 1 - (now - started) / 0.25)
                draw_text_centered(frame, text, h - int(95 * sc), 1.5 * sc * pop, color, max(2, int(3 * sc)))

            progress = max((t.wave.progress for t in tracks), default=0.0)
            bar_w = int(w * 0.3)
            x0, y0 = (w - bar_w) // 2, h - int(50 * sc)
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + int(14 * sc)), (60, 60, 60), -1)
            cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * progress), y0 + int(14 * sc)), SKY, -1)
            cv2.putText(frame, f"{arduino.status}  |  hands: {len(hands)}", (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc, WHITE, 1, cv2.LINE_AA)

            cv2.imshow(window, fit_to_window(frame, window))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if key == ord("t"):
                last_wave = now
                arduino.wave()
                effects.hearts(w / 2, h / 2, 25)
            arduino.poll()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        arduino.close()


if __name__ == "__main__":
    main()
