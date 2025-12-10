import math

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pycaw.pycaw import AudioUtilities

print("=" * 60)
print("Initializing Combined Gesture Control (Mouse + Volume)...")
print("=" * 60)

# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
print("✓ MediaPipe hand tracking initialized (2 hands)")

# ---------- Screen Info (for mouse) ----------
screen_width, screen_height = pyautogui.size()
print(f"✓ Screen size: {screen_width} x {screen_height}")

# ---------- Audio Control (pycaw, EndpointVolume style) ----------
try:
    device = AudioUtilities.GetSpeakers()
    volume = device.EndpointVolume  # direct handle
    vol_range = volume.GetVolumeRange()
    min_vol = vol_range[0]
    max_vol = vol_range[1]
    print("✓ Audio control initialized (EndpointVolume)")
except Exception as e:
    print(f"✗ Audio error: {e}")
    print("Volume control will be disabled, but mouse control will still work.")
    volume = None
    # Safe defaults if needed
    min_vol = -65.25
    max_vol = 0.0

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("✗ Cannot open webcam")
    raise SystemExit

print("✓ Webcam ready")

# ---------- State Variables ----------
is_muted = False

# Mouse click + movement state
left_click_down = False          # are we currently in "bent" state?
prev_cursor_x = None
prev_cursor_y = None
cursor_dead_zone = 15            # pixels – ignore tiny movements

# Volume smoothing state
last_vol_percentage = None       # last volume we applied (0–100)
vol_step = 2                     # change volume only if it moves by >= 2%


# ---------- Helper Functions ----------
def calculate_distance(point1, point2):
    """Euclidean distance between two 2D points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def is_palm_open(landmarks, img_shape):
    """Detect if palm is open (all fingers extended)."""
    h, w = img_shape[:2]

    fingertips = [8, 12, 16, 20]
    finger_bases = [5, 9, 13, 17]

    fingers_extended = 0

    for tip, base in zip(fingertips, finger_bases):
        tip_y = landmarks[tip].y * h
        base_y = landmarks[base].y * h
        # fingertip clearly above its base
        if tip_y < base_y - 20:
            fingers_extended += 1

    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    if abs(thumb_tip.x - thumb_base.x) * w > 30:
        fingers_extended += 1

    return fingers_extended >= 4


def is_fist(landmarks, img_shape):
    """Detect if hand is in a fist position."""
    h, w = img_shape[:2]

    fingertips = [8, 12, 16, 20]
    palm_base = landmarks[0]

    fingers_closed = 0

    for tip_idx in fingertips:
        tip = landmarks[tip_idx]
        dist = calculate_distance(
            (tip.x * w, tip.y * h),
            (palm_base.x * w, palm_base.y * h)
        )
        if dist < 120:
            fingers_closed += 1

    return fingers_closed >= 3


def get_pinch_distance(landmarks, img_shape):
    """Get distance between thumb tip and index finger tip."""
    h, w = img_shape[:2]

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
    x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

    distance = calculate_distance((x1, y1), (x2, y2))
    return distance, (x1, y1), (x2, y2)


print("\n" + "=" * 60)
print("  COMBINED GESTURE CONTROL ACTIVE")
print("=" * 60)
print("👈 Left hand:")
print("    • Move middle finger MCP = Move Mouse")
print("    • Bend index finger (tip below PIP) = Left Click")
print("\n👉 Right hand:")
print("    • Open palm (5 fingers) = MUTE")
print("    • Fist = UNMUTE")
print("    • Pinch thumb + index = Volume 0–100%")
print("\nPress 'Q' to quit")
print("=" * 60 + "\n")

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror for natural control
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "Show your hands"
    status_color = (255, 255, 255)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # "Left" or "Right" according to MediaPipe
            handedness = results.multi_handedness[idx].classification[0].label

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

            landmarks = hand_landmarks.landmark

            # ---------- LEFT HAND: Mouse Control ----------
            if handedness == "Left":
                gesture_text = "Left hand: Mouse control"
                status_color = (200, 200, 255)

                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mid = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]

                # Use middle finger MCP as cursor anchor
                mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                cursor_x = int(mcp.x * screen_width)
                cursor_y = int(mcp.y * screen_height)

                # Dead zone for smoother movement
                if prev_cursor_x is None:
                    prev_cursor_x, prev_cursor_y = cursor_x, cursor_y

                dx = cursor_x - prev_cursor_x
                dy = cursor_y - prev_cursor_y

                if abs(dx) > cursor_dead_zone or abs(dy) > cursor_dead_zone:
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)
                    prev_cursor_x, prev_cursor_y = cursor_x, cursor_y

                # Debounced click (one click per bend)
                is_bent = index_tip.y >= index_mid.y

                if is_bent and not left_click_down:
                    pyautogui.click()
                    left_click_down = True
                elif not is_bent and left_click_down:
                    left_click_down = False

            # ---------- RIGHT HAND: Volume Control ----------
            elif handedness == "Right" and volume is not None:
                if is_palm_open(landmarks, frame.shape):
                    if not is_muted:
                        volume.SetMute(1, None)
                        is_muted = True
                    gesture_text = "Right hand: PALM - MUTED"
                    status_color = (0, 0, 255)
                    cv2.putText(frame, "MUTED", (220, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                elif is_fist(landmarks, frame.shape):
                    if is_muted:
                        volume.SetMute(0, None)
                        is_muted = False
                    gesture_text = "Right hand: FIST - UNMUTED"
                    status_color = (0, 255, 0)
                    cv2.putText(frame, "PLAYING", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

                else:
                    # Pinch for volume
                    distance, (x1, y1), (x2, y2) = get_pinch_distance(landmarks, frame.shape)

                    # Draw pinch markers
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(frame, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
                    cv2.circle(
                        frame,
                        ((x1 + x2) // 2, (y1 + y2) // 2),
                        12,
                        (0, 255, 255),
                        cv2.FILLED
                    )

                    # Map distance → volume %
                    vol_percentage = np.interp(distance, [20, 200], [0, 100])
                    vol_percentage = float(np.clip(vol_percentage, 0, 100))

                    # Round to nearest step (e.g. 2%)
                    vol_percentage_rounded = int(round(vol_percentage / vol_step) * vol_step)

                    # Only update if changed enough
                    if (last_vol_percentage is None or
                            abs(vol_percentage_rounded - last_vol_percentage) >= vol_step):
                        vol_db = np.interp(
                            vol_percentage_rounded,
                            [0, 100],
                            [min_vol, max_vol]
                        )
                        volume.SetMasterVolumeLevel(vol_db, None)
                        last_vol_percentage = vol_percentage_rounded

                    gesture_text = f"Right hand: Volume {vol_percentage_rounded}%"
                    status_color = (0, 255, 255)

                    # Volume bar
                    bar_x, bar_y = 50, 150
                    bar_w, bar_h = 35, 250

                    cv2.rectangle(
                        frame,
                        (bar_x, bar_y),
                        (bar_x + bar_w, bar_y + bar_h),
                        (255, 255, 255),
                        3
                    )

                    fill_height = int(np.interp(
                        vol_percentage_rounded,
                        [0, 100],
                        [0, bar_h]
                    ))
                    cv2.rectangle(
                        frame,
                        (bar_x, bar_y + bar_h - fill_height),
                        (bar_x + bar_w, bar_y + bar_h),
                        (0, 255, 0),
                        cv2.FILLED
                    )

                    cv2.putText(
                        frame,
                        f"{vol_percentage_rounded}%",
                        (bar_x - 10, bar_y + bar_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

    # ---------- UI Overlay ----------
    cv2.rectangle(frame, (0, 0), (640, 60), (50, 50, 50), cv2.FILLED)
    cv2.putText(
        frame,
        gesture_text,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    cv2.putText(
        frame,
        "Press 'Q' to Quit",
        (450, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.imshow("Combined Gesture Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Combined Gesture Control Stopped. Goodbye!")
