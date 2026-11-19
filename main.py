
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# ==========================================
# 1. CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not open!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Hand model not found!")
    cap.release()
    exit()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = HandLandmarker.create_from_options(options)

# ==========================================
# 3. CANVAS AND VARIABLES
# ==========================================

pink_canvas = None
white_canvas = None

previous_x = None
previous_y = None

smooth_x = None
smooth_y = None

is_drawing = False
last_timestamp = 0

PINK = (255, 50, 220)
WHITE = (255, 255, 255)

PINK_THICKNESS = 7
WHITE_THICKNESS = 2

PINCH_START = 0.32
PINCH_STOP = 0.48

# ==========================================
# 4. MAIN LOOP
# ==========================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera frame not received!")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        if pink_canvas is None:

            pink_canvas = np.zeros(
                (height, width, 3), dtype=np.uint8
            )

            white_canvas = np.zeros(
                (height, width, 3), dtype=np.uint8
            )

        # ==================================
        # 5. HAND DETECTION
        # ==================================

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = max(
            int(time.monotonic() * 1000),
            last_timestamp + 1
        )

        last_timestamp = timestamp

        results = detector.detect_for_video(
            mp_image, timestamp
        )

        # ==================================
        # 6. THUMB AND INDEX DETECTION
        # ==================================

        if results.hand_landmarks:

            landmarks = results.hand_landmarks[0]

            # Index finger tip
            index_tip = landmarks[8]

            # Thumb tip
            thumb_tip = landmarks[4]

            # Palm reference points
            index_base = landmarks[5]
            pinky_base = landmarks[17]

            x = int(index_tip.x * width)
            y = int(index_tip.y * height)

            thumb_x = int(thumb_tip.x * width)
            thumb_y = int(thumb_tip.y * height)

            # ==================================
            # 7. PINCH DETECTION
            # ==================================

            pinch_distance = math.hypot(
                x - thumb_x,
                y - thumb_y
            )

            palm_size = math.hypot(
                (index_base.x - pinky_base.x) * width,
                (index_base.y - pinky_base.y) * height
            )

            if palm_size > 0:

                pinch_ratio = pinch_distance / palm_size

                # Start drawing when pinched
                if not is_drawing:

                    if pinch_ratio < PINCH_START:
                        is_drawing = True

                        # Start a new stroke
                        previous_x = None
                        previous_y = None

                # Stop drawing when released
                else:

                    if pinch_ratio > PINCH_STOP:
                        is_drawing = False

                        previous_x = None
                        previous_y = None

            # ==================================
            # 8. SMOOTH INDEX FINGER MOVEMENT
            # ==================================

            if smooth_x is None:

                smooth_x = x
                smooth_y = y

            else:

                movement = math.hypot(
                    x - smooth_x,
                    y - smooth_y
                )

                alpha = 0.25 + 0.45 * min(
                    movement / 40, 1
                )

                smooth_x = (
                    alpha * x
                    + (1 - alpha) * smooth_x
                )

                smooth_y = (
                    alpha * y
                    + (1 - alpha) * smooth_y
                )

            current_x = int(smooth_x)
            current_y = int(smooth_y)

            # ==================================
            # 9. DRAW USING INDEX FINGER
            # ==================================

            if is_drawing:

                if previous_x is not None:

                    cv2.line(
                        pink_canvas,
                        (previous_x, previous_y),
                        (current_x, current_y),
                        PINK,
                        PINK_THICKNESS,
                        cv2.LINE_AA
                    )

                    cv2.line(
                        white_canvas,
                        (previous_x, previous_y),
                        (current_x, current_y),
                        WHITE,
                        WHITE_THICKNESS,
                        cv2.LINE_AA
                    )

                previous_x = current_x
                previous_y = current_y

            else:

                previous_x = None
                previous_y = None

            # ==================================
            # 10. CURSOR
            # ==================================

            cv2.circle(
                frame,
                (current_x, current_y),
                7,
                (0, 255, 0),
                -1
            )

            # Show drawing status
            status = "DRAWING ON" if is_drawing else "DRAWING OFF"

            status_color = (
                (0, 255, 0) if is_drawing
                else (0, 0, 255)
            )

            cv2.putText(
                frame,
                status,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )

        else:

            # Hand removed: stop drawing
            is_drawing = False

            previous_x = None
            previous_y = None

            smooth_x = None
            smooth_y = None

        # ==================================
        # 11. NEON GLOW
        # ==================================

        glow = cv2.GaussianBlur(
            pink_canvas,
            (0, 0),
            sigmaX=8,
            sigmaY=8
        )

        neon = cv2.addWeighted(
            glow, 1.0,
            pink_canvas, 1.0,
            0
        )

        neon = cv2.addWeighted(
            neon, 1.0,
            white_canvas, 0.9,
            0
        )

        frame = cv2.addWeighted(
            frame, 0.75,
            neon, 1.0,
            0
        )

        # ==================================
        # 12. INSTRUCTIONS
        # ==================================

        cv2.putText(
            frame,
            "PINCH THUMB + INDEX: START",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "MOVE INDEX TO WRITE",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "RELEASE PINCH: STOP",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2
        )

        cv2.putText(
            frame,
            "C: CLEAR | Q: QUIT",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2
        )

        # ==================================
        # 13. DISPLAY
        # ==================================

        cv2.imshow(
            "Neon Handwriting",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):

            pink_canvas[:] = 0
            white_canvas[:] = 0

            previous_x = None
            previous_y = None

        if key == ord('q'):
            break

finally:

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("Program closed successfully!")
