import cv2
import time
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"

PRIMARY_CAMERA_INDEX = 1
FALLBACK_CAMERA_INDEX = 0

CONFIDENCE = 0.20

# How long to keep the last driver box visible
BOX_GRACE_PERIOD = 2.0

# How long before declaring driver missing
MISSING_TIMEOUT = 3.0


# Your model classes
AWAKE = "awake"
DROWSY = "drowsy"
MOBILE = "mobile"
SEAT_BELT = "seat belt"


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    print("Trying USB webcam...")

    cap = cv2.VideoCapture(
        PRIMARY_CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if cap.isOpened():

        print("USB webcam connected.")

        return cap

    print("USB webcam unavailable.")
    print("Trying camera index 0...")

    cap.release()

    cap = cv2.VideoCapture(
        FALLBACK_CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if cap.isOpened():

        print("Fallback camera connected.")

        return cap

    return None


# ============================================================
# BANNER
# ============================================================

def draw_banner(frame, state, hazards):

    width = frame.shape[1]

    if state == "ALERT":

        color = (0, 0, 255)

        hazard_text = ", ".join(
            h.upper() for h in hazards
        )

        text = f"WARNING: {hazard_text}"

    elif state == "MISSING":

        color = (0, 165, 255)

        text = "NO DRIVER DETECTED"

    else:

        color = (0, 180, 0)

        text = "DRIVER STATUS: OK"

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 70),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (20, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" DRIVER MONITORING SYSTEM")
    print("========================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = YOLO(MODEL_PATH)

    print("Model loaded.")
    print("Classes:", model.names)

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = open_camera()

    if cap is None:

        print("ERROR: Could not open camera.")

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    state = "OK"

    active_hazards = set()

    # Last time we detected a driver-related object
    last_driver_detection = time.time()

    # --------------------------------------------------------
    # LAST DRIVER BOX
    # --------------------------------------------------------

    last_driver_box = None

    last_driver_box_time = 0

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE,
            verbose=False
        )

        # ----------------------------------------------------
        # FRAME STATUS
        # ----------------------------------------------------

        driver_detected = False

        current_hazards = set()

        current_driver_box = None

        current_driver_label = None

        current_driver_confidence = 0

        seatbelt_detected = False

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        if results and results[0].boxes is not None:

            for box in results[0].boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                class_name = (
                    class_name
                    .strip()
                    .lower()
                )

                # ------------------------------------------------
                # COORDINATES
                # ------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # ------------------------------------------------
                # AWAKE
                # ------------------------------------------------

                if class_name == AWAKE:

                    driver_detected = True

                    current_driver_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    current_driver_label = "AWAKE"

                    current_driver_confidence = confidence

                # ------------------------------------------------
                # DROWSY
                # ------------------------------------------------

                elif class_name == DROWSY:

                    driver_detected = True

                    current_hazards.add("DROWSY")

                    # Drowsy is also evidence of driver
                    current_driver_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    current_driver_label = "DROWSY"

                    current_driver_confidence = confidence

                # ------------------------------------------------
                # MOBILE
                # ------------------------------------------------

                elif class_name == MOBILE:

                    driver_detected = True

                    current_hazards.add("MOBILE")

                    # Mobile detection is evidence that
                    # the driver is present
                    current_driver_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    current_driver_label = "MOBILE"

                    current_driver_confidence = confidence

                # ------------------------------------------------
                # SEAT BELT
                # ------------------------------------------------

                elif class_name == SEAT_BELT:

                    seatbelt_detected = True

                # ------------------------------------------------
                # DRAW NON-DRIVER OBJECTS
                # ------------------------------------------------

                if class_name == SEAT_BELT:

                    color = (255, 255, 0)

                    label = (
                        f"Seat Belt "
                        f"{confidence:.2f}"
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA
                    )

        # ========================================================
        # UPDATE DRIVER BOX
        # ========================================================

        if current_driver_box is not None:

            last_driver_box = current_driver_box

            last_driver_box_time = time.time()

            last_driver_label = current_driver_label

            last_driver_confidence = current_driver_confidence

            last_driver_detection = time.time()

        # ========================================================
        # DRIVER DETECTION TIMER
        # ========================================================

        time_since_driver = (
            time.time()
            - last_driver_detection
        )

        # ========================================================
        # STATE
        # ========================================================

        if driver_detected:

            active_hazards = current_hazards

            if len(current_hazards) > 0:

                state = "ALERT"

            else:

                state = "OK"

        else:

            # Don't immediately change state

            if time_since_driver >= MISSING_TIMEOUT:

                state = "MISSING"

        # ========================================================
        # DRAW PERSISTENT DRIVER BOX
        # ========================================================

        time_since_box = (
            time.time()
            - last_driver_box_time
        )

        if (
            last_driver_box is not None
            and time_since_box <= BOX_GRACE_PERIOD
        ):

            x1, y1, x2, y2 = last_driver_box

            # --------------------------------------------
            # Box color based on current state
            # --------------------------------------------

            if state == "ALERT":

                color = (0, 0, 255)

            elif state == "MISSING":

                color = (0, 165, 255)

            else:

                color = (0, 255, 0)

            # --------------------------------------------
            # Draw box
            # --------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # --------------------------------------------
            # Label
            # --------------------------------------------

            label = (
                f"{last_driver_label} "
                f"{last_driver_confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

        # ========================================================
        # BANNER
        # ========================================================

        draw_banner(
            frame,
            state,
            active_hazards
        )

        # ========================================================
        # STATUS INFORMATION
        # ========================================================

        height = frame.shape[0]

        cv2.putText(
            frame,
            f"STATE: {state}",
            (20, height - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Last detection: "
            f"{time_since_driver:.1f}s",
            (20, height - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if seatbelt_detected:

            belt_text = "SEAT BELT: DETECTED"

        else:

            belt_text = "SEAT BELT: NOT DETECTED"

        cv2.putText(
            frame,
            belt_text,
            (20, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ========================================================
        # DISPLAY
        # ========================================================

        cv2.imshow(
            "Driver Monitoring System",
            frame
        )

        # ========================================================
        # EXIT
        # ========================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
