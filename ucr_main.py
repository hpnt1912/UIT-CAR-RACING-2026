from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import cv2
import numpy as np


# ============================================================
# UCR 2026 - LINE FOLLOWING V13
# Stable PID + Partial/Lost Line Recovery
# ============================================================


# ============================================================
# 1. CAMERA
# ============================================================

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180

CENTER_X = IMAGE_WIDTH // 2


# ============================================================
# 2. PID
# ============================================================

# PID đang dùng theo setup ổn nhất của bạn
KP = 0.20
KD = 0.50

# Không dùng heading gain
# Không dùng KI

MAX_ANGLE = 25.0


# ============================================================
# 3. ERROR FILTER
# ============================================================

ERROR_ALPHA = 0.60

# Không cho error nhảy quá lớn giữa 2 frame
MAX_ERROR_CHANGE = 14.0


# ============================================================
# 4. SPEED
# ============================================================

SPEED_STRAIGHT = 23.0
SPEED_MILD = 21.0
SPEED_MEDIUM = 19.0
SPEED_SHARP = 16.0
SPEED_DANGER = 12.0

# Khi mất line
SPEED_LOST = 10.0


# ============================================================
# 5. LOST LINE
# ============================================================

# Khi mất toàn bộ line:
# giữ steering cũ trong số frame này
LOST_HOLD_FRAMES = 5


# ============================================================
# 6. LINE DETECTOR
# ============================================================

# Các vùng ảnh
FAR_Y1 = 95
FAR_Y2 = 120

MID_Y1 = 120
MID_Y2 = 145

NEAR_Y1 = 150
NEAR_Y2 = 178


# Vùng x cần tìm line
SCAN_X1 = 110
SCAN_X2 = 210

# Bước scan
SCAN_STEP = 3

# Line sáng hơn nền bao nhiêu
THRESHOLD_OFFSET = 15


# ============================================================
# 7. GLOBAL STATE
# ============================================================

previous_error = 0.0
filtered_error = 0.0

last_error = 0.0
last_valid_error = 0.0

last_steering = 0.0
last_valid_steering = 0.0

partial_counter = 0
lost_counter = 0

frame_count = 0


# ============================================================
# 8. UTILITY
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


# ============================================================
# 9. DETECT LINE IN ONE BAND
# ============================================================

def detect_line_band(gray, y1, y2):
    """
    Tìm vị trí line trong một vùng ảnh.

    Mỗi scanline:
        - lấy median độ sáng
        - tìm pixel sáng hơn median + threshold
        - chọn vị trí sáng nhất

    Sau đó lấy median của các điểm tìm được.
    """

    points = []

    y1 = max(0, y1)
    y2 = min(gray.shape[0], y2)

    for y in range(y1, y2):

        row = gray[y, SCAN_X1:SCAN_X2]

        if len(row) == 0:
            continue

        median_value = np.median(row)

        threshold = median_value + THRESHOLD_OFFSET

        candidates = np.where(row > threshold)[0]

        if len(candidates) == 0:
            continue

        # Chọn pixel sáng nhất
        best_index = candidates[np.argmax(row[candidates])]

        x = SCAN_X1 + best_index

        points.append(x)

    if len(points) < 3:
        return None

    return float(np.median(points))


# ============================================================
# 10. DETECT FAR / MID / NEAR
# ============================================================

def detect_line(gray):
    far = detect_line_band(
        gray,
        FAR_Y1,
        FAR_Y2
    )

    mid = detect_line_band(
        gray,
        MID_Y1,
        MID_Y2
    )

    near = detect_line_band(
        gray,
        NEAR_Y1,
        NEAR_Y2
    )

    return far, mid, near


# ============================================================
# 11. CALCULATE ERROR
# ============================================================

def calculate_error(far, mid, near):
    """
    Ưu tiên:

        NEAR
          ↓
        MID
          ↓
        FAR
          ↓
        giữ error cũ
          ↓
        mất hoàn toàn
    """

    global partial_counter
    global lost_counter
    global last_valid_error

    # --------------------------------------------------------
    # CASE 1:
    # NEAR detected
    # --------------------------------------------------------

    if near is not None:

        partial_counter = 0
        lost_counter = 0

        error = near - CENTER_X

        # Kiểm tra error jump
        if abs(error - last_valid_error) > MAX_ERROR_CHANGE:

            return last_valid_error

        last_valid_error = error

        return error


    # --------------------------------------------------------
    # CASE 2:
    # NEAR mất nhưng MID còn
    # --------------------------------------------------------

    if mid is not None:

        partial_counter += 1
        lost_counter = 0

        error = mid - CENTER_X

        # Nếu nhảy quá lớn
        if abs(error - last_valid_error) > MAX_ERROR_CHANGE:

            return last_valid_error

        last_valid_error = error

        return error


    # --------------------------------------------------------
    # CASE 3:
    # Chỉ còn FAR
    # --------------------------------------------------------

    if far is not None:

        partial_counter += 1
        lost_counter = 0

        # FAR không đủ tin cậy để đổi hướng.
        # Giữ error cũ.
        return last_valid_error


    # --------------------------------------------------------
    # CASE 4:
    # Mất hoàn toàn line
    # --------------------------------------------------------

    partial_counter = 0

    lost_counter += 1

    return None


# ============================================================
# 12. FILTER ERROR
# ============================================================

def filter_error(error):
    """
    Low-pass filter để giảm việc xe lắc trái/phải.
    """

    global filtered_error

    if error is None:
        return None

    filtered_error = (
        ERROR_ALPHA * error
        +
        (1.0 - ERROR_ALPHA) * filtered_error
    )

    return filtered_error


# ============================================================
# 13. CALCULATE STEERING
# ============================================================

def calculate_steering(error):
    """
    PID chỉ dùng P + D.

    steering = KP * error + KD * derivative
    """

    global previous_error
    global last_steering
    global last_valid_steering

    if error is None:

        return last_steering


    # --------------------------------------------------------
    # Derivative
    # --------------------------------------------------------

    derivative = error - previous_error


    # --------------------------------------------------------
    # PD
    # --------------------------------------------------------

    steering = (
        KP * error
        +
        KD * derivative
    )


    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    steering = clamp(
        steering,
        -MAX_ANGLE,
        MAX_ANGLE
    )


    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    previous_error = error

    last_steering = steering
    last_valid_steering = steering

    return steering


# ============================================================
# 14. SPEED CONTROL
# ============================================================

def calculate_speed(error):

    if error is None:
        return SPEED_LOST


    error_abs = abs(error)


    # --------------------------------------------------------
    # Straight
    # --------------------------------------------------------

    if error_abs < 3:

        return SPEED_STRAIGHT


    # --------------------------------------------------------
    # Mild correction
    # --------------------------------------------------------

    elif error_abs < 6:

        return SPEED_MILD


    # --------------------------------------------------------
    # Medium correction
    # --------------------------------------------------------

    elif error_abs < 10:

        return SPEED_MEDIUM


    # --------------------------------------------------------
    # Sharp correction
    # --------------------------------------------------------

    elif error_abs < 15:

        return SPEED_SHARP


    # --------------------------------------------------------
    # Very sharp
    # --------------------------------------------------------

    else:

        return SPEED_DANGER


# ============================================================
# 15. LOST LINE CONTROL
# ============================================================

def lost_line_control():
    """
    Khi camera mất line:

    Frame 1 -> giữ lái cũ
    Frame 2 -> giữ lái cũ
    ...
    Frame 5 -> giữ lái cũ

    Sau đó:
        steering = 0

    => xe đi thẳng
    """

    global last_steering

    if lost_counter <= LOST_HOLD_FRAMES:

        return last_steering

    else:

        return 0.0


# ============================================================
# 16. DRAW DEBUG
# ============================================================

def draw_debug(frame, far, mid, near, error, steering, speed):

    # Center line
    cv2.line(
        frame,
        (CENTER_X, 0),
        (CENTER_X, IMAGE_HEIGHT),
        (255, 255, 255),
        1
    )


    # FAR
    if far is not None:

        x = int(far)

        cv2.circle(
            frame,
            (x, int((FAR_Y1 + FAR_Y2) / 2)),
            4,
            (0, 255, 0),
            -1
        )


    # MID
    if mid is not None:

        x = int(mid)

        cv2.circle(
            frame,
            (x, int((MID_Y1 + MID_Y2) / 2)),
            4,
            (0, 255, 0),
            -1
        )


    # NEAR
    if near is not None:

        x = int(near)

        cv2.circle(
            frame,
            (x, int((NEAR_Y1 + NEAR_Y2) / 2)),
            5,
            (0, 0, 255),
            -1
        )


    # Text
    if error is None:

        error_text = "None"

    else:

        error_text = f"{error:+.1f}"


    text1 = (
        f"F={far} "
        f"M={mid} "
        f"N={near}"
    )

    text2 = (
        f"Err={error_text} "
        f"Steer={steering:+.2f} "
        f"Speed={speed:.1f}"
    )

    text3 = (
        f"Lost={lost_counter} "
        f"Partial={partial_counter}"
    )


    cv2.putText(
        frame,
        text1,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        text2,
        (5, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        text3,
        (5, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    return frame


# ============================================================
# 17. MAIN
# ============================================================

def main():

    global frame_count

    print("[UCR 2026] Starting system...")

    try:

        while True:

            frame_count += 1


            # =================================================
            # GET STATUS
            # =================================================

            state = GetStatus()


            # =================================================
            # GET CAMERA
            # =================================================

            raw_image = GetRaw()


            if raw_image is None:

                print("[WARNING] Camera returned None")

                AVControl(
                    SPEED_LOST,
                    0.0
                )

                continue


            # =================================================
            # RESIZE IF NECESSARY
            # =================================================

            if (
                raw_image.shape[1] != IMAGE_WIDTH
                or
                raw_image.shape[0] != IMAGE_HEIGHT
            ):

                raw_image = cv2.resize(
                    raw_image,
                    (IMAGE_WIDTH, IMAGE_HEIGHT)
                )


            # =================================================
            # GRAYSCALE
            # =================================================

            gray = cv2.cvtColor(
                raw_image,
                cv2.COLOR_BGR2GRAY
            )


            # =================================================
            # DETECT LINE
            # =================================================

            far, mid, near = detect_line(gray)


            # =================================================
            # CALCULATE RAW ERROR
            # =================================================

            raw_error = calculate_error(
                far,
                mid,
                near
            )


            # =================================================
            # FILTER ERROR
            # =================================================

            if raw_error is not None:

                error = filter_error(
                    raw_error
                )

            else:

                error = None


            # =================================================
            # NORMAL DRIVING
            # =================================================

            if error is not None:

                speed = calculate_speed(
                    error
                )

                steering = calculate_steering(
                    error
                )


            # =================================================
            # LOST LINE
            # =================================================

            else:

                speed = SPEED_LOST

                steering = lost_line_control()


            # =================================================
            # SEND CONTROL
            # =================================================

            AVControl(
                float(speed),
                float(steering)
            )


            # =================================================
            # DEBUG IMAGE
            # =================================================

            debug_frame = raw_image.copy()

            debug_frame = draw_debug(
                debug_frame,
                far,
                mid,
                near,
                error,
                steering,
                speed
            )


            cv2.imshow(
                "UCR 2026 - Front Camera",
                debug_frame
            )


            # =================================================
            # TERMINATE
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):

                print("[UCR] Q pressed.")

                break


    except KeyboardInterrupt:

        print("\n[UCR] Keyboard interrupt.")


    except Exception as e:

        print(
            f"\n[!] Error: {e}"
        )


    finally:

        print(
            "\n[UCR 2026] Closing connection..."
        )

        try:

            CloseSocket()

        except Exception:

            pass


        cv2.destroyAllWindows()

        print(
            "[UCR 2026] Exited."
        )


# ============================================================
# 18. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()