# Gọi trực tiếp các hàm từ ucr_lib
from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import cv2

def main():
    print("[UCR 2026] Starting system...")
    try:
        while True:
            # 1. Fetch data from Unity
            state = GetStatus()
            raw_image = GetRaw()

            # 2. Display status & camera
            print(f"Status: {state}")
            cv2.imshow('UCR 2026 - Front Camera', raw_image)
            
            # Control limits: max speed = 90, max angle = 25
            speed = 20.0
            angle = 0.0
            # ==========================================

            # 3. Send commands to Unity
            AVControl(speed, angle)

            # Press 'q' on the camera window to quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except Exception as e:
        print(f"\n[!] Error (Is Unity running?): {e}")
        
    finally:
        print('\n[UCR 2026] Closing connection...')
        CloseSocket()
        cv2.destroyAllWindows()
        print("[UCR 2026] Exited.")

if __name__ == "__main__":
    main()
