import cv2
import numpy as np
import json

# === THEM: CAU HINH KICH THUOC HIEN THI ===
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()

# Load anh hoac frame dau video
cap = cv2.VideoCapture("demo1.mkv")
ret, frame = cap.read()
cap.release()

# Luu kich thuoc goc
original_height, original_width = frame.shape[:2]
print(f"Kich thuoc goc video: {original_width}x{original_height}")

# Resize de hien thi
display_frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))

# Tinh ti le de chuyen doi toa do
scale_x = original_width / DESIRED_WIDTH
scale_y = original_height / DESIRED_HEIGHT

cv2.namedWindow("Draw ROI")
cv2.setMouseCallback("Draw ROI", mouse_callback)

while True:
    temp = display_frame.copy()

    # Ve cac diem
    for p in points:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    # Noi thanh polygon
    if len(points) > 1:
        cv2.polylines(temp, [np.array(points)], False, (0, 255, 0), 2)

    cv2.imshow("Draw ROI", temp)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    elif key == ord('s'):  # SAVE
        if len(points) >= 3:
            # Chuyen toa do tu display ve kich thuoc goc
            original_points = []
            for p in points:
                orig_x = int(p[0] * scale_x)
                orig_y = int(p[1] * scale_y)
                original_points.append([orig_x, orig_y])
            
            roi = np.array(original_points, dtype=np.int32).tolist()

            with open("roi.json", "w") as f:
                json.dump({"ROI": roi}, f)

            print(f"Saved ROI to roi.json (original size: {original_width}x{original_height})")
            print(f"ROI points: {original_points}")
            break
        else:
            print("Can it nhat 3 diem!")

    elif key == ord('c'):  # CLEAR
        points = []

cv2.destroyAllWindows()