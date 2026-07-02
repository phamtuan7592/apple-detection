import cv2
import numpy as np
from object_detection import ObjectDetector
import sys
import os
import json

# === CẤU HÌNH KÍCH THƯỚC HIỂN THỊ ===
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

print("Loading YOLO11 model with built-in tracking...")
od = ObjectDetector(model_path="best.pt", min_score_thresh=0.35)
print("Model loaded successfully!")

# === HỆ THỐNG TRACKING VÀ ĐẾM ===
# THAY ĐỔI: Dùng track_ids từ YOLO thay vì tự quản lý
counted_ids = set()          # Lưu track_ids đã đếm
total_unique_count = 0       # Tổng số duy nhất

def get_object_center(box):
    """Lấy tâm của bounding box"""
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def calculate_distance(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# THAY ĐỔI: Hàm đếm đơn giản hơn với track_ids
def update_tracking_and_count(bboxes_display, track_ids, roi_display):
    """Cập nhật tracking và đếm các đối tượng DUY NHẤT với YOLO track_ids"""
    global total_unique_count, counted_ids
    
    current_in_roi = 0
    filtered_boxes = []
    filtered_ids = []
    filtered_scores = []
    filtered_track_ids = []
    
    # Duyệt từng detection với track_id
    for i, (box_disp, tid) in enumerate(zip(bboxes_display, track_ids)):
        # Bỏ qua nếu không có track_id hợp lệ
        if tid == -1:
            continue
            
        if is_inside_roi(box_disp, roi_display):
            current_in_roi += 1
            filtered_boxes.append(box_disp)
            filtered_track_ids.append(tid)
            
            # Đếm nếu ID chưa được đếm
            if tid not in counted_ids:
                counted_ids.add(tid)
                total_unique_count += 1
    
    return total_unique_count, current_in_roi, filtered_boxes, filtered_track_ids

def is_inside_roi(box, roi):
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0

# --- MENU CHỌN NGUỒN ---
print("\n" + "="*40)
print("HỆ THỐNG GIÁM SÁT THÔNG MINH")
print("="*40)
print("1. Sử dụng Webcam (Real-time)")
print("2. Sử dụng File Video")
print("3. Thoát")
print("-"*40)

choice = input("Nhập lựa chọn của bạn (1, 2 hoặc 3): ")

if choice == '1':
    cap = cv2.VideoCapture(0)
    source_label = "Webcam"
elif choice == '2':
    video_path = "demo1.mkv"
    cap = cv2.VideoCapture(video_path)
    source_label = f"Video: {video_path}"
else:
    sys.exit()

if not cap.isOpened():
    print(f"\n[LỖI] Không thể mở được {source_label}!")
    sys.exit()

# --- TẢI VÙNG ROI ---
if os.path.exists("roi.json"):
    with open("roi.json", "r") as f:
        data = json.load(f)
        ROI_original = np.array(data["ROI"], dtype=np.int32)
else:
    ROI_original = np.array([[0, 278], [1277, 145], [1266, 718], [6, 713], [1, 280]], dtype=np.int32)

# Lấy kích thước gốc
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tính tỉ lệ scale
scale_x = DESIRED_WIDTH / original_width
scale_y = DESIRED_HEIGHT / original_height

# Tạo cửa sổ 
cv2.namedWindow("Nhan dang anh & Tong hop", cv2.WINDOW_NORMAL)

print("\n" + "="*50)
print("HƯỚNG DẪN:")
print("- ESC: Thoát")
print("- SPACE: Tạm dừng/Tiếp tục")
print("="*50)

frame_count = 0
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_display = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
        ROI_display = (ROI_original * [scale_x, scale_y]).astype(np.int32)
        
        # ============ THAY ĐỔI CHÍNH ============
        # Detect trên frame gốc với TRACKING
        bboxes, class_ids, scores, track_ids = od.detect(
            frame, 
            conf=0.35, 
            track=True,    # Bật tracking
            persist=True   # Giữ tracking qua frame
        )
        
        # Scale bounding boxes
        bboxes_display = []
        valid_track_ids = []
        for i, box in enumerate(bboxes):
            x, y, w, h = box
            x_disp = int(x * scale_x)
            y_disp = int(y * scale_y)
            w_disp = int(w * scale_x)
            h_disp = int(h * scale_y)
            bboxes_display.append([x_disp, y_disp, w_disp, h_disp])
            
            # Lấy track_id tương ứng
            if i < len(track_ids):
                valid_track_ids.append(track_ids[i])
            else:
                valid_track_ids.append(-1)
        
        # Cập nhật tracking và đếm (dùng track_ids từ YOLO)
        total_unique, current_in_roi, filtered_boxes, filtered_track_ids = update_tracking_and_count(
            bboxes_display, valid_track_ids, ROI_display
        )
        
        # Vẽ các box đã lọc
        filtered_ids = []
        filtered_scores = []
        for i, box_disp in enumerate(bboxes_display):
            if any(np.array_equal(box_disp, fb) for fb in filtered_boxes):
                # Lấy class_id và score tương ứng
                idx = bboxes_display.index(box_disp)
                filtered_ids.append(class_ids[idx])
                filtered_scores.append(scores[idx])
        # ========================================
        
        # === VẼ KẾT QUẢ (GIỮ NGUYÊN PHẦN NÀY) ===
        frame_result = frame_display.copy()
        
        # Vẽ ROI
        cv2.polylines(frame_result, [ROI_display], True, (0, 255, 255), 2)
        
        # Vẽ bounding boxes (GIỮ NGUYÊN)
        frame_result = od.draw_boxes(
            frame_result, 
            filtered_boxes, 
            filtered_ids, 
            filtered_scores, 
            line_thickness=2
        )
        
        # === HIỂN THỊ THÔNG TIN (GIỮ NGUYÊN) ===
        # Số lượng hiện tại trong ROI
        cv2.putText(frame_result, f'Objects in ROI now: {current_in_roi}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # TỔNG SỐ DUY NHẤT ĐÃ ĐẾM 
        cv2.putText(frame_result, f'TOTAL COUNT: {total_unique}', (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
    
        # Hiển thị cửa sổ
        cv2.imshow("Nhan dang anh & Tong hop", frame_result)
    
    # Xử lý phím
    key = cv2.waitKey(50)
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE
        paused = not paused
       
# In kết quả cuối cùng
print("\n" + "="*50)
print("KẾT QUẢ GIÁM SÁT")
print("="*50)
print(f"Tổng số đối tượng DUY NHẤT đã đi qua ROI: {total_unique_count}")
print(f"Tổng số frame đã xử lý: {frame_count}")
print("="*50)

cap.release()
cv2.destroyAllWindows()
od.close()