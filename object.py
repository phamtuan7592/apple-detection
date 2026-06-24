import cv2
import numpy as np
from object_detection import ObjectDetector
import sys
import os
import json

# === CẤU HÌNH KÍCH THƯỚC HIỂN THỊ ===
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

print("Loading YOLO11 model...")
od = ObjectDetector(model_path="best.pt", min_score_thresh=0.35)
print("Model loaded successfully!")

# === HỆ THỐNG TRACKING VÀ ĐẾM ===
next_object_id = 0
tracked_objects = {}  # {object_id: {'last_position': (cx, cy), 'counted': False}}
counted_objects = set()
total_unique_count = 0

def get_object_center(box):
    """Lấy tâm của bounding box"""
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def calculate_distance(p1, p2):
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def update_tracking_and_count(bboxes_display, roi_display, frame_count, max_distance=50):
    """Cập nhật tracking và đếm các đối tượng DUY NHẤT"""
    global next_object_id, total_unique_count, tracked_objects, counted_objects
    
    current_detections = []
    
    # Lấy tâm của tất cả các bounding box hiện tại
    for box in bboxes_display:
        center = get_object_center(box)
        inside_roi = cv2.pointPolygonTest(roi_display, center, False) >= 0
        current_detections.append({
            'box': box,
            'center': center,
            'inside_roi': inside_roi
        })
    
    # Cập nhật tracking cho các object hiện có
    matched_current_ids = set()
    
    # Duyệt từng detection hiện tại
    for detection in current_detections:
        center = detection['center']
        inside_roi = detection['inside_roi']
        
        # Tìm object đã tồn tại gần nhất
        matched_id = None
        min_dist = float('inf')
        
        for obj_id, obj_data in tracked_objects.items():
            dist = calculate_distance(center, obj_data['last_position'])
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                matched_id = obj_id
        
        if matched_id is not None:
            # Cập nhật object hiện có
            matched_current_ids.add(matched_id)
            tracked_objects[matched_id]['last_position'] = center
            
            # KIỂM TRA ĐẾM: Chỉ đếm khi object lần đầu vào ROI
            if inside_roi and not tracked_objects[matched_id]['counted']:
                total_unique_count += 1
                tracked_objects[matched_id]['counted'] = True
                counted_objects.add(matched_id)
        else:
            # Tạo object mới
            tracked_objects[next_object_id] = {
                'last_position': center,
                'counted': False,
                'first_seen_frame': frame_count
            }
            
            # Nếu object mới đã ở trong ROI ngay từ đầu -> đếm luôn
            if inside_roi:
                total_unique_count += 1
                tracked_objects[next_object_id]['counted'] = True
                counted_objects.add(next_object_id)
            
            matched_current_ids.add(next_object_id)
            next_object_id += 1
    
    # Xóa các object không còn xuất hiện
    objects_to_remove = []
    for obj_id in list(tracked_objects.keys()):
        if obj_id not in matched_current_ids and tracked_objects[obj_id]['counted']:
            objects_to_remove.append(obj_id)
    
    for obj_id in objects_to_remove:
        del tracked_objects[obj_id]
    
    return total_unique_count

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

def is_inside_roi(box, roi):
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0

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
        
        # Detect trên frame gốc
        bboxes, class_ids, scores = od.detect(frame, conf=0.35)
        
        # Scale bounding boxes
        bboxes_display = []
        for box in bboxes:
            x, y, w, h = box
            x_disp = int(x * scale_x)
            y_disp = int(y * scale_y)
            w_disp = int(w * scale_x)
            h_disp = int(h * scale_y)
            bboxes_display.append([x_disp, y_disp, w_disp, h_disp])
        
        # Cập nhật tracking và đếm TỔNG DUY NHẤT
        total_unique = update_tracking_and_count(bboxes_display, ROI_display, frame_count)
        
        # Đếm số lượng hiện tại trong ROI
        current_in_roi = 0
        filtered_boxes = []
        filtered_ids = []
        filtered_scores = []
        
        for box_disp, cid, score in zip(bboxes_display, class_ids, scores):
            if is_inside_roi(box_disp, ROI_display):
                current_in_roi += 1
                filtered_boxes.append(box_disp)
                filtered_ids.append(cid)
                filtered_scores.append(score)
        
        # Vẽ kết quả
        frame_result = frame_display.copy()
        frame_result = od.draw_boxes(frame_result, filtered_boxes, filtered_ids, filtered_scores, line_thickness=2)
        
        # Vẽ ROI
        cv2.polylines(frame_result, [ROI_display], True, (0, 255, 255), 2)
        
        # === HIỂN THỊ THÔNG TIN ===
        # Số lượng hiện tại trong ROI
        cv2.putText(frame_result, f'Objects in ROI now: {current_in_roi}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # TỔNG SỐ DUY NHẤT ĐÃ ĐẾM 
        cv2.putText(frame_result, f'TOTAL COUNT: {total_unique}', (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Hiển thị cửa sổ duy nhất
        cv2.imshow("Nhan dang anh & Tong hop", frame_result)
    
    # Xử lý phím
    key = cv2.waitKey(50)
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE
        paused = not paused
       
# In kết quả cuối cùng
print("KẾT QUẢ GIÁM SÁT\n")
print(f"Tổng số đối tượng DUY NHẤT đã đi qua ROI: {total_unique_count}")
print(f"Tổng số frame đã xử lý: {frame_count}")
print("="*50)

cap.release()
cv2.destroyAllWindows()
od.close()