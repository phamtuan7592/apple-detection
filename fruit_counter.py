import cv2
import numpy as np
from object_detection import ObjectDetector
import sys
import os
import json
import csv
from datetime import datetime

# === CẤU HÌNH HỆ THỐNG ===
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

print("Đang tải mô hình YOLO (Phân loại trái cây)...")
od = ObjectDetector(model_path="best.pt", min_score_thresh=0.35) # Dùng file best.pt train cho trái cây của bạn

print("Khởi tạo bộ lọc MOG2 (Background Subtractor)...")
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 

# === CẤU TRÚC DỮ LIỆU ĐẾM (NÂNG CẤP) ===
counted_ids = set()        # Tập hợp chứa các ID đã được đếm
class_counts = {}          # Dictionary đếm số lượng từng loại trái cây (VD: {'Táo': 5, 'Cam': 3})
total_unique_count = 0     # Tổng toàn bộ

def is_inside_roi(box, roi):
    """Kiểm tra tâm của bounding box có nằm trong ROI không"""
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0

# --- MENU CHỌN NGUỒN ---
print("\n" + "="*40)
print("HỆ THỐNG KIỂM ĐẾM TRÁI CÂY THÔNG MINH")
print("="*40)
print("1. Sử dụng Webcam (Real-time)")
print("2. Sử dụng File Video")
print("3. Thoát")
choice = input("Nhập lựa chọn của bạn: ")

if choice == '1':
    cap = cv2.VideoCapture(0)
elif choice == '2':
    video_path = "demo1.mkv" # Thay bằng video trái cây trên băng chuyền của bạn
    cap = cv2.VideoCapture(video_path)
else:
    sys.exit()

# --- TẢI VÙNG ROI ---
if os.path.exists("roi.json"):
    with open("roi.json", "r") as f:
        data = json.load(f)
        ROI_original = np.array(data["ROI"], dtype=np.int32)
else:
    ROI_original = np.array([[0, 278], [1277, 145], [1266, 718], [6, 713], [1, 280]], dtype=np.int32)

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale_x = DESIRED_WIDTH / original_width
scale_y = DESIRED_HEIGHT / original_height

cv2.namedWindow("He Thong Dem Trai Cay", cv2.WINDOW_NORMAL)

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
        
        # 1. Bộ lọc chuyển động (Ngầm)
        blurred = cv2.GaussianBlur(frame_display, (5, 5), 0)
        fg_mask = backSub.apply(blurred)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   
        edges = cv2.Canny(fg_mask, 50, 150)
        
        # 2. Nhận dạng & Tracking (Sử dụng Tracker của YOLO)
        bboxes, class_ids, scores, track_ids = od.detect(frame, conf=0.35)
        
        # Danh sách dữ liệu hợp lệ sau lọc
        val_bboxes, val_ids, val_scores, val_tids = [], [], [], []
        current_in_roi = 0
        
        # 3. Lọc nhiễu và đếm trái cây
        for box, cid, score, tid in zip(bboxes, class_ids, scores, track_ids):
            if tid == -1: continue # Bỏ qua đối tượng chưa có ID ổn định
            
            x_disp = int(box[0] * scale_x)
            y_disp = int(box[1] * scale_y)
            w_disp = int(box[2] * scale_x)
            h_disp = int(box[3] * scale_y)
            
            # Giới hạn khung hình
            x_disp = max(0, min(x_disp, DESIRED_WIDTH-1))
            y_disp = max(0, min(y_disp, DESIRED_HEIGHT-1))
            w_disp = min(w_disp, DESIRED_WIDTH - x_disp)
            h_disp = min(h_disp, DESIRED_HEIGHT - y_disp)
            
            if w_disp <= 0 or h_disp <= 0: continue
                
            # Bộ lọc của bạn: Kiểm tra xem vật thể có đang chuyển động trên băng chuyền không
            roi_fg = fg_mask[y_disp:y_disp+h_disp, x_disp:x_disp+w_disp]
            total_pixels = w_disp * h_disp
            motion_ratio = np.sum(roi_fg == 255) / (total_pixels + 1e-6)
            
            if motion_ratio > 0.05: # Vật thể thực sự di chuyển
                current_box = [x_disp, y_disp, w_disp, h_disp]
                
                val_bboxes.append(current_box)
                val_ids.append(cid)
                val_scores.append(score)
                val_tids.append(tid)
                
                # NẾU TRÁI CÂY NẰM TRONG ROI VÀ CHƯA ĐƯỢC ĐẾM
                if is_inside_roi(current_box, ROI_display):
                    current_in_roi += 1
                    
                    if tid not in counted_ids:
                        counted_ids.add(tid)
                        total_unique_count += 1
                        
                        # Tăng bộ đếm cho loại trái cây tương ứng
                        class_name = od.classes.get(cid, "Unknown")
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
        
        # 4. Đồ họa (UI Dashboard)
        frame_result = frame_display.copy()
        
        # Vẽ ROI
        cv2.polylines(frame_result, [ROI_display], True, (255, 0, 255), 2)
        
        # Vẽ Box
        frame_result = od.draw_boxes(frame_result, val_bboxes, val_ids, val_scores, val_tids, line_thickness=2)
        
        # VẼ BẢNG THỐNG KÊ (DASHBOARD LÊN MÀN HÌNH)
        # Khung đen mờ
        cv2.rectangle(frame_result, (10, 10), (350, 150 + len(class_counts)*35), (0, 0, 0), -1)
        
        # Tiêu đề
        cv2.putText(frame_result, "THONG KE TRAI CAY", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame_result, f"Tong trong hinh: {current_in_roi}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Lặp in số lượng từng loại trái cây
        y_offset = 110
        for fruit_name, count in class_counts.items():
            cv2.putText(frame_result, f"- {fruit_name}: {count}", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 35
            
        cv2.putText(frame_result, "-"*20, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(frame_result, f"TONG CONG: {total_unique_count}", (20, y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow("He Thong Dem Trai Cay", frame_result)
    
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # Spacebar
        paused = not paused

cap.release()
cv2.destroyAllWindows()
od.close()

# === KẾT THÚC VÀ XUẤT BÁO CÁO ===
print("\n" + "="*50)
print("KẾT QUẢ GIÁM SÁT THU HOẠCH:\n")
print(f"Tổng cộng đã đếm: {total_unique_count} trái cây")
for fruit, count in class_counts.items():
    print(f" - {fruit}: {count}")
print("="*50)

# XUẤT FILE CSV
if total_unique_count > 0:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"BaoCao_TraiCay_{timestamp}.csv"
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = cv2.csv.writer if hasattr(cv2, 'csv') else csv.writer(file)
        writer.writerow(["Loai Trai Cay", "So Luong"])
        for fruit, count in class_counts.items():
            writer.writerow([fruit, count])
        writer.writerow(["TONG CONG", total_unique_count])
        writer.writerow(["Thoi gian", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        
    print(f"[THÀNH CÔNG] Đã xuất báo cáo ra file: {csv_filename}")