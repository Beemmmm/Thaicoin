import cv2
import numpy as np
import os
import time
import sys
import torch

# เพิ่ม path สำหรับ YOLOv5 imports
# *ต้องแน่ใจว่าโฟลเดอร์ 'Thai-coin-detection-main/yolov5' อยู่ในตำแหน่งที่ถูกต้องเทียบกับไฟล์นี้*
sys.path.append(os.path.join(os.path.dirname(__file__), 'Thai-coin-detection-main', 'yolov5'))

# YOLOv5 imports
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, plot_one_box, xyxy2xywh
from utils.torch_utils import select_device, time_synchronized

# -----------------
# Global Settings
# -----------------
imgsz = 640
my_confidence = 0.25  # Confidence threshold for detection
my_threshold = 0.30   # NMS IOU threshold
my_filterclasses = None
# กำหนด Path ไปยังไฟล์ weights ของโมเดล
my_weight = os.path.join(
    os.path.dirname(__file__),
    'Thai-coin-detection-main', 'trained', 'trained_v1-9_set1+2_90img_600e', 'weights', 'best.pt'
)

# -----------------
# Model Loading
# -----------------
# โหลดโมเดลไปยังอุปกรณ์ที่เหมาะสม (CPU หรือ GPU)
device = select_device('')
model = attempt_load(my_weight, map_location=device)
model.eval() # ตั้งค่าโมเดลเป็นโหมดประเมินผล

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [
    (232, 182, 0),  # 5Baht
    (0, 204, 255),  # 1Baht
    (69, 77, 246),  # 10Baht
    (51, 136, 222), # 2Baht
    (222, 51, 188), # .50Baht
]

# -----------------
# Main Process Function
# -----------------

def main_process(input_img):
    """
    ฟังก์ชันหลักในการประมวลผลภาพเพื่อตรวจจับเหรียญ
    
    Args:
        input_img (np.ndarray): รูปภาพในรูปแบบ OpenCV BGR (np.uint8)

    Returns:
        tuple: (
            img0_result,      # np.ndarray: รูปภาพที่มีกรอบและข้อมูลสรุป
            total_value,      # int: มูลค่ารวมของเหรียญที่ตรวจจับได้
            breakdown,        # dict: จำนวนเหรียญแยกตามชนิด (เช่น {'5Baht': 2})
            class_count,      # list: จำนวนเหรียญแยกตาม index class
            detection_details # list: รายละเอียดการตรวจจับทั้งหมด
        )
    """
    try:
        img0 = input_img.copy()

        # Pre-process: letterbox, BGR to RGB, transpose, contiguous array
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)

        # Convert to torch tensor
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad(): # ไม่จำเป็นต้องคำนวณ gradient ในช่วง inference
            pred = model(img, augment=True)[0]
        t2 = time_synchronized()
        print(f'Inference time: {(t2 - t1) * 1000:.1f}ms')

        # Apply NMS (Non-Maximum Suppression)
        pred = non_max_suppression(
            pred, 
            my_confidence, 
            my_threshold, 
            classes=my_filterclasses, 
            agnostic=False
        )
        
        total = 0
        class_count = [0 for _ in range(len(names))]
        breakdown = {}
        detection_details = [] 
        
        print(f"Number of predictions (after NMS): {len(pred)}")
        
        # Process detections
        for i, det in enumerate(pred):
            # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]] # สำหรับ normalization
            if det is not None and len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                print(f"Detected {len(det)} objects with confidence > {my_confidence}")
                
                for *xyxy, conf, cls in reversed(det):
                    class_idx = int(cls)
                    coin_name = names[class_idx]
                    
                    # คำนวณค่าเหรียญ
                    try:
                        # Extract numerical value from coin name (e.g., '5Baht' -> 5)
                        # รองรับเหรียญ 50 สตางค์
                        if coin_name == '.50Baht':
                            coin_value = 0.5
                        else:
                            coin_value = float(coin_name.replace('Baht', '').replace('baht', '').strip())
                    except:
                        coin_value = 0
                    
                    class_count[class_idx] += 1
                    total += coin_value
                    
                    # วาดกรอบและ label
                    label = f'{coin_name} ({conf:.2f})'
                    plot_one_box(xyxy, img0, label=label, color=colors[class_idx], line_thickness=2)
                    
                    # เพิ่มใน breakdown
                    breakdown[coin_name] = breakdown.get(coin_name, 0) + 1
                    
                    # เก็บรายละเอียด
                    detection_details.append({
                        'class': coin_name,
                        'confidence': float(conf),
                        # แปลงเป็น list ของ int ก่อนส่งออก
                        'bbox': [int(x) for x in xyxy],
                        'value': coin_value
                    })
                    
                    print(f"Detected: {coin_name} with confidence {conf:.3f}")
            else:
                print(f"No detections in prediction {i}") 
        
        # แสดงข้อมูลสรุปบนภาพ
        y_offset = 30
        for i, class_name in enumerate(names):
            if class_count[i] > 0:
                text = f"{class_name}: {class_count[i]} coin(s)"
                # ตรวจสอบว่า `img0` ยังเป็นภาพอยู่หรือไม่
                if img0 is not None:
                    img0 = cv2.putText(img0, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
        
        # แสดงยอดรวม
        if img0 is not None:
            img0 = cv2.putText(img0, f"Total Value: {total:.2f} Baht", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            img0 = cv2.putText(img0, f"Total Coins: {sum(class_count)}", (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        print(f"Detection summary: {breakdown}")
        print(f"Total value: {total:.2f}, Total coins: {sum(class_count)}")
        
        return img0, total, breakdown, class_count, detection_details
        
    except Exception as e:
        print(f"Error in main_process: {e}")
        # Return default values in case of error
        return input_img, 0, {}, [0] * len(names), []

# -----------------
# Example Usage (ถ้าต้องการทดสอบ)
# -----------------
if __name__ == '__main__':
    # ***ต้องเปลี่ยน 'path/to/your/image.jpg' เป็น path รูปภาพจริงของคุณ***
    # คุณสามารถสร้างรูปภาพทดสอบขึ้นมาก่อนได้
    try:
        # สมมติว่ามีไฟล์รูปภาพชื่อ 'test_coin.jpg' อยู่ในโฟลเดอร์เดียวกัน
        TEST_IMAGE_PATH = 'test_coin.jpg' 
        
        if os.path.exists(TEST_IMAGE_PATH):
            print(f"Loading image from {TEST_IMAGE_PATH}...")
            # โหลดรูปภาพ
            test_img = cv2.imread(TEST_IMAGE_PATH)

            if test_img is not None:
                # รันฟังก์ชันประมวลผลหลัก
                print("Starting detection process...")
                result_img, total_val, breakdown_dict, counts, details = main_process(test_img)
                print("\n--- Detection Results ---")
                print(f"Total Value: {total_val:.2f} Baht")
                print(f"Breakdown: {breakdown_dict}")

                # แสดงรูปภาพผลลัพธ์
                cv2.imshow("Coin Detection Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # บันทึกรูปภาพผลลัพธ์
                cv2.imwrite("result_test_coin.jpg", result_img)
                print("Result image saved as 'result_test_coin.jpg'")
                
            else:
                print(f"Error: Could not load image from {TEST_IMAGE_PATH}. Check file integrity.")
        else:
            print(f"Warning: Test image '{TEST_IMAGE_PATH}' not found. Please create a test image or adjust the path.")

    except Exception as e:
        print(f"An error occurred during example usage: {e}")