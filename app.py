from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import time
# YOLOv5 imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Thai-coin-detection-main', 'yolov5'))
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, plot_one_box, xyxy2xywh
from utils.torch_utils import select_device, time_synchronized
import torch

app = Flask(__name__)

# ตั้งค่าตามไฟล์ run.py - ปรับลดค่า confidence และ threshold
imgsz = 640
my_confidence = 0.25  # ลดจาก 0.80 เป็น 0.25
my_threshold = 0.30   # ลดจาก 0.45 เป็น 0.30
my_filterclasses = None
my_weight = os.path.join(
    os.path.dirname(__file__),
    'Thai-coin-detection-main', 'trained', 'trained_v1-9_set1+2_90img_600e', 'weights', 'best.pt'
)

# Load model
device = select_device('')
model = attempt_load(my_weight, map_location=device)
imgsz = 640  # check_img_size(imgsz, s=model.stride.max())

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [
    (232, 182, 0),  # 5Baht
    (0, 204, 255),  # 1Baht
    (69, 77, 246),  # 10Baht
    (51, 136, 222), # 2Baht
    (222, 51, 188), # .50Baht
]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def main_process(input_img):
    """ฟังก์ชันประมวลผลตามไฟล์ run.py - ปรับปรุงแล้ว"""
    try:
        img0 = input_img.copy()

        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=True)[0]
        
        # ปรับ NMS parameters เพื่อให้ detect ได้หลายเหรียญ
        pred = non_max_suppression(
            pred, 
            my_confidence, 
            my_threshold, 
            classes=my_filterclasses, 
            agnostic=False  # เปลี่ยนเป็น False เพื่อแยกแต่ละ class
        )
        t2 = time_synchronized()

        total = 0
        class_count = [0 for _ in range(len(names))]
        breakdown = {}
        detection_details = []  # เก็บรายละเอียดการ detect
        
        print(f"Number of predictions: {len(pred)}")  # Debug
        
        for i, det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                print(f"Detected {len(det)} objects with confidence > {my_confidence}")  # Debug info
                
                for *xyxy, conf, cls in reversed(det):
                    class_idx = int(cls)
                    class_count[class_idx] += 1
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    
                    # ปรับการแสดงผล label
                    coin_name = names[class_idx]
                    label = f'{coin_name} ({conf:.2f})'
                    
                    # คำนวณค่าเหรียญ
                    try:
                        coin_value = int(coin_name.replace('baht', '').replace('Baht', '').strip())
                    except:
                        coin_value = 0
                    
                    total += coin_value
                    
                    # วาดกรอบและ label
                    plot_one_box(xyxy, img0, label=label, color=colors[class_idx], line_thickness=2)
                    
                    # เพิ่มใน breakdown
                    breakdown[coin_name] = breakdown.get(coin_name, 0) + 1
                    
                    # เก็บรายละเอียด
                    detection_details.append({
                        'class': coin_name,
                        'confidence': float(conf),
                        'bbox': [int(x) for x in xyxy],
                        'value': coin_value
                    })
                    
                    print(f"Detected: {coin_name} with confidence {conf:.3f}")  # Debug
            else:
                print(f"No detections in prediction {i}")  # Debug
        
        # แสดงข้อมูลสรุปบนภาพ
        y_offset = 30
        for i, class_name in enumerate(names):
            if class_count[i] > 0:
                text = f"{class_name}: {class_count[i]} coin(s)"
                img0 = cv2.putText(img0, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        # แสดงยอดรวม
        img0 = cv2.putText(img0, f"Total: {total} Baht", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        img0 = cv2.putText(img0, f"Coins: {sum(class_count)}", (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        print(f"Detection summary: {breakdown}")  # Debug info
        print(f"Total value: {total}, Total coins: {sum(class_count)}")  # Debug info
        
        return img0, total, breakdown, class_count, detection_details
        
    except Exception as e:
        print(f"Error in main_process: {e}")
        # Return default values in case of error
        return input_img, 0, {}, [0] * len(names), []

@app.route('/detect-coin-image', methods=['POST'])
def detect_coin_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img0 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img0 is None:
        return jsonify({'error': 'Invalid image'}), 400

    # ใช้ฟังก์ชัน main_process ที่ปรับปรุงแล้ว
    img_result, total_value, breakdown, class_count, detection_details = main_process(img0)

    # แปลงภาพเป็น base64
    _, buffer = cv2.imencode('.jpg', img_result)
    img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    # สร้าง breakdown ในรูปแบบที่ frontend ต้องการ
    breakdown_list = []
    for coin_name, count in breakdown.items():
        try:
            coin_value = int(coin_name.replace('baht', '').replace('Baht', '').replace('บาท', '').strip())
        except:
            coin_value = 0
        breakdown_list.append({
            'value': coin_value,
            'count': count,
            'name': coin_name
        })
    
    return jsonify({
        'image': img_base64, 
        'breakdown': breakdown_list, 
        'total_value': total_value,
        'total_coins': sum(breakdown.values()),
        'class_count': class_count,
        'detection_details': detection_details,  # เพิ่มรายละเอียด
        'debug_info': {
            'confidence_threshold': my_confidence,
            'nms_threshold': my_threshold,
            'total_detections': len(detection_details)
        }
    })

@app.route('/detect-coins', methods=['POST'])
def detect_coins():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # ใช้ฟังก์ชัน main_process ที่ปรับปรุงแล้ว
    img_result, total_value, breakdown, class_count, detection_details = main_process(img)

    # แปลงภาพเป็น base64
    _, buffer = cv2.imencode('.jpg', img_result)
    img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

    # สร้าง breakdown ในรูปแบบที่ frontend ต้องการ
    breakdown_list = []
    for coin_name, count in breakdown.items():
        try:
            coin_value = int(coin_name.replace('baht', '').replace('Baht', '').replace('บาท', '').strip())
        except:
            coin_value = 0
        breakdown_list.append({
            'value': coin_value,
            'count': count,
            'name': coin_name
        })

    return jsonify({
        'coins': [],  # ใช้ YOLOv5 แทน OpenCV circles
        'total': sum(breakdown.values()),
        'breakdown': breakdown_list,
        'total_value': total_value,
        'image': img_base64,
        'detection_details': detection_details
    })

# เพิ่ม route สำหรับปรับค่า threshold แบบ real-time
@app.route('/adjust-threshold', methods=['POST'])
def adjust_threshold():
    global my_confidence, my_threshold
    
    data = request.json
    if 'confidence' in data:
        my_confidence = float(data['confidence'])
    if 'threshold' in data:
        my_threshold = float(data['threshold'])
    
    return jsonify({
        'confidence': my_confidence,
        'threshold': my_threshold,
        'message': 'Thresholds updated successfully'
    })

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
