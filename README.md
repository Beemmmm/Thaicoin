# ระบบตรวจนับเหรียญไทย (Thai Coin Counting System)

## 📝 คำอธิบาย (Description)

โปรเจกต์นี้คือเว็บแอปพลิเคชันสำหรับตรวจจับและนับเหรียญไทยจากภาพถ่าย พัฒนาขึ้นเพื่อแก้ปัญหาการนับเหรียญด้วยมือที่ใช้เวลานานและอาจเกิดข้อผิดพลาด ระบบใช้เทคโนโลยีการตรวจจับวัตถุ (Object Detection) ด้วยโมเดล **YOLOv5** เพื่อจำแนกชนิดของเหรียญ (1, 2, 5, 10 บาท) และสรุปมูลค่ารวมให้อัตโนมัติ

---

## ✨ คุณสมบัติหลัก (Features)

* **อัปโหลดรูปภาพ**: ผู้ใช้สามารถอัปโหลดไฟล์รูปภาพเหรียญจากอุปกรณ์ของตนเองได้
* **ตรวจจับและจำแนกเหรียญ**: ใช้โมเดล YOLOv5 เพื่อค้นหาและระบุชนิดของเหรียญแต่ละเหรียญในภาพ
* **สรุปผลลัพธ์**:
    * แสดงภาพผลลัพธ์พร้อมกรอบ (Bounding Box) ตีล้อมรอบเหรียญที่ตรวจพบ
    * แสดงจำนวนเหรียญแต่ละชนิด
    * คำนวณและแสดงมูลค่ารวมของเหรียญทั้งหมด

---

## 🛠️ เทคโนโลยีที่ใช้ (Technology Stack)

* **Backend**: Python, Flask
* **AI Model**: YOLOv5
* **AI & Image Processing**: PyTorch, OpenCV, NumPy
* **Frontend**: HTML
* **Development Tools**: Visual Studio Code, LabelImg

---
## 🙏 กิตติกรรมประกาศ (Acknowledgements)

โปรเจกต์นี้ได้ใช้ชุดข้อมูล "Thai Coin Detection" ที่พัฒนาโดยคุณ **Saharat S.** ขอขอบคุณสำหรับการแบ่งปันชุดข้อมูลที่มีประโยชน์นี้สู่สาธารณะ

* **Dataset Repository:** [saharatss/Thai-coin-detection](https://github.com/saharatss/Thai-coin-detection)
---
## ⚙️ การติดตั้งและใช้งาน (Installation & Usage)

### **ข้อกำหนดเบื้องต้น (Prerequisites)**

* Python 3.8+
* Git

### **ขั้นตอนการติดตั้ง (Setup)**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Beemmmm/Thaicoin.git](https://github.com/Beemmmm/Thaicoin.git)
    cd Thaicoin
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
   
    ```bash
    pip install -r requirements.txt
    ```

### **การรันโปรเจกต์ (Running the Application)**

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```

2.  **เปิดเว็บเบราว์เซอร์** และเข้าไปที่:
    `http://127.0.0.1:5000`

3.  **ใช้งาน:**
    คลิกปุ่มเพื่ออัปโหลดรูปภาพเหรียญ ระบบจะประมวลผลและแสดงผลลัพธ์

---


## ⛔ ข้อจำกัดของระบบ (Limitations)

* ระบบไม่สามารถตรวจนับเหรียญที่ซ้อนทับกันได้
* ความแม่นยำอาจลดลงในสภาพแสงที่ไม่เหมาะสม
* รองรับเฉพาะเหรียญไทยสกุลเงินบาทรุ่นปัจจุบันเท่านั้น
* ไม่รองรับการประมวลผลวิดีโอ