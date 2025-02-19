import cv2
import torch
from ultralytics import YOLO
import time

# تحميل نموذج YOLOv5
model = YOLO("yolov5s.pt")

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# التحقق من تشغيل الكاميرا بنجاح
if not cap.isOpened():
    print("Error: Could not open the camera. Please check the connection.")
    exit()
else:
    print("Camera opened successfully.")

# مؤقت لتنفيذ الكشف كل 3 ثوانٍ
last_capture_time = time.time()

while True:
    # التقاط إطار من الكاميرا
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    # تنفيذ الكشف كل 3 ثوانٍ
    current_time = time.time()
    if current_time - last_capture_time >= 3:
        last_capture_time = current_time  # تحديث المؤقت
        
        # تنفيذ YOLO على الإطار
        results = model(frame)

        # عدّ السيارات المكتشفة فقط
        car_count = 0
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # الحصول على رقم الفئة
                if class_id == 2:  # الفئة 2 تمثل السيارات في COCO
                    car_count += 1
                    # استخراج إحداثيات الصندوق المحيط
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # رسم مستطيل حول السيارة
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # إضافة اسم الفئة فوق المستطيل
                    cv2.putText(frame, "Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # طباعة عدد السيارات المكتشفة
        print(f"🚗 Number of cars detected: {car_count}")

    # عرض الإطار مع الكشف عن السيارات
    cv2.imshow("Car Detection", frame)

    # الخروج عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير الكاميرا وإغلاق النوافذ
cap.release()
cv2.destroyAllWindows()
