import cv2
from yolov5 import YOLOv5
import time

# Load the YOLOv5 model
model = YOLOv5("yolov5s.pt", device="cpu")  # Replace "cpu" with "cuda" if using a GPU

# Open the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera. Please check the connection.")
else:
    print("Camera opened successfully.")

# Initialize the timer
last_capture_time = time.time()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    # Check if 3 seconds have passed since the last capture
    current_time = time.time()
    if current_time - last_capture_time >= 3:  # Every 3 seconds
        # Update the last capture time
        last_capture_time = current_time

        # Detect vehicles in the frame
        results = model.predict(frame)

        # Count the detected vehicles
        car_count = 0
        for detection in results.pred[0]:
            class_id = int(detection[5])  # Class ID
            if class_id == 2:  # Class ID 2 represents cars in the COCO dataset
                car_count += 1

        # Display the number of detected vehicles
        print(f"Number of vehicles detected: {car_count}")

        # Display the frame with detection results
        results.show()

        # Save the frame as an image (optional)
        output_image_path = f"output/frame_{int(current_time)}.jpg"  # Replace with your save path
        cv2.imwrite(output_image_path, frame)
        print(f"Frame saved as: {output_image_path}")

    # Display the frame in a window
    cv2.imshow('Camera Feed', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()