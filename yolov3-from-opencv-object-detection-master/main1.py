import cv2
import os
import threading
import serial
import time
import numpy as np
import util
import easyocr
import keyboard
import pyfirmata  # Import pyfirmata for Arduino communication
from deepface import DeepFace
import time

class FacialRecognitionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.label = None
        self.recognized_user = None
        self.start_recognition = True  # Flag to start facial recognition

    def run(self):
        face_labels = {}
        face_dataset_path = r'C:\Users\Wei Yi\PycharmProjects\Car1\yolov3-from-opencv-object-detection-master\Face'
        for label in os.listdir(face_dataset_path):
            label_path = os.path.join(face_dataset_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    if image_file.endswith('.jpg') or image_file.endswith('.png'):
                        face_labels[label] = os.path.join(label_path, image_file)

        cap = cv2.VideoCapture(1)  # Use the default webcam (index 1) for facial recognition

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.start_recognition:
                faces = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(frame, 1.3, 5)
                recognized = False
                for (x, y, w, h) in faces:
                    face_img = frame[y:y + h, x:x + w]
                    face_img = cv2.resize(face_img, (160, 160))
                    for name, image_path in face_labels.items():
                        result = DeepFace.verify(face_img, image_path, model_name='Facenet', enforce_detection=False)
                        if result['verified']:
                            self.label = 1  # Facial recognition allowed
                            self.recognized_user = name
                            recognized = True
                            break
                    if recognized:
                        break
                if recognized:
                    print(f"Facial Recognition: {self.recognized_user}")
                    break
                else:
                    self.label = 0  # Unknown user, entry denied

            cv2.imshow('Facial Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class PlateRecognitionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.label = None
        self.recognition_message = None
        self.start_recognition = True  # Flag to start plate recognition

    def run(self):
        dataset_folder = r"C:\Users\Wei Yi\PycharmProjects\Car1\yolov3-from-opencv-object-detection-master\Car Image"

        cap = cv2.VideoCapture(0)  # Use the USB webcam (index 0) for plate recognition

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.start_recognition:
                cv2.imshow('License Plate Recognition', frame)

                if cv2.waitKey(1) & 0xFF == ord(' '):
                    img_path = os.path.join(dataset_folder, "1.png")
                    cv2.imwrite(img_path, frame)
                    print(f"Image 1.png saved.")
                    self.recognition_message, self.label = self.process_captured_image(img_path)
                    break

        cap.release()
        cv2.destroyAllWindows()

    def process_captured_image(self, img_path):
        allowed_plates = {
            "PJP 1382": "Alex Lee",
            "RX 6326": "Shermaine Tan",
            "PLJ 7123": "Wei Yi",
        }

        model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
        model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
        class_names_path = os.path.join('.', 'model', 'class.names')

        with open(class_names_path, 'r') as f:
            class_names = [j[:-1] for j in f.readlines() if len(j) > 2]

        net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
        net.setInput(blob)
        detections = util.get_outputs(net)
        bboxes = []
        class_ids = []
        scores = []
        for detection in detections:
            bbox = detection[:4]
            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
            bbox_confidence = detection[4]
            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])
            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)
        bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)
        reader = easyocr.Reader(['en'])
        recognized_plate = False
        for bbox_, bbox in enumerate(bboxes):
            xc, yc, w, h = bbox
            img = cv2.rectangle(img,
                                (int(xc - (w / 2)), int(yc - (h / 2))),
                                (int(xc + (w / 2)), int(yc + (h / 2))),
                                (0, 255, 0),
                                10)
            license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
            license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
            _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
            output = reader.readtext(license_plate_gray)
            for out in output:
                text_bbox, text, text_score = out
                print(text, text_score)
                if text in allowed_plates:
                    self.label = 1  # Plate recognition allowed
                    self.recognition_message = f"License Plate: {text} | Owner: {allowed_plates[text]}"
                    recognized_plate = True
                    break
                else:
                    self.label = 0  # Unauthorized car, entry denied
                    self.recognition_message = f"Unauthorized Car: License Plate - {text}"
        if recognized_plate:
            return self.recognition_message, self.label
        else:
            return "Error, please scan again.", 0  # Plate not recognized

if __name__ == "__main__":
    # Initialize serial communication with Arduino
    ser = serial.Serial('COM8', 9600)
    time.sleep(1)

    while True:  # Loop until 'y' key is pressed
        # Wait for signal '1' from Arduino to start recognition
        while ser.readline().decode().strip() != '1':
            pass
        print("Received signal '1'. Starting recognition...")

        if keyboard.is_pressed('y'):
            break

        # Initialize facial recognition thread
        facial_thread = FacialRecognitionThread()
        # Initialize plate recognition thread only if facial recognition was successful
        plate_thread = PlateRecognitionThread()

        # Start facial recognition thread
        facial_thread.start()

        # Wait for facial recognition thread to finish
        facial_thread.join()

        # Check the result of facial recognition
        if facial_thread.label == 1:
            # Start plate recognition thread
            plate_thread.start()

            # Wait for plate recognition thread to finish
            plate_thread.join()

            # Check if both facial and plate recognition were successful
            entry_status = int(facial_thread.label) + int(plate_thread.label)

            if entry_status == 2:
                print("Allowed Entry")
                ser.write(b'1')
                time.sleep(5)
            else:
                print("Unauthorized User.")
                ser.write(b'2')
                time.sleep(5)
        else:
            print("Facial recognition failed. Cannot proceed with plate recognition.")
