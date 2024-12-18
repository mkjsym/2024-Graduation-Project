from ultralytics import YOLO
import cvzone
import cv2
import math

#use mp4 video as source
cap = cv2.VideoCapture(r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-Graduation-Project/Sources/Data/fire.mp4')
#use webcam as source
# cap = cv2.VideoCapture(0)

model = YOLO(r'C:/Users/mkjsy/Desktop/YM/Source Code/VSCode/GitHub/2024-Graduation-Project/Sources/Data/fire_model.pt')
# Reading the classes
classnames = ['fire']

while cap.isOpened():
    success, frame = cap.read()
    if (not success):
        print('failed to read video')
        continue

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Getting bbox,confidence and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    cv2.imshow('frame', frame)

    #ESC를 눌러 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
