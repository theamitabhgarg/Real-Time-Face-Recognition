import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for face in faces:
        x, y, w, h = face

        offset = 10

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(frame.shape[1], x + w + offset)
        y2 = min(frame.shape[0], y + h + offset)

        face_offset = frame[y1:y2, x1:x2]

        if face_offset.size > 0:
            face_selection = cv2.resize(face_offset, (100, 100))
            cv2.imshow("Face", face_selection)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()