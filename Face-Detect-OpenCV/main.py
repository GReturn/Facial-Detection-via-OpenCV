import time
import cv2
import pickle
import numpy as np

# Start-up

print("[INFO] Setting up video capture...")

device = 1
video = cv2.VideoCapture(device, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(0.5)

print("[INFO] Activating Haar Cascade classifiers...")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

time.sleep(1)

# End start-up

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


font = cv2.FONT_HERSHEY_TRIPLEX
display_face = '\"face\"'
display_eye = '\"eye\"'
color = (127, 100, 255)
stroke = 2

if not video.isOpened():
    video.open(device)

while video.isOpened():
    ret, frame = video.read()

    if ret == False:
        print("Live video feed crashed. Possible cause: faulty driver.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ROI - Region Of Interest
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)

        if confidence >= 45 and confidence <= 85:
            name = (labels[id_])
            cv2.putText(frame, name + " " + str(np.floor(confidence)) + "%", (x, y), font,
                        0.85, color, stroke, cv2.LINE_AA)
        else:
            cv2.putText(frame, display_face, (x, y), font,
                        1, color, stroke, cv2.LINE_AA)

        # eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
