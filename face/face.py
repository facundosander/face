import cv2
import dlib
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_eyes_nose_mouth_points(shape):
    eyes_nose_mouth_points = []
    for i in range(36, 48):  # ojos
        eyes_nose_mouth_points.append((shape.part(i).x, shape.part(i).y))
    for i in range(60, 68):  # boca
        eyes_nose_mouth_points.append((shape.part(i).x, shape.part(i).y))

    return np.array(eyes_nose_mouth_points, dtype=np.int32)

cap = cv2.VideoCapture(0)  # Usa la cámara web predeterminada

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        eyes_nose_mouth_points = get_eyes_nose_mouth_points(shape)

        hull = cv2.convexHull(eyes_nose_mouth_points)
        cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)

    cv2.imshow("Seguimiento facial", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()