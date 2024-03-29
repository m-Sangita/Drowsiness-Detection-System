import cv2
import numpy as np
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from pygame import mixer
import smtplib
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

sender_email="msangita426@gmail.com"
receiver_email="msangita426@gmail.com"
password="eqfedpdaqqwyfxtf"
m="Your driver is drowsy"
mixer.init()
sound1= mixer.Sound(r'D:\python\alarm.wav')
sound2= mixer.Sound(r'D:\python\alarm_01.wav')
score=0
cap = cv2.VideoCapture(0)
model = load_model(r'D:\python\model.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
color=(0,0,0)
server=smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login(sender_email,password)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        upper_lip=[]
        lower_lip=[]
        for i in range(50, 53):
            upper_lip.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(58, 61):
            lower_lip.append((landmarks.part(i).x, landmarks.part(i).y))
        upper_lip_mean = np.mean(upper_lip, axis=0)
        lower_lip_mean = np.mean(lower_lip, axis=0)
        lip_dist = euclidean_dist(upper_lip_mean, lower_lip_mean)
        left_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_points = [42,43,44,45,46,47]
        x1 = [landmarks.part(point).x for point in left_eye_points]
        y1 = [landmarks.part(point).y for point in left_eye_points]
        x1_min, y1_min = min(x1), min(y1)
        x1_max, y1_max = max(x1), max(y1)
        cv2.rectangle(frame, (x1_min, y1_min-10), (x1_max, y1_max+10), (0, 255, 0), 2)
        x2 = [landmarks.part(point).x for point in right_eye_points]
        y2 = [landmarks.part(point).y for point in right_eye_points]
        x2_min, y2_min = min(x2), min(y2)
        x2_max, y2_max = max(x2), max(y2)
        cv2.rectangle(frame, (x2_min, y2_min-10), (x2_max, y2_max+10), (0, 255, 0), 2)
        left_eye=frame[y1_min-10:y1_max+10,x1_min:x1_max]
        left_eye= cv2.resize(left_eye,(80,80))
        left_eye=left_eye/255.0
        left_eye=left_eye.reshape((80,80,3))
        left_eye= np.expand_dims(left_eye,axis=0)
        right_eye=frame[y2_min-10:y2_max+10,x2_min:x2_max]
        right_eye= cv2.resize(right_eye,(80,80))
        right_eye=right_eye/255.0
        right_eye=left_eye.reshape((80,80,3))
        right_eye= np.expand_dims(right_eye,axis=0)
        prediction1=model.predict(left_eye)
        prediction2=model.predict(right_eye)
        print("Left eye:", prediction1)
        print("right_eye:",prediction2)
        if(prediction1[0][0]>0.5 and prediction2[0][0]>0.5):
            print(prediction1)
            print(prediction2)
            print("opened")
            score=0
        if(prediction1[0][1]>0.4 and prediction2[0][1]>0.4):
            print(prediction1)
            print(prediction2)
            print("closed")
            score=score+1
        if(score>=10 and lip_dist>40):
            sound1.play()
            server.sendmail(sender_email,receiver_email,m)
            score=0
        elif(score<10 and lip_dist>40):
            sound2.play()
        elif(score>10 and lip_dist<40):
            sound1.play()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if cv2.waitKey(33) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
