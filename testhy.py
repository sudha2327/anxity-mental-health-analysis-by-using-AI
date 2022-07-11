import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
alg="haarcascade_eye.xml"
#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')

haar_cascade=cv2.CascadeClassifier(alg)
cam=cv2.VideoCapture(0)
count_anx=0
count_desp=0
count_normal=0
while True:
    _,img=cam.read()
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(gray_img,1.32,5)

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=3)
        roi_gray=gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #normalizing
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])

        emotions = ('moderate', 'mild', 'mild', 'Depressed', 'Depressed', 'neutral', 'neutral')
        predicted_emotion = emotions[max_index]
        if predicted_emotion=='mild':
            s="mild: 7%"
            print("depressed-level"+s)
            
             
        elif predicted_emotion=='Depressed':
            s="Sever : 27%"
            print("depressed-level"+s)
        
            
        elif predicted_emotion=='neutral':
            s="normal : 7%"
            print("depressed-level"+s)
            
        elif predicted_emotion=='moderate':
            s="Moderate: 16%"
            print("depressed-level"+s);
            
        else:
            s="normal 8%"
            print("depressed-level"+s);
            
            
        cv2.putText(img,s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        resized_img = cv2.resize(img, (1000, 700))
        
        cv2.imshow('Facial emotion analysis ',resized_img)
        cv2.setWindowProperty("Facial emotion analysis ", cv2.WND_PROP_TOPMOST, 1)


    # text=cv2.putText(img,str1,(10,20),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(0,0,255),2)
    # cv2.imshow("Facedetection",img)
    if cv2.waitKey(1) & 0xFF==ord('s'):
        cv2.imwrite('sudha.jpg',img)
        break;

    key=cv2.waitKey(10)

    if key==27:
        break

cam.release()
cv2.destroyAllWindows()
    
