from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

import os
import cv2
import time
import imutils
import argparse
import numpy as np

from math import pow, sqrt
from imutils.video import FPS






facenet = cv2.dnn.readNet( 'caffe/deploy.prototxt',  'caffe/res10_300x300_ssd_iter_140000.caffemodel')
model=load_model('model')



def predict_mask(frame, facenet, model):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.9))
    facenet.setInput(blob)
    detections = facenet.forward()
    faces=[]
    coordinates=[]
    predictions=[]

    for i in range(0, detections.shape[2]):
        confidence= detections[0,0,i,2]
        if  confidence > 0.5:
            rectangle=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (X, y, endX, endY) = rectangle.astype('int')
            (X,y)=(max(0,X),max(0,y))
            (endX, endY)=(min(w-1,endX), min(h-1, endY))
            face=frame[y:endY, X:endX]
            frame=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            face=np.expand_dims(face,axis=0)
            faces.append(face)
            coordinates.append((X,y,endX,endY))
    if len(faces)>0:
        predictions=model.predict(faces)
    return (coordinates, predictions)



def gen():
    
    #Initialize Video Stream (Use src = 0 for Webcam or src = 'path to video input')
    print('[Status] Starting Video Stream...')
    vs = VideoStream(src = 0).start()
    
    fps = FPS().start()

    #Loop Video Stream
    while True:
        frame=vs.read()
        frame=imutils.resize(frame, width=400)

        (coordinates, predictions)= predict_mask(frame,facenet,model)

        for(rect , predict) in zip (coordinates, predictions):
            (X, y, endX, endY) = rect
            (mask, withoutMask) = predict
            label = "Mask" if mask > withoutMask else "Without Mask"
            color = (0, 255, 0) if label == "Mask" else (0,0,255)
            label="{}: {:.2f}%".format(label, max(mask,withoutMask)*100)
            cv2.putText(frame, label, (X, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (X,y), (endX, endY), color, 2)
        # cv2.imshow("Mask Detector", frame)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        fps.update()


    fps.stop()