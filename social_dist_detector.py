import numpy as np
import cv2
import imutils
import os
import time

def check_distance(a,  b):

    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2       
    
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False

def intial_setup():
    global net, ln, LABELS

    weights = "yolo/yolov3.weights"
    config = "yolo/yolov3.cfg"
    labelsPath = "yolo/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")  
    
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def image_processing(image):

    global processedImg
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    confidences = []
    outline = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)

    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = [] 
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)

        for i in range(len(center)):
            for j in range(len(center)):
                close = check_distance(center[i], center[j])

                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
        index = 0

        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                org = (150, 250) 
                # fontScale 
                fontScale = 1
                # Blue color in BGR 
                color = (0, 0, 255) 
                # Line thickness of 2 px 
                thickness = 2
                x,y,w,h = 130,220,175,60
              # Draw black background rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,0), -1)
                # Using cv2.putText() method 
                cv2.putText(frame, 'Warning', org, font,  
                                fontScale, color, thickness, cv2.LINE_AA) 
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    processedImg = frame.copy()
            



def sd_gen():
    """Video streaming generator function."""
    frame_number = 0
    filename = "videos/example.mp4"

    cap = cv2.VideoCapture(filename)
    # cap = cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        current_img = frame.copy()
        current_img = imutils.resize(current_img, width=480)
        video = current_img.shape
        frame_number += 1
        Frame = current_img

        if(frame_number%5 == 0 or frame_number == 1):

            intial_setup()
            image_processing(current_img)
            Frame = processedImg

        frame = cv2.imencode('.jpg', processedImg)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
              break