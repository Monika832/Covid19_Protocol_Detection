import cv2
import numpy as np
import time
import copy
import imutils
from scipy.spatial import distance as dist

# Load Yolo
net = cv2.dnn.readNet("yolov4-tiny-obj_final.weights", "yolov4-tiny-obj.cfg")
classes = ["without_mask","with_mask"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
MIN_DIST=100

def processImage(img):
# Loading image
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

# Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


 # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    centroids=[]
    results=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
               # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

 # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                centroids.append((center_x, center_y))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = [(0,0,255),(0,255,0)]
    font = cv2.FONT_HERSHEY_PLAIN

    if len(indexes) > 0:
       for i in indexes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    violate=set()

    if len(results) >=2:
        centr=np.array([r[2] for r in results])
        d=dist.cdist(centr, centr, metric="euclidean")
        for i in range(0, d.shape[0]):
            for j in range(i+1, d.shape[1]):
                if(d[i,j]< MIN_DIST):
                    violate.add(i)
                    violate.add(j)
        for (i, (wgh, coor, c)) in enumerate(results):
            (x,y,w,h)= coor
            (cx,cy)=c
            t="No"
            col=(0,255,0)
            if i in violate:
                t="Yes"
                col=(0,0,255)
            cv2.putText(img, t, (x,y-10),font,0.85, col, 2)

    for i in range(len(boxes)):
       if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
    cv2.putText(img, "Social Distancing Violations: {}".format(len(violate)), (10, img.shape[0] - 25), font, 0.85, (0, 0, 255), 3)
    return img

# define a video capture object
# vid = cv2.VideoCapture("video.mp4")
# #vid = cv2.VideoCapture("video2.mp4")
# #vid = cv2.VideoCapture(0)
# frame_width = int(vid.get(3))
# frame_height = int(vid.get(4))
# while(True):
#     start = time.time()
#     ret, frame = vid.read()
#     width,height,channel = frame.shape
#     frame = processImage(frame)
#     frame = cv2.resize(frame,(height,width))
#     end = time.time()
#     cv2.putText(frame,f"{round(1/(end-start),2)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1,1)
#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

# vid.release()
# cv2.destroyAllWindows()


img = cv2.imread("crowd.jpg")
width,height,channel = img.shape
img = processImage(img)
img = cv2.resize(img,(height,width))
cv2.imshow("img",img)
cv2.waitKey(0)