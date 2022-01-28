import cv2 as cv
import numpy as np

import time
import sys
import os

from numpy.core.fromnumeric import size
from numpy.lib.function_base import disp

# define globals
ABS_PATH = '/'.join(os.path.abspath(__file__).split('/')[0:-2])
print(ABS_PATH)
PROJ_PATH=os.path.join(ABS_PATH,'yolov3')
CONFIG_PATH=os.path.join(ABS_PATH,PROJ_PATH,'cfg/yolov3.cfg')
WEIGHTS_PATH=os.path.join(ABS_PATH, PROJ_PATH,'weights/yolov3.weights')

CONFIDENCE=0.5
SCORE_THRESHOLD=0.5
IOU_THRESHOLD=0.5



def load_net(CONFIG_PATH,WEIGHTS_PATH):
  # load YOLOV3 Network
  net=cv.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
  # use cuDNN
  net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
  return net

def blobify(input,type='i',scale=0.5):
  if type == 'i':
    # image
    mat = cv.imread(input)
    h,w = mat.shape[:2]
    print("{} {}".format(w,h))
    # mat = cv.resize(mat, (int(h*scale),int(w*scale)))
    # w,h = mat.shape[:2]
  elif type == 'v':
    # video
    pass
  scale_factor = 1/255.0
  blob = cv.dnn.blobFromImage(mat, scale_factor, (416,416), swapRB=True,crop=False)
  # print dimensions
  print("image.shape:", mat.shape)
  print("blob.shape:", blob.shape)
  return (blob,mat,(w,h))

def blobify_stream(dev):
  _, frame = dev.read()
  w,h = frame.shape[:2]
  scale_factor = 1/255.0
  blob = cv.dnn.blobFromImage(frame, scale_factor, (416,416), swapRB=True,crop=False)
  return (blob,frame,(w,h))



def predict(net,blob):
  input,_,dim = blob
  w,h = dim
  net.setInput(input)
  ln = net.getLayerNames()
  # print(net.getUnconnectedOutLayers())
  ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
  layer_outputs = net.forward(ln)
  boxes, confidences, class_ids = [], [], []
  for output in layer_outputs:
    # loop over each of the object detections
    for detection in output:
        # extract the class id (label) and confidence (as a probability) of
        # the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # discard out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > CONFIDENCE:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
  # return a tuple
  return (boxes, confidences, class_ids)

def draw_boxes(image, prediction,labels):
  colors=np.random.randint(0,255, size=(len(labels),3), dtype='uint8')
  boxes, confidences, class_ids = prediction
  font_scale = 1
  thickness = 1
  idxs = cv.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
  # ensure at least one detection exists
  if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        # print("class: {}, label: {}".format(class_ids[i],labels[class_ids[i]]))
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv.FILLED)
        # add opacity (transparency to the box)
        image = cv.addWeighted(overlay, 0.6, image, 0.4, 0)
        # now put the text (label: confidence %)
        cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
  return image

def display(image):
  cv.namedWindow('image', cv.WINDOW_NORMAL)
  cv.resizeWindow('image', 900,600)
  cv.imshow('image', image)
  cv.waitKey(0)
  cv.destroyAllWindows()
  return None

def main():
  labels_file = 'data/coco.names'
  input_file = 'images/street.jpg'
  labels=open(os.path.join(PROJ_PATH,labels_file)).read().strip().split('\n')
  net = load_net(CONFIG_PATH,WEIGHTS_PATH)
  ## object detection on static image
  # 
  # blob,mat,dim =  blobify(input_file)
  # output = predict(net,(blob,mat,dim))
  # display(draw_boxes(mat, output, labels_file))
  dev = cv.VideoCapture(0)
  while True:
    blob,mat,dim = blobify_stream(dev)
    output = predict(net,(blob,mat,dim))
    detect = draw_boxes(mat,output,labels)
    cv.imshow('video',detect)
    if cv.waitKey(1) == ord('q'):
      cv.release()
      break
  return None

if __name__ == '__main__':
  main()