# terminal command to run file:
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from matplotlib import pyplot as plt

import numpy as np
import argparse
import imutils
import time
import cv2

def sum_white(img):

    # get total number of pixels in image
    dimensions = img.shape
    total_pix = dimensions[0]*dimensions[1]

    n_white_pix = np.sum(img == 255)
    percent = (n_white_pix/total_pix)*100

    # print('Total: ', total_pix)
    # print('Number of white pixels:', n_white_pix)
    print('Yellow percentage: ', str(percent)+'%')
    return percent

def hist(image):
    #img = cv2.imread(image)
    color = ('b','g','r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def postScan(img):

    YELLOW_MIN = np.array([20, 100, 100],np.uint8)# (best)yellow min 20, 100, 100
    YELLOW_MAX = np.array([30, 255, 255],np.uint8)# (best)yellow max 30, 255, 255

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imwrite('output1.jpg', hsv_img)
    cv2.imshow('HSV_human',hsv_img)
    frame_threshed = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
    percentage = sum_white(frame_threshed)
    per_round = round(percentage,2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_threshed,str(per_round)+'%',(10,100),font,1,(255,255,0),2,cv2.LINE_AA)
    #cv2.imwrite('output2.jpg', frame_threshed)
    #print('Yellow percentage: ', str(percentage)+'%')
    cv2.imshow('Yellow',frame_threshed)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			# EDIT START
			if confidence >= 0.999:
				print(confidence, startX, endX, startY, endY)
				#cv2.imshow('human',frame)
				human = frame[startY:endY, startX:endX]
				cv2.imshow('human',human)
				postScan(human)
			#EDIT END
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
