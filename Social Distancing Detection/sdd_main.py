# python sdd_main.py --input vid1.mp4

# import the necessary packages
from config_detect import social_distancing_config as config
from config_detect.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
args = vars(ap.parse_args())

# To load the COCO class labels
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# To derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
print("Path:",weightsPath)
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# To load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# To determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# To initialize the video stream
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)


# To loop over the frames from the video stream
while True:
	# To read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then end of the stream is reached
	if not grabbed:
		break

	# To resize the frame and then detect people in it
	#frame = imutils.resize(frame, width=700)
	frame = cv2.resize(frame,(600,500))
	results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

	# To initialize the set of indexes that violate the minimum social distance
	violate = set()

	# ensure there are at least two people detections
	if len(results) >= 2:
		# To extract all centroids from the results
		centroids = np.array([r[2] for r in results])
		# To compute Euclidean distances between all pairs of the centroids
		D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check the distance between centroid pairs
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates
		(startX, startY, endX, endY) = bbox
		# initialize the color of the annotation
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then update the color
		if i in violate:
			color = (0, 0, 255)

		# draw a bounding box around the person
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# draw the centroid coordinates of the person
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# display the total number of social distancing violations on the
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)


	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
