# import the necessary packages
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2

def detect_people(frame, net, ln, personIdx=0):
	# To get the dimensions of the frame
	(H, W) = frame.shape[:2]
	results = []

	# construct a blob from the input frame 
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	#Forward propagation to YOLO to get bounding boxes and probabilities
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# To initialize lists of detected bounding boxes, centroids and confidences
	boxes = []
	centroids = []
	confidences = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# To extract the class ID and confidence of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personIdx and confidence > MIN_CONF:
				# To scale the bounding box coordinates back relative to image size
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# To derive the top and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# To update the lists
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	#To apply non-maxima suppression 
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	#To ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes
		for i in idxs.flatten():
			# To extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# To update our results list 
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# To return the list of results
	return results
