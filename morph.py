import cv2
import numpy as np
import matplotlib.pyplot as plt 
from Otsu import *
from perspectivetransform import *
from imutils import contours

#ANSWER_KEY_5 = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0}
#ANSWER_KEY_20 = {0: 0, 1: 2, 2: 0, 3: 3, 4: 4, 5: 0, 6: 3, 7: 3, 8: 4, 9: 0, 10: 0, 11: 4, 12: 2, 13: 3, 14: 4, 15: 0, 16: 1, 17: 2, 18: 3, 19: 4}
#ANSWER_KEY_10 = {0: 0, 1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 0, 7: 3, 8: 3, 9: 4}

def plt_imshow(title, image):
  # convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()


image = cv2.imread('C:/Users/User/Downloads/20questions.jpg')
image = cv2.resize(image, (525,700))

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)

#do otsu threshold on gray image
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# apply morphology
kernel = np.ones((7,7), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
#plt_imshow('Morphological operation', morph)


# get largest contour
conts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = conts[0] if len(conts) == 2 else conts[1]
area_thresh = 0
for c in conts:
    area = cv2.contourArea(c)
    if area > area_thresh:
        area_thresh = area
        big_contour = c

# draw white filled largest contour on black just as a check to see it got the correct region
page = np.zeros_like(image)
y = cv2.drawContours(page, [big_contour], 0, (255,255,255), -1)
# get perimeter and approximate a polygon
peri = cv2.arcLength(big_contour, True)
corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)
# draw polygon on input image from detected corners
polygon = image.copy()
cv2.polylines(polygon, [corners], True, (0,0,255), 1, cv2.LINE_AA)
#print(len(corners))
#print(corners)


paper = top_down_view(image, corners.reshape(4, 2))
warped = top_down_view(gray, corners.reshape(4, 2))


ot1 = Otsu(warped)


cnts = cv2.findContours(ot1.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

allContourImage = paper.copy()
cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
print("Total contours found after edge detection {}".format(len(cnts)))
#show_images([allContourImage], ["All contours from edge detected image"])
#plt_imshow("Perspective Projection", allContourImage)


for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 9 and h >= 9 and ar >= 0.7 and ar <= 1.4:
		questionCnts.append(c)
       
#

questionCnts = contours.sort_contours(questionCnts,
	method="top-to-bottom")[0]
correct = 0


questionsContourImage = paper.copy()
cv2.drawContours(questionsContourImage, questionCnts, -1, (0, 0, 255), 3)
#plt_imshow("All questions contours after filtering questions", questionsContourImage)


# each question has 5 possible answers, to loop over the
# question in batches of 5

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	# sort the contours for the current question from
	# left to right, then initialize the index of the
	# bubbled answer
	cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
	bubbled = None

	# loop over the sorted contours
	for (j, c) in enumerate(cnts):
		# construct a mask that reveals only the current
		# "bubble" for the question
		mask = np.zeros(ot1.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		# apply the mask to the thresholded image, then
		# count the number of non-zero pixels in the
		# bubble area
		mask = cv2.bitwise_and(ot1, ot1, mask=mask)
		total = cv2.countNonZero(mask)

		# if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	# initialize the contour color and the index of the
	# *correct* answer
	color = (0, 0, 255)
	k = ANSWER_KEY[q]

	# check to see if the bubbled answer is correct
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1

	# draw the outline of the correct answer on the test
	cv2.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 20.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
plt_imshow("Original", image)
plt_imshow("Exam", paper)



