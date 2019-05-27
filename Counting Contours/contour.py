import argparse
import imutils
import cv2

# Constucting Argument Parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# Loading and displaying input image supplied via command line
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# applying edge detection and finding the outlines of objects in
# images using canny
edged = cv2.Canny(gray, 30, 100)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# Thresholding the image by setting all pixel values less than 238
# to 255 (white; foreground) and all pixel values >= 238 to 0
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 238, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# Finding contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

# Looping over all contours and drawing each contour on output
# image with 3px thick maroon outline
for c in cnts:
	cv2.drawContours(output, [c], -1, (0, 0, 159), 3)
	cv2.imshow("Contours", output)
	cv2.waitKey(0)

# Drawing the total number of contours found in maroon
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

# Applying erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# Applying dilations to increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# Applying a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)