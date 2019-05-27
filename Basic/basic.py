import cv2
import imutils

#Loading Image and accessing shape of image and pixels.
image = cv2.imread("pic.jpg")
(h,w,d) = image.shape
print("Height is {}, Width is {} and Depth is {}".format(h,w,d))
(b,g,r)=image[250,250]
print("R={}, G={}, B={} for pixel image[250,250]".format(r, g, b))
cv2.imshow("Image",image);
cv2.waitKey(0)

#Extracting Region of Interest
roi = image[180:450,60:300]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

#Resizing Image without keeping Aspect Ratio - Fixed Resizing
resized = cv2.resize(image, (300,400))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

#Resizing Image with keeping Aspect Ratio - Aspect Ratio Resizing
resized = imutils.resize(image, width=400)
cv2.imshow("Aspect Ratio Resizing using Imutils Resize", resized)
cv2.waitKey(0)

# Rotating image 30 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w//2,h//2)
mat = cv2.getRotationMatrix2D(center, -30, 1.0)
rotated = cv2.warpAffine(image, mat, (w, h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

# Normal rotation about centre using imutils with less code
rotated = imutils.rotate(image, -30)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(image, 30)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# Applying a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image, (11,11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# Drawing a 3px thick red rectangle surrounding the face
rect = image.copy()
cv2.rectangle(rect, (90, 190),(280,400), (0, 0, 255), 3)
cv2.imshow("Rectangle", rect)
cv2.waitKey(0)

# Drawing a blue 30px (filled in) circle on the centre of image
cir = image.copy()
cv2.circle(cir,(w//2,h//2),30,(255,0,0),-1)
cv2.imshow("Circle",cir)
cv2.waitKey(0)

# Drawing a 5px thick white line from x=60,y=20 to x=400,y=200
ln = image.copy()
cv2.line(ln, (25, 78), (400, 450), (0, 0, 0), 5)
cv2.imshow("Line", ln)
cv2.waitKey(0)

# Drawing text on the image
txt = image.copy()
cv2.putText(txt, "Computer Vision", (0, 50), 
	cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 125), 2)
cv2.putText(txt, "Basics", (100, 100), 
	cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 125), 2)
cv2.imshow("Text", txt)
cv2.waitKey(0)