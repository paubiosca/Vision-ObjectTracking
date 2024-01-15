import numpy as np
import cv2
import os

from utils import calculate_histogram, calculate_gradient_orientation, masked_orientations

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

cwd = os.getcwd()
print(cwd)

# Open the video file
dir_path = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(dir_path, 'Test-Videos', 'VOT-Ball.mp4')
cap = cv2.VideoCapture(video_path)

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored 
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cpt = 1
while True:
	ret, frame = cap.read()
	if ret:
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Compute gradient orientation and magnitude
		gradient_magnitude, gradient_orientation = calculate_gradient_orientation(frame)

		# Create separate windows for each image and move them to different positions
		cv2.imshow('Gradient Magnitude', gradient_magnitude)
		cv2.moveWindow('Gradient Magnitude', 0, 400)  # Adjust the position as needed

		cv2.imshow('Gradient Orientation', gradient_orientation)
		cv2.moveWindow('Gradient Orientation', 400, 0)  # Adjust the position as needed

		masked_orientation = masked_orientations(frame)
		cv2.imshow('Masked Orientation', masked_orientation)
		cv2.moveWindow('Masked Orientation', 400, 400)

		hue_image = hsv[:,:,0]
		cv2.imshow('Hue Channel', hue_image)
		cv2.moveWindow('Hue Channel', 800, 400)  # Adjust the position as needed

		dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
		cv2.imshow('BackProjection', dst)
		cv2.moveWindow('BackProjection', 800, 0)  # Adjust the position as needed

		ret, track_window = cv2.meanShift(dst, track_window, term_crit)
		r, c, h, w = track_window
		frame_tracked = cv2.rectangle(frame, (r, c), (r+h, c+w), (255, 0, 0), 2)
		cv2.imshow('Original', frame_tracked)
		cv2.moveWindow('Original', 0, 0)  # Adjust the position as needed
        
        

		k = cv2.waitKey(60) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('Frame_%04d.png' % cpt, frame_tracked)
		cpt += 1
	else:
		break

cv2.destroyAllWindows()
cap.release()
