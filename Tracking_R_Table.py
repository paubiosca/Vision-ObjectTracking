import numpy as np
import cv2
import os

roi_defined = False

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r)
        w = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True

def calculate_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    return magnitude, orientation

def build_r_table(orientation, magnitude_threshold):
    r_table = {}
    height, width = orientation.shape[:2]
    cy, cx = height // 2, width // 2  # Assuming center coordinates

    for i in range(height):
        for j in range(width):
            value = orientation[i, j]
            if magnitude[i, j] > magnitude_threshold:
                if value not in r_table:
                    r_table[value] = []
                r_table[value].append((cy - i, cx - j))
    return r_table

def hough_transform(image, r_table):
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image_grey.shape[:2]
    accumulator = np.zeros((height, width), dtype=np.uint64)

    for i in range(height):
        for j in range(width):
            value = image_grey[i, j]
            if (value > 0):  # Check if any value in the array is greater than 0
                if value in r_table:
                    for r, theta in r_table[value]:
                        a = int(j + r * np.cos(np.radians(theta)))
                        b = int(i + r * np.sin(np.radians(theta)))
                        if 0 <= a < width and 0 <= b < height:
                            accumulator[b, a] += 1

    return accumulator


def display_detection(image, argmax_result):
    print(argmax_result)
    print(np.unravel_index(argmax_result, image.shape))
    r, c = np.unravel_index(argmax_result, image.shape)
    cv2.circle(image, (c, r), 10, (0, 0, 255), 2)  # Red circle at the detected location

cwd = os.getcwd()
print(cwd)

# Open the video file
dir_path = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(dir_path, 'Test-Videos', 'VOT-Ball.mp4')

# Initialize variables and video capture
roi_defined = False
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# Variables for tracking
track_window = None
roi_hist = None
term_crit = None

while True:
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    if roi_defined:
        cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
        track_window = (r, c, h, w)
        roi = frame[c:c + w, r:r + h]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        break
    else:
        frame = clone.copy()
    if key == ord("q"):
        break

cpt = 1
while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute gradient orientation and magnitude
        magnitude, orientation = calculate_gradient(frame)

        r_table = build_r_table(orientation, magnitude_threshold=100)
        accumulator = hough_transform(frame, r_table)

        # Find the location of the maximum value in the accumulator
        max_loc = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        
        # Display the detection by argmax
        display_detection(frame, max_loc)

        cv2.imshow('Frame with Detection', frame)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png' % cpt, frame)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
