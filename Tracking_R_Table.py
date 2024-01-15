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
    

def build_r_table(roi_grey):
    """Build R_Table for ROI object 

    Args:
        roi_grey: Image of the ROI in Grey.
        
        return: r_table: (key: absolute of orientation | values: displacement vectors)
    """
    def calculate_gradient(image_grey):
        grad_x = cv2.Sobel(image_grey, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_grey, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        return magnitude, orientation
    
    magnitude, orientation = calculate_gradient(roi_grey)
    magnitude_threshold = np.mean(magnitude)
    print(f"Magnitude threshold ROI: {magnitude_threshold}")
    
    r_table = {}
    height, width = roi_grey.shape[:2]
    cy, cx = height // 2, width // 2  # Assuming center coordinates

    for i in range(height):
        for j in range(width):
            value = orientation[i, j]
            if magnitude[i, j] > magnitude_threshold:
                angle = int(np.abs(value))
                if angle not in r_table:
                    r_table[angle] = []
                
                # print(f"Angle: {angle} | Coordinates: {cy - i, cx - j}")
                r_table[angle].append((cy - i, cx - j))
    return r_table

def hough_transform(image_grey, r_table):
    """
    :param image_grey: input original image (grey)
    :param r_table: table for template
    :return:
        accumulator with searched votes
    """
    
    def calculate_gradient(image_grey):
        grad_x = cv2.Sobel(image_grey, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_grey, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        return magnitude, orientation
    
    height, width = image_grey.shape[:2]
    accumulator = np.zeros((height + 50, width + 50))

    magnitude, orientation = calculate_gradient(image_grey)
    magnitude_threshold = np.mean(magnitude) * 3
    print(f"Magnitude threshold frame: {magnitude_threshold}")
    assert(magnitude.shape == image_grey.shape[:2])
    
    for i in range(height):
        for j in range(width):
            if (magnitude[i, j] > magnitude_threshold):
                theta = int(np.abs(orientation[i, j]))
                if (theta in r_table):
                    vectors = r_table[theta]
                    for vector in vectors:
                        accumulator[vector[0] + i, vector[1] + j] += 1

    return accumulator


def find_detection(accumulator, image):
    def find_maximum(accumulator):
        rowId, colId = np.unravel_index(accumulator.argmax(), accumulator.shape)
        return rowId, colId
    
    r, c = find_maximum(accumulator)
    return r, c



def main():
    cwd = os.getcwd()
    print(cwd)

    # Open the video file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    video_path = os.path.join(dir_path, 'Test-Videos', 'VOT-Ball.mp4')

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    clone = frame.copy()
    cv2.namedWindow("First image")
    cv2.setMouseCallback("First image", define_ROI)

    # Variables for tracking
    track_window = None
    roi_hist = None
    term_crit = None

    # while True:
    #     cv2.imshow("First image", frame)
    #     key = cv2.waitKey(1) & 0xFF
    #     if roi_defined:
    #         cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
    #     else:
    #         frame = clone.copy()
            
    #     if key == ord("q"):
    #         break
    
    r = 196
    c = 110
    h = 55
    w = 52
    
    track_window = (r, c, h, w)
    print(r, c, h, w)
    
    roi = frame[c:c + w, r:r + h]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    cpt = 1
    while True:
        ret, frame = cap.read()
        if ret:
            # Compute gradient orientation and magnitude
            roi_grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            r_table = build_r_table(roi_grey)

            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            accumulator = hough_transform(frame_grey, r_table)

            # Display the detection by argmax
            r, c = find_detection(accumulator, frame_grey)
            print(f"Detection: {r, c}, Value: {accumulator[r, c]}")

            # Draw the circle on the frame
            cv2.circle(frame, (c, r), 30, (0, 255, 0), 2)  # Adjust the circle size and thickness

            cv2.imshow('Frame', frame)
            cv2.moveWindow('Frame', 0, 0)

            cv2.imshow('ROI', roi)
            cv2.moveWindow('ROI', 500, 0)

            # Normalize the accumulator to 0-255 and convert to 8-bit
            normalized_accumulator = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
            accumulator_8bit = cv2.convertScaleAbs(normalized_accumulator)

            # Apply a colormap to the 8-bit accumulator
            colored_accumulator = cv2.applyColorMap(accumulator_8bit, cv2.COLORMAP_JET)

            # Draw the circle on the colored accumulator
            cv2.circle(colored_accumulator, (c, r), 30, (0, 0, 255), 2)  # Adjust the circle size and thickness

            cv2.imshow('Accumulator', colored_accumulator)
            cv2.moveWindow('Accumulator', 800, 0)

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
    
if __name__ == '__main__':
    main()
