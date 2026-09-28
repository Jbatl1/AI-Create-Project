#& "D:\VSCode projects\AI Create Project\.venv\Scripts\python.exe" Test.py
import cv2 as cv
import numpy as np
import pytesseract as pytes
from matplotlib import pyplot as plt

pytes.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#this function loops through the ROI frames and checks for the presence of red pixels, which indicates a kill highlight and returns a list of tthe indexes of those frames.
def findFramesWithKills(frames_dict):
    kill_frame_indices = []

    previous_red_count = 0
    red_threshold = 50
    increase_threshold = 25

    for frame_idx, frame_img in frames_dict.items():

        hsv = cv.cvtColor(frame_img, cv.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv.bitwise_or(mask1, mask2)

        current_red_count = cv.countNonZero(red_mask)

        red_increased = (
            current_red_count > red_threshold
            and current_red_count > previous_red_count + increase_threshold
        )

        if red_increased:
            kill_frame_indices.append(frame_idx)

        previous_red_count = current_red_count

    return kill_frame_indices

# this function takes an image and returns the top right of the image where the killfeed is located.
def getROI(img):
    height = img.shape[0]
    width = img.shape[1]

    startY = 0
    startX = width - (width * 0.2)

    roi = img[int(startY):int(height * 0.3), int(startX):int(width)]
    return roi

# this function takes a video path and returns a dictionary of frames captured every 4 seconds.
def getFrames(videoPath):
    cap = cv.VideoCapture(videoPath)

    if not cap.isOpened():
        print("Error opening video file")
        return {}

    frames = {}
    current_frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames[current_frame_index] = getROI(frame)
        current_frame_index += 1

    cap.release()

    print(f"Total frames captured: {len(frames)}")
    return frames


import cv2 as cv
import numpy as np

def cornerDetectTest(img):
    result = img.copy()

    # 1. Convert to HSV and mask red pixels
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)

    # 2. Find contours of the red shapes directly
    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Ignore small red pixel noise
        if cv.contourArea(cnt) > 50: 
            # Get the strict outer rectangle coordinates
            x, y, w, h = cv.boundingRect(cnt)
            
            # The 4 exact corner coordinates
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            
            # Draw green circles on the 4 corners
            for pt in corners:
                cv.circle(result, pt, 2, (0, 255, 0), -1)

    return result




def testTextExtraction():
    #get image from path
    img_path = "csExampleKillFeed.jpg"
    image = cv.imread(img_path)



    #craete ROI from grey image
    roi = getROI(image)
    cornerImage = cornerDetectTest(roi)
    # cv.imshow("roi", roi)
    # cv.imshow("corners", cornerImage)
    # cv.imwrite("cornerResult.png", cornerImage)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    #create ROIs for each box

    #print extracted text from rgb
    extracted_text = pytes.image_to_string(roi)
    print(" Extracted Text:\n")
    print(extracted_text)


def main():
    # videoPath = "exampleClip.mp4"

    # frames = getFrames(videoPath)

    # framesWithKills = findFramesWithKills(frames)

    # print(f"Frames with kills: {framesWithKills}")

    # for i in framesWithKills:
    #     cv.imshow(f"Frame {i}", frames[i])
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    testTextExtraction();


if __name__ == "__main__":
    main()