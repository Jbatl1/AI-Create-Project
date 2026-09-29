#& "D:\VSCode projects\AI Create Project\.venv\Scripts\python.exe" Test.py
import cv2 as cv
import numpy as np
import pytesseract as pytes
from matplotlib import pyplot as plt


def maskRedPixels(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)
    return red_mask



#this function loops through the ROI frames and checks for the presence of red pixels, which indicates a kill highlight and returns a list of tthe indexes of those frames.
def findFramesWithKills(frames_dict):
    kill_frame_indices = []

    previous_red_count = 0
    red_threshold = 50
    increase_threshold = 25

    for frame_idx, frame_img in frames_dict.items():

        red_mask = maskRedPixels(frame_img)

        current_red_count = cv.countNonZero(red_mask)

        red_increased = (
            current_red_count > red_threshold
            and current_red_count > previous_red_count + increase_threshold
        )

        if red_increased:
            kill_frame_indices.append(frame_idx)

        previous_red_count = current_red_count

    return kill_frame_indices




def getKillFeedROI(img):
    height = img.shape[0]
    width = img.shape[1]

    startY = 0
    startX = width - (width * 0.2)

    roi = img[int(startY):int(height * 0.3), int(startX):int(width)]
    return roi





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

        frames[current_frame_index] = getKillFeedROI(frame)
        current_frame_index += 1

    cap.release()

    print(f"Total frames captured: {len(frames)}")
    return frames




def killFeedCornerDetect(frames):
    result = frames.copy()

    for i, frame in frames.items():
    # 1. Convert to HSV and mask red pixels
        red_mask = maskRedPixels(frame)
    
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
                    cv.circle(result[i], pt, 2, (0, 255, 0), -1)

    return result




def main():
    videoPath = "exampleClip.mp4"
    frames = getFrames(videoPath)
    frameIndexWithKills = findFramesWithKills(frames)
    framesWithKillsMarked =  killFeedCornerDetect(frameIndexWithKills)
    cv.imshow("markedFrame", framesWithKillsMarked[0])
    cv.imshow("markedFrame", framesWithKillsMarked[20])
    cv.imshow("markedFrame", framesWithKillsMarked[40])
    cv.waitKey(0)
    cv.destroyAllWindows()


    






if __name__ == "__main__":
    main()