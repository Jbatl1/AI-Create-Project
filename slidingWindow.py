import cv2 as cv
import numpy as np

def maskRedPixels(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Higher saturation threshold (150) prevents false positives from grey UI noise
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    return cv.bitwise_or(mask1, mask2)



def getKillFeedROIs(frame):
    ROIs = []
    height, width = frame.shape[:2]
    
    startX = int(width - (width * 0.175))
    endX = width
    startY = int(height * .07)
    endY = int(height * 0.099)

    h = endY - startY

    for i in range(10):
        startY += h + 2
        endY += h + 2
        ROIs.append(frame[startY:endY, startX:endX])
    return ROIs

    




def main():
    framePath = "AI-Create-Project\clips\csExampleKillFeed.jpg"
    frameTest = cv.imread(framePath)
    if (frameTest is None):
        print("frame not loaded")
    else:
        rois = getKillFeedROIs(frameTest)
        
        for frame in rois:
            cv.imshow("Test", frame)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()