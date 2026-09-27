import cv2 as cv
import numpy as np

#this function loops through the ROI frames and checks for the presence of red pixels, which indicates a kill highlight and returns a list of tthe indexes of those frames.
def findFramesWithKills(frames_dict):
    kill_frame_indices = []

    previous_red_count = 0
    red_threshold = 50
    increase_threshold = 25

    for frame_idx, frame_img in frames_dict.items():

        # Create a separate HSV copy for detection
        hsv = cv.cvtColor(frame_img, cv.COLOR_BGR2HSV)

        # Red HSV ranges
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks ONLY for detecting red
        mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv.bitwise_or(mask1, mask2)

        # Count red pixels
        current_red_count = cv.countNonZero(red_mask)

        # Check if red increased significantly
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


def main():
    videoPath = "exampleClip.mp4"
    # Get frames from the video
    frames = getFrames(videoPath)

    # Find frames with kills
    framesWithKills = findFramesWithKills(frames)
    print(f"Frames with kills: {framesWithKills}")

    for i in framesWithKills:
        hsv = cv.cvtColor(frames[i], cv.COLOR_BGR2HSV)
        cv.imshow(f"Frame {i}", hsv)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()