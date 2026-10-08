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


def getKillFeedROI(img):
    height, width = img.shape[:2]
    startX = int(width - (width * 0.20))
    endY = int(height * 0.30)

    # Return top-right 20% width and 30% height
    return img[0:endY, startX:width]


def getFrames(videoPath):
    cap = cv.VideoCapture(videoPath)

    if not cap.isOpened():
        print(f"Error opening video file: {videoPath}")
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


def killFeedCornerDetect(frames_dict, kill_indices):
    marked_frames = {}

    for idx in kill_indices:
        frame = frames_dict[idx]
        result = frame.copy()

        red_mask = maskRedPixels(frame)
        contours, _ = cv.findContours(
            red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            # Filter out minor pixel noise
            if cv.contourArea(cnt) > 50:
                x, y, w, h = cv.boundingRect(cnt)
                corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

                for pt in corners:
                    cv.circle(result, pt, 3, (0, 255, 0), -1)

        marked_frames[idx] = result

    return marked_frames


def main():
    videoPath = "AI-Create-Project/clips/exampleClip.mp4"
    
    # 1. Capture ROI frames from video
    all_frames = getFrames(videoPath)
    if not all_frames:
        return

    # 2. Identify indices where kills occur
    kill_indices = findFramesWithKills(all_frames)
    print(f"Frames where kills were detected: {kill_indices}")

    if not kill_indices:
        print("No kill highlights detected.")
        return

    # 3. Detect corners on those identified kill frames
    marked_kill_frames = killFeedCornerDetect(all_frames, kill_indices)

    # 4. Preview detected kill frames sequentially (press any key for next frame)
    print("Displaying detected kill frames. Press any key in the window to step through...")
    for idx, frame_img in marked_kill_frames.items():
        cv.imshow(f"Kill Detected at Frame {idx}", frame_img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()