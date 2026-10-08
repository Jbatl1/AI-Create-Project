import cv2 as cv
import numpy as np


def maskRedPixels(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Clean HSV range targeting CS2 red kill highlight boxes
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)

    return red_mask


def getKillFeedROI(img):
    height, width = img.shape[:2]
    startX = int(width - (width * 0.25))
    endY = int(height * 0.35)
    return img[0:endY, startX:width]


def detectKillEvents(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return [], 30.0

    fps = cap.get(cv.CAP_PROP_FPS) or 60.0

    # Cooldown guard: require at least 1.5 seconds between separate kill triggers
    cooldown_frames = int(fps * 1.5)
    frames_since_last_kill = cooldown_frames  # Start ready to detect

    kill_events = []
    kill_in_progress = False
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = getKillFeedROI(frame)
        red_mask = maskRedPixels(roi)
        current_red_count = cv.countNonZero(red_mask)

        contours, _ = cv.findContours(
            red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        has_valid_contour = False
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0
            if (
                cv.contourArea(cnt) > 80
                and w > 20
                and h > 8
                and aspect_ratio > 1.2
            ):
                has_valid_contour = True
                break

        # Increment cooldown counter on every frame
        frames_since_last_kill += 1

        # 1. NEW KILL TRIGGER
        # Must pass: Valid contour, red threshold, NOT currently locked, AND cooldown satisfied
        if not kill_in_progress and frames_since_last_kill >= cooldown_frames:
            if current_red_count > 100 and has_valid_contour:
                kill_events.append(
                    {
                        "kill_num": len(kill_events) + 1,
                        "onset_frame": frame_idx,
                        "red_pixels": current_red_count,
                    }
                )
                kill_in_progress = True
                frames_since_last_kill = 0  # Reset cooldown timer

        # 2. RESET TRIGGER
        # Only reset the lock when the red box disappears/fades significantly
        elif kill_in_progress:
            if current_red_count < 40:
                kill_in_progress = False

        frame_idx += 1

    cap.release()
    return kill_events, fps


def main():
    video_path = "AI-Create-Project/clips/exampleClip.mp4"
    print("Processing video for unique kill events...")

    events, fps = detectKillEvents(video_path)

    print(f"\nSuccessfully detected {len(events)} kill events!\n")

    for event in events:
        onset_frame = event["onset_frame"]
        kill_time_sec = onset_frame / fps
        clip_start_sec = max(0.0, kill_time_sec - 2.0)

        print(f"--- Kill #{event['kill_num']} ---")
        print(f"  Exact Kill Onset Frame: {onset_frame}")
        print(f"  Timestamp of Kill: {kill_time_sec:.2f}s")
        print(f"  2s Prior Clip Start: {clip_start_sec:.2f}s")
        print(
            f"  ffmpeg Command: ffmpeg -ss {clip_start_sec:.2f} -i \"{video_path}\" -t 2.0 -c copy \"kill_{event['kill_num']}.mp4\"\n"
        )


if __name__ == "__main__":
    main()