import sys
import cv2 as cv
import numpy as np

# ----------------------------- CONFIG -----------------------------------
VIDEO_PATH = "AI-Create-Project/clips/exampleClip.mp4"

# Kill feed ROI as fractions of the frame (top-right corner)
ROI_X_START = 0.70   # start 70% across
ROI_Y_END = 0.30     # top 30% of the frame

# HSV range for the red kill highlight (raise S/V minimums to reject scenery)
RED_LOWER_1 = np.array([0, 140, 90])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([165, 140, 90])
RED_UPPER_2 = np.array([180, 255, 255])

# Box shape filters, relative to the FULL frame so it works at any resolution.
# A feed row is a wide, short rectangle.
MIN_BOX_H_FRAC = 0.015   # min row height  (fraction of frame height)
MAX_BOX_H_FRAC = 0.060   # max row height
MIN_BOX_W_FRAC = 0.060   # min row width   (fraction of frame width)
MIN_ASPECT = 1.8         # width / height

# Fraction of the box's bounding rect that is red. A thin border is low,
# a red-tinted filled box is high. Tune after running with --debug.
MIN_FILL = 0.03
MAX_FILL = 0.95

# A box count must hold for this many consecutive frames before we trust it.
# Filters flicker, fade-in animation and one-frame false positives.
STABLE_FRAMES = 3

# Don't register a second trigger within this many seconds of the previous one
# ONLY if you want a hard cooldown. 0 = off (recommended for quick kills).
MIN_GAP_SECONDS = 0.5
# -------------------------------------------------------------------------


def get_kill_feed_roi(img):
    h, w = img.shape[:2]
    return img[0:int(h * ROI_Y_END), int(w * ROI_X_START):w]


def mask_red_pixels(roi):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.bitwise_or(
        cv.inRange(hsv, RED_LOWER_1, RED_UPPER_1),
        cv.inRange(hsv, RED_LOWER_2, RED_UPPER_2),
    )
    # Close small gaps in the border (keeps thin outlines intact, no OPEN step)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
    return cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


def find_feed_boxes(frame):
    """Return a list of (x, y, w, h) rects that look like red kill-feed rows."""
    full_h, full_w = frame.shape[:2]
    roi = get_kill_feed_roi(frame)
    mask = mask_red_pixels(roi)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if h == 0:
            continue
        if not (MIN_BOX_H_FRAC * full_h <= h <= MAX_BOX_H_FRAC * full_h):
            continue
        if w < MIN_BOX_W_FRAC * full_w:
            continue
        if w / float(h) < MIN_ASPECT:
            continue
        fill = cv.countNonZero(mask[y:y + h, x:x + w]) / float(w * h)
        if not (MIN_FILL <= fill <= MAX_FILL):
            continue
        boxes.append((x, y, w, h))

    # Sort top to bottom so results are consistent frame to frame
    boxes.sort(key=lambda b: b[1])
    return boxes


def detect_kill_events(video_path, debug=False):
    """
    Counts red boxes in the kill feed each frame. A kill is registered whenever
    the (debounced) box count goes UP. Because each kill adds its own box, a
    new kill is still detected while the previous kill's box is on screen.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return [], 30.0

    fps = cap.get(cv.CAP_PROP_FPS) or 60.0
    min_gap_frames = int(fps * MIN_GAP_SECONDS)

    kill_events = []
    confirmed_count = 0     # box count we currently trust
    candidate_count = 0     # count we're currently verifying
    candidate_streak = 0    # consecutive frames the candidate has held
    candidate_start = 0     # first frame of the candidate streak
    last_kill_frame = -10**9
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count = len(find_feed_boxes(frame))

        if count == candidate_count:
            candidate_streak += 1
        else:
            candidate_count = count
            candidate_streak = 1
            candidate_start = frame_idx

        if candidate_streak == STABLE_FRAMES and candidate_count != confirmed_count:
            if candidate_count > confirmed_count:
                new_kills = candidate_count - confirmed_count
                if candidate_start - last_kill_frame >= min_gap_frames:
                    for _ in range(new_kills):
                        kill_events.append(
                            {
                                "kill_num": len(kill_events) + 1,
                                "onset_frame": candidate_start,
                                "boxes_on_screen": candidate_count,
                            }
                        )
                    last_kill_frame = candidate_start
                    if debug:
                        print(
                            f"[debug] frame {candidate_start}: "
                            f"{confirmed_count} -> {candidate_count} boxes"
                        )
            confirmed_count = candidate_count

        frame_idx += 1

    cap.release()
    return kill_events, fps


def save_debug_frames(video_path, frames):
    """Dump ROI + mask + detected boxes for specific frames to tune the config."""
    cap = cv.VideoCapture(video_path)
    for f in frames:
        cap.set(cv.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        roi = get_kill_feed_roi(frame)
        mask = mask_red_pixels(roi)
        annotated = roi.copy()
        for (x, y, w, h) in find_feed_boxes(frame):
            cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imwrite(f"dbg_{f}_roi.png", annotated)
        cv.imwrite(f"dbg_{f}_mask.png", mask)
    cap.release()


def main():
    debug = "--debug" in sys.argv
    print("Processing video for kill events...")

    events, fps = detect_kill_events(VIDEO_PATH, debug=debug)
    print(f"\nDetected {len(events)} kill events.\n")

    for ev in events:
        onset = ev["onset_frame"]
        t = onset / fps
        start = max(0.0, t - 2.0)
        print(f"--- Kill #{ev['kill_num']} ---")
        print(f"  Onset frame: {onset}  ({t:.2f}s)")
        print(f"  Boxes on screen: {ev['boxes_on_screen']}")
        print(
            f'  ffmpeg: ffmpeg -ss {start:.2f} -i "{VIDEO_PATH}" -t 4.0 '
            f'-c:v libx264 -c:a aac "kill_{ev["kill_num"]}.mp4"\n'
        )

    # To inspect a suspicious frame, e.g. around your old false positive:
    # save_debug_frames(VIDEO_PATH, [536, 539, 542])


if __name__ == "__main__":
    main()