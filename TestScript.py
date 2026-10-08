import cv2 as cv
import numpy as np


def debugPipeline(video_path):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open video file at path: '{video_path}'")
        print("Check if the relative path or file name is incorrect.")
        return

    # 1. Print Video Properties
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(
        f"✅ Video Opened Successfully!\n"
        f"   Resolution: {width}x{height}\n"
        f"   FPS: {fps}\n"
        f"   Total Frames: {total_frames}\n"
    )

    frame_idx = 0
    detections_found = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop top-right killfeed
        startX = int(width - (width * 0.22))
        endY = int(height * 0.35)
        roi = frame[0:endY, startX:width]

        # Red Mask
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(
            hsv, np.array([0, 120, 70]), np.array([10, 255, 255])
        )
        mask2 = cv.inRange(
            hsv, np.array([165, 120, 70]), np.array([180, 255, 255])
        )
        red_mask = cv.bitwise_or(mask1, mask2)

        # Count non-zero red pixels
        red_pixel_count = cv.countNonZero(red_mask)

        # Find contours
        contours, _ = cv.findContours(
            red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        contour_match = False
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0

            # Test conditions
            if (
                cv.contourArea(cnt) > 80
                and w > 20
                and h > 8
                and aspect_ratio > 1.2
            ):
                contour_match = True
                cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if red_pixel_count > 120 or contour_match:
            detections_found += 1
            print(
                f"Frame {frame_idx}: Red Pixels={red_pixel_count}, Contour Match={contour_match}"
            )

            # Show visual popup for the first 3 detected frames
            if detections_found <= 3:
                cv.imshow("Detected ROI", roi)
                cv.imshow("Red Mask", red_mask)
                print(
                    "   (Press ANY KEY on the image window to resume playback...)"
                )
                cv.waitKey(0)

        frame_idx += 1

    cap.release()
    cv.destroyAllWindows()
    print(f"\nDebug Complete. Total Kill Frames Detected: {detections_found}")


if __name__ == "__main__":
    debugPipeline("clips/exampleClip.mp4")