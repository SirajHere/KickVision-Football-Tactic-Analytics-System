import cv2

video_path = r"D:\major 2\input_videos\match.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("⚠️ OpenCV cannot open the video file!")
else:
    print("✅ OpenCV successfully opened the video.")
    cap.release()
