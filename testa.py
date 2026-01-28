import cv2

def get_video_properties(file_path):
    video_capture = cv2.VideoCapture(file_path)
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video: {file_path}")
    
    # Extract video properties
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    
    video_capture.release()
    
    # Return video properties as a dictionary
    return {
        "frame_count": frame_count,
        "resolution": (frame_width, frame_height),
        "frame_rate": frame_rate
    }

def display_video_properties(video_paths):
    for video_path in video_paths:
        try:
            video_properties = get_video_properties(video_path)
            print(f"Video Info for {video_path}:")
            print(f"  Frames: {video_properties['frame_count']}")
            print(f"  Resolution: {video_properties['resolution'][0]}x{video_properties['resolution'][1]}")
            print(f"  Frame rate: {video_properties['frame_rate']}\n")
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # List of video paths
    video_files = [
        r"D:\major 2\input_videos\match.mp4",
        r"D:\major 2\input_videos\match2.mp4"
    ]
    
    # Display properties for both videos
    display_video_properties(video_files)
