import cv2
import subprocess

def get_video_properties(input_video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)
    
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    
    # Get video properties
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get codec information using ffprobe
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', input_video_path
    ]
    codec = subprocess.check_output(command).decode('utf-8').strip()
    
    video_capture.release()
    
    return frame_rate, width, height, codec, frame_count

def resize_video(input_video_path, output_video_path, frame_rate, resolution, codec, original_frame_count):
    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)
    
    if not video_capture.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")
    
    # Create VideoWriter object to save the resized video
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, resolution)
    
    frame_num = 0
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Resize the frame to match the target resolution
        resized_frame = cv2.resize(frame, resolution)
        
        # Write the resized frame to the output video
        video_writer.write(resized_frame)
        
        frame_num += 1
        
        # Break if we've reached the frame count of the original video
        if frame_num >= original_frame_count:
            break
    
    # Release the video objects
    video_capture.release()
    video_writer.release()
    print(f"Resized video saved to {output_video_path}")

# Paths to the videos
input_video_path = "d:/major 2/input_videos/match.mp4"
output_video_path = "d:/major 2/output_videos/resized_match.mp4"

# Read the properties of the original video (match.mp4)
frame_rate, original_width, original_height, codec, original_frame_count = get_video_properties( "D:\\major 2\\input_videos\\match.mp4")

# Check and ensure codec consistency (fallback to 'mp4v' if codec is not compatible)
if codec not in ['h264', 'avc1', 'mp4v']:
    print("Warning: Codec not compatible, falling back to 'mp4v'.")
    codec = 'mp4v'

# Resize match2 video to match all properties of match.mp4
resize_video(input_video_path, output_video_path, frame_rate, (original_width, original_height), codec, original_frame_count)
