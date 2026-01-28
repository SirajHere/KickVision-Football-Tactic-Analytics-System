import os
import subprocess
from flask import Flask, request, send_file

app = Flask(__name__)

UPLOAD_DIRECTORY = os.path.abspath('uploadss')
DEMO_VIDEO_DIRECTORY = os.path.abspath('demo_videos')  # Folder for demo videos

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(DEMO_VIDEO_DIRECTORY, exist_ok=True)

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    # Save uploaded file
    video_file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    file.save(video_file_path)
    video_file_path = os.path.normpath(video_file_path)

    return run_prediction(video_file_path, is_demo=False)


@app.route('/process_demo_video', methods=['POST'])
def process_demo_video():
    demo_video_name = request.form.get('demo_video')
    if not demo_video_name:
        return "No demo video provided", 400

    demo_video_path = os.path.join(DEMO_VIDEO_DIRECTORY, demo_video_name)
    
    if not os.path.exists(demo_video_path):
        return f"Demo video '{demo_video_name}' not found", 404

    return run_prediction(demo_video_path, is_demo=True)


def run_prediction(video_path, is_demo=False):
    try:
        if is_demo:
            script_path = os.path.abspath('predicta.py')  # Use predicta.py for demo videos
        else:
            script_path = os.path.abspath('predict.py')  # Use predict.py for uploaded files

        print(f"Running command: python {script_path} {video_path}")

        # Run the respective script (predict.py or predicta.py)
        result = subprocess.Popen(
            ['python', script_path, video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout_data, stderr_data = result.communicate()

        print("stdout:", stdout_data.strip())
        print("stderr:", stderr_data.strip())

        # Extract processed video path
        processed_video_path = stdout_data.strip().split("\n")[-1]
        print(f"Extracted processed video path: {processed_video_path}")

        if os.path.exists(processed_video_path):
            return send_file(processed_video_path, mimetype='video/avi', as_attachment=False)
        else:
            return f"Processed video not found at {processed_video_path}", 500

    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
