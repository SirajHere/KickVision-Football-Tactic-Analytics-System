import sys
import cv2  
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator# or any other processing library

def process_video(video_file_path):
    # Example video processing code (you can replace this with your logic)
    print(video_file_path)
    sys.stdout.flush() 
    video_frames = read_video(video_file_path)

    # Initialize Tracker
    tracker = Tracker("C:/xampp/htdocs/sameer/models/best.pt")

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False,stub_path=r"c:/xampp/htdocs/sameer/stubs/track_stubs.pkl")
    
    # Filter out low-confidence detections
    for frame_num in range(len(tracks["players"])):
        tracks["players"][frame_num] = {
            pid: track for pid, track in tracks["players"][frame_num].items() if track.get("confidence", 1.0) > 0.5
        }

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=False,stub_path=r"c:/xampp/htdocs/sameer/stubs/camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    if "ball" in tracks:
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    if tracks["players"]:
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    
        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = (
                    team_assigner.team_colors.get(team, "Unknown")
                )

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        if "ball" in tracks and frame_num < len(tracks["ball"]):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(
                    tracks["players"][frame_num][assigned_player]["team"]
                )
            elif team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None)
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    
    save_video(output_video_frames, "C:/xampp/htdocs/sameer/output_videos/output_video.avi")
    output_path="C:/xampp/htdocs/sameer/output_videos/output_video.avi"
    
    return output_path

if __name__ == '__main__':
    video_file_path = sys.argv[1]
    full_path = os.path.abspath(video_file_path)
    #print(f"Received video path and starting : {full_path}") 
    #sys.stdout.flush() 
    processed_video_path = process_video(full_path)
    print(processed_video_path) 
    sys.stdout.flush()  # Output the processed file path
