import cvzone
from ultralytics import YOLO
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import cv2
import time
import os

input_videos = [ 'input/spooltech_217.mp4']
output_folder = 'output'
frame_skip_interval = 1  # Process every 5th frame

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize YOLO face detection model
facemodel = YOLO('yolov8m-face.pt')

# Function to process a single video
def process_video(input_path, output_path):
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Initialize video writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = FFMPEG_VideoWriter(filename=output_path, size=(frame_width, frame_height), fps=fps)
    #out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print(f"Processing video {input_path}. FPS: {fps}, Frame size: {frame_width}x{frame_height}")

    previous_frame = None
    frame_count = 0
    cnt = 0

    while cap.isOpened():
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip_interval == 0:
            # Perform face detection and blur every nth frame
            cnt=cnt+1
            print(f"Frame processing num: {frame_count}")
            face_result = facemodel.predict(frame, conf=0.30)
            for info in face_result:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w = y2 - y1, x2 - x1

                    # Draw rectangle around detected face
                    #cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
                    # Blur detected face
                    face = frame[y1:y1 + h, x1:x1 + w]
                    face = cv2.blur(face, (40, 40))
                    frame[y1:y1 + h, x1:x1 + w] = face

            # Store the processed frame to be duplicated
            previous_frame = frame.copy()

        # If not processing, reuse the previous frame
        if previous_frame is not None:
            output = cv2.cvtColor(previous_frame, cv2.COLOR_BGRA2RGB)
            out.write_frame(output)
        else:
            output = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            out.write_frame(output)       # Initial frames before the first processed one
        
        frame_count += 1

    # Release video capture and writer
    cap.release()
    out.close()
    print(f"Total frame count: {frame_count}")
    print(f"Finished processing video {input_path}, saved to {output_path}")

if __name__ == "__main__":
    # Record start time
    start_time = time.time()

    # Process each video in the list
    for path in input_videos:
        video_name = os.path.basename(path)
        output_path = os.path.join(output_folder, f'{video_name}')
        process_video(path, output_path)

    # Record end time
    end_time = time.time()

    # Calculate total processing time
    total_time = end_time - start_time

    print(f"Total processing time for all videos: {total_time:.2f} seconds")