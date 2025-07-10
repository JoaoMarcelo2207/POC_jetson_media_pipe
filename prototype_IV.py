import cv2
import sys, os, argparse
from collections import deque
import time
import psutil

# Import additional functions
sys.path.append(os.path.join("lib"))
import fifo_manager as fifo, graphic_functions as gf
sys.path.append(os.path.join("lib", "threads"))
from landmark_thread import LandmarkProcessor



# Graphic Config
COLOR_STD = (0, 0, 255)    # Red - Standard
COLOR_ABOUT = (255, 255, 0)  # Light Blue - class "about"
COLOR_INTERVIEW = (0, 255, 0) # Green - class "interview"
COLOR_HAVE = (255,0,0) # Blue - class "have"
color_buffer = deque(maxlen=gf.PLOT_SIZE)
for _ in range(gf.PLOT_SIZE):
    color_buffer.append(COLOR_STD)
WINDOW_SIZE = 10  # Window points that will be coloried after prediction
SEAL_COLOR = None
LAST_COLOR = COLOR_STD
SEAL_COUNTER = 0
FIFO_SIZE = 150 


def video_capture_with_canvas(video_path, display):
    """
    Captures frames from the camera or video and displays them on a canvas with sections for the camera, scatter plot, and line chart.
    Adds an additional line chart below the existing one, showing raw (non-normalized) measures.
    Overlays facial landmarks correctly aligned to the face.
    """
    
    # Open video
    if video_path:
        print(f"Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        video_originalFPS = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video Original FPS: {video_originalFPS:.2f}")
        frame_duration = 1.0 / video_originalFPS
    else:
        print("Capturing from webcam...")
        cap = cv2.VideoCapture(0)
        # Resolution forced
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return 
    
    # Canvas dimensions and sections
    canvas, positions, camera_space, scatter_space, line_chart_space = gf.create_canvas(
        camera_space=(600, 400), scatter_space=(600, 400), line_chart_space=(1200, 200)
    )

    # Adjust canvas for an additional line chart
    canvas_height = canvas.shape[0] + line_chart_space[1]
    canvas = cv2.resize(canvas, (canvas.shape[1], canvas_height))
    positions["line_chart_raw"] = (positions["line_chart"][0], positions["line_chart"][1] + line_chart_space[1])

    time_series_buffer_normalized = deque(maxlen=gf.PLOT_SIZE)
    time_series_buffer_raw = deque(maxlen=gf.PLOT_SIZE)

    # Color buffer
    color_buffer = deque(maxlen=gf.PLOT_SIZE)

    #RAM Usage
    pid = os.getpid()
    process = psutil.Process(pid)
    
    #Start thread for landmark detection
    landmark_thread = LandmarkProcessor()
    landmark_thread.start()
    
    while True:
        loop_start = time.perf_counter()
        ret, img_all = cap.read()

        if (not ret or img_all is None) and video_path is not None:
            print("End of video or error reading frame.")
            break
        elif not ret or img_all is None:
            print("Error capturing frame from the camera.")
            break
        
        # Sends the frame for the thread
        landmark_thread.set_frame(img_all)
        # gets the results from the thread
        landmarks = landmark_thread.get_landmarks()

        #If the thread didn't detect anything (first interaction)
        if landmarks is None:
            continue
        
        # Normalized and raw points
        normalized_landmarks = gf.normalization(landmarks, (gf.normal_height_img, gf.normal_width_img))
        raw_landmarks = landmarks  # Direct raw points without normalization

        # Overlay landmarks on the original frame before resizing
        for (x, y) in landmarks:
            cv2.circle(img_all, (x, y), 2, (0, 255, 0), -1)

        # Resize the camera frame to the designated section
        camera_frame = cv2.resize(img_all, (camera_space[0], camera_space[1]))

        canvas[positions["camera"][1]:positions["camera"][1] + camera_space[1],
            positions["camera"][0]:positions["camera"][0] + camera_space[0]] = camera_frame

        # Plot points in the scatter section
        canvas = gf.plot_scatter(canvas, scatter_space, positions["scatter"], normalized_landmarks)

        # Calculate measures and update buffers
        measures_normalized = gf.calculate_measures_distances(normalized_landmarks)
        measures_raw = gf.calculate_measures_distances(raw_landmarks)

        time_series_buffer_normalized.append(measures_normalized["m3"] if measures_normalized else None)
        time_series_buffer_raw.append(measures_raw["m3"] if measures_raw else None)
        
        # Start FIFOs with defined size if it wasan't started before
        if not fifo.measures_fifos:
            fifo.initialize_fifos(measures_normalized.keys(), FIFO_SIZE)

        #Add values to the fifo
        fifo.update_fifos(measures_normalized)

        #Inference Matrix            
        matrices_subfifos = fifo.prepare_subfifo_matrix()

        emotion_classes = []
        probs = []
        
        global SEAL_COLOR, SEAL_COUNTER

        # Verify if subFIFOS preparation was successeful
        if matrices_subfifos is not None:
            # Inference for the 3 subfifos
            results = fifo.infer_emotions_for_subfifos(matrices_subfifos)
            # Process each subFIFO result (A, B, C)
            for result in results:
                subfifo_name, emotion_class, prob = result  # Unpacks the 3 values
                # Define a color based on the class
                new_color = COLOR_STD # standard color
                if emotion_class == 3:
                    #new_color = COLOR_INTERVIEW
                    emotion_class = "interview"
                elif emotion_class == 0:
                    emotion_class = "about"
                    new_color = COLOR_ABOUT
                elif emotion_class == 2:
                    #new_color = COLOR_HAVE
                    emotion_class = "have"
                else:
                    #new_color = COLOR_ALEATORIO
                    emotion_class = "aleatorio"
                # Shows the inference
                print(f"SubFIFO {subfifo_name} - {emotion_class}, Probabilidade: {prob:.2f}")

                # Graphic Window that will be colored
                SEAL_COLOR = new_color
                SEAL_COUNTER = WINDOW_SIZE

                emotion_classes.append(emotion_class)
                probs.append(prob)

        # Update color buffer 
        if SEAL_COUNTER > 0:
            color_buffer.append(SEAL_COLOR)
            SEAL_COUNTER -= 1
        else:
            color_buffer.append(COLOR_STD)

        # Plot the time-series   
        canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart"], list(time_series_buffer_normalized), color_buffer=color_buffer)
        
        # Plot the raw time-series in the additional line chart section
        canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart_raw"], list(time_series_buffer_raw), color=(255, 0, 0))

        #FPS + RAM Usage
        loop_end = time.perf_counter()
        ram_usage = process.memory_info().rss / 1024 / 1024  # in MB
        if video_path: # Sincronize the execution with video real time
                loop_duration = loop_end - loop_start
                # Wait the necessesary time to keep frame_duration (example: 1/30s = 0.033s)
                sleep_time = max(0, frame_duration - loop_duration)
                time.sleep(sleep_time)
                # FPS after sleep
                fps = 1.0 / (loop_duration + sleep_time) if (loop_duration + sleep_time) > 0 else 0.0
        else:
            fps = 1.0 / (loop_end - loop_start) if loop_end > loop_start else 0.0
    
        label_text = f"FPS: {fps:.2f} | RAM: {ram_usage:.1f} MB"
        fps_pos = (positions["camera"][0] + 10, positions["camera"][1] + 20)
        cv2.putText(canvas, label_text, fps_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if display:
            cv2.imshow("Canvas", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    landmark_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Capture ended.")

def main():
    parser = argparse.ArgumentParser(description="Process video or capture from camera.")
    parser.add_argument('-d', '--display', action='store_true', help='display frames in real time')
    parser.add_argument('-v', '--video', type=str, help='Path to video file (if provided, processes video instead of camera)')
    
    args = parser.parse_args()

    video_capture_with_canvas(args.video, args.display)

if __name__ == "__main__":
    main()
