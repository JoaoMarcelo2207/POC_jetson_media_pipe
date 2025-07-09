from typing import OrderedDict
import cv2
import sys, os, argparse
import tensorflow as tf
from collections import deque
import mediapipe as mp


# Import additional functions
sys.path.append(os.path.join("../lib"))
import video_adjuster_functions as vid_adj_fun, fifo_manager as fifo, graphic_functions as gf


# Configurações do gráfico
COLOR_STD = (0, 0, 255)    # Vermelho - padrão
COLOR_ABOUT = (255, 0, 0)  # Azul - classe "about"
COLOR_INTERVIEW = (0, 255, 0) # Verde - classe "interview"
color_buffer = deque(maxlen=gf.PLOT_SIZE)
for _ in range(gf.PLOT_SIZE):
    color_buffer.append(COLOR_STD)  # Buffer de cores
WINDOW_SIZE = 15  # Janela de pontos a serem coloridos após a predição
SEAL_COLOR = None
LAST_COLOR = COLOR_STD
SEAL_COUNTER = 0

def video_capture_with_canvas(video_path, display):
    """
    Captures frames from the camera and displays them on a canvas with sections for the camera, scatter plot, and line chart.
    Adds an additional line chart below the existing one, showing raw (non-normalized) measures.
    Overlays facial landmarks correctly aligned to the face.
    """

    #paths for the NN model
    model_path = r"C:\Users\joao.miranda\Documents\POC\POC_jetson_media_pipe\Neural Network [POC]\model_for_3_emotions.keras"
    model = tf.keras.models.load_model(model_path)
    
    # Abrir vídeo ou webcam
    if video_path:
        print(f"Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("Capturing from webcam...")
        cap = cv2.VideoCapture(0)

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

    # Inicializa o buffer de cores vazio
    color_buffer = deque(maxlen=gf.PLOT_SIZE)
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                           max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, img_all = cap.read()

            if (not ret or img_all is None) and video_path is not None:
                print("End of video or error reading frame.")
                break
            elif not ret or img_all is None:
                print("Error capturing frame from the camera.")
                break

            img_rgb = cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)      
            
            if results.multi_face_landmarks:
                landmarks = []

                for lm in results.multi_face_landmarks[0].landmark:
                    h, w, _ = img_all.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))

                # Normalized and raw points
                normalized_points = gf.normalization(landmarks, (gf.normal_height_img, gf.normal_width_img))
                raw_points = landmarks  # Direct raw points without normalization

                # Overlay landmarks on the original frame before resizing
                for (x, y) in landmarks:
                    cv2.circle(img_all, (x, y), 2, (0, 255, 0), -1)

                # Resize the camera frame to the designated section
                camera_frame = cv2.resize(img_all, (camera_space[0], camera_space[1]))

                canvas[positions["camera"][1]:positions["camera"][1] + camera_space[1],
                    positions["camera"][0]:positions["camera"][0] + camera_space[0]] = camera_frame

                # Plot points in the scatter section
                canvas = gf.plot_scatter(canvas, scatter_space, positions["scatter"], normalized_points)

                # Calculate measures and update buffers
                measures_normalized = gf.calculate_measures_distances(normalized_points)
                measures_raw = gf.calculate_measures_distances(raw_points)

                new_value_norm = measures_normalized["m3"] if measures_normalized else None
                new_value_raw = measures_raw["m3"] if measures_raw else None

                time_series_buffer_normalized.append(new_value_norm)
                time_series_buffer_raw.append(new_value_raw)
            
                # Plot the time-series   
                canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart"], list(time_series_buffer_normalized), color = (255, 0, 255))
                
                # Plot the raw time-series in the additional line chart section
                canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart_raw"], list(time_series_buffer_raw), color=(255, 0, 0))

            if display:
                cv2.imshow("Canvas", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
