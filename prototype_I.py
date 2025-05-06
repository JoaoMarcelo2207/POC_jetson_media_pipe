from typing import OrderedDict
import dlib
import cv2
from imutils import face_utils
import sys, os, argparse
import tensorflow as tf
from collections import deque

# Import additional functions
sys.path.append(os.path.join("lib"))
import video_adjuster_functions as vid_adj_fun, fifo_manager as fifo, graphic_functions as gf

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

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


def camera_capture_with_canvas(detector, predictor, display):
    """
    Captures frames from the camera and displays them on a canvas with sections for the camera, scatter plot, and line chart.
    Adds an additional line chart below the existing one, showing raw (non-normalized) measures.
    Overlays facial landmarks correctly aligned to the face.
    """
    #Capture from the camera
    cap = cv2.VideoCapture(0)

    #paths for the NN model
    model_path = r"C:\Users\joao.miranda\Documents\POC\Neural Network [POC]\model_for_3_emotions.keras"
    model = tf.keras.models.load_model(model_path)

    if not cap.isOpened():
        print("Error accessing the camera.")
        return

    print("Capturing frames from the camera...")

    # Canvas dimensions and sections
    canvas, positions, camera_space, scatter_space, line_chart_space = gf.create_canvas(
        camera_space=(600, 400), scatter_space=(600, 400), line_chart_space=(1200, 200)
    )

    # Adjust canvas for an additional line chart
    canvas_height = canvas.shape[0] + line_chart_space[1]
    canvas = cv2.resize(canvas, (canvas.shape[1], canvas_height))
    positions["line_chart_raw"] = (positions["line_chart"][0], positions["line_chart"][1] + line_chart_space[1])

    # Initialize buffers for the time-series history
    time_series_buffer_normalized = [0] * 200
    time_series_buffer_raw = [0] * 200

    while True:
        ret, img_all = cap.read()

        if not ret or img_all is None:
            print("Error capturing frame from the camera.")
            break

        img_gray = cv2.cvtColor(img_all, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray, 0)

        if len(faces) != 0:
            face = faces[0]
            landmarks = face_utils.shape_to_np(predictor(img_gray, face))

            # Normalized and raw points
            CENTER_NOSE_BASED_POINTS_NORMALIZED = gf.normalization(landmarks, (gf.normal_height_img, gf.normal_width_img))
            CENTER_NOSE_BASED_POINTS_RAW = landmarks  # Direct raw points without normalization

            # Overlay landmarks on the original frame before resizing
            for (x, y) in landmarks:
                cv2.circle(img_all, (x, y), 2, (0, 255, 0), -1)

            # Resize the camera frame to the designated section
            camera_frame = cv2.resize(img_all, (camera_space[0], camera_space[1]))

            canvas[positions["camera"][1]:positions["camera"][1] + camera_space[1],
                   positions["camera"][0]:positions["camera"][0] + camera_space[0]] = camera_frame

            # Plot points in the scatter section
            canvas = gf.plot_scatter(canvas, scatter_space, positions["scatter"], CENTER_NOSE_BASED_POINTS_NORMALIZED)

            # Calculate measures and update buffers
            measures_normalized = gf.calculate_measures_distances(CENTER_NOSE_BASED_POINTS_NORMALIZED)
            measures_raw = gf.calculate_measures_distances(CENTER_NOSE_BASED_POINTS_RAW)

            new_value_norm = measures_normalized["m3"] if measures_normalized else None
            new_value_raw = measures_raw["m3"] if measures_raw else None

            time_series_buffer_normalized.append(new_value_norm)
            time_series_buffer_raw.append(new_value_raw)
            
            #FIFO
            FIFO_SIZE = 150  # Definição do tamanho da FIFO

            # Inicializa os FIFOs com o tamanho definido
            if not fifo.measures_fifos:
                fifo.initialize_fifos(measures_normalized.keys(), FIFO_SIZE)

            fifo.update_fifos(measures_normalized)

            # Obter a matriz no formato (32, 22)
            #fifo_matrix = fifo.get_fifo_matrix()

            #Matriz para inferencia
            
            fifo_matrix_inf = fifo.prepare_data_for_inference(45)

            emotion_class = None
            prob = None
            
            if fifo_matrix_inf is not None:
                emotion_class, prob = fifo.infer_emotion(model, fifo_matrix_inf)
            if emotion_class is not None and prob is not None:
                print(f"Classe Predita: {emotion_class}, Probabilidade: {prob:.2f}")
            else:
                print("Inferência falhou: matriz inválida.")

            # Plot the time-series   
            canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart"], list(time_series_buffer_normalized), emotion_class)
            
            # Plot the raw time-series in the additional line chart section
            canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart_raw"], list(time_series_buffer_raw), color=(255, 0, 0))

        if display:
            cv2.imshow("Canvas", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Capture ended.")


def offline_video_capture_with_canvas(video_path, detector, predictor, display):
    """
    Processes a video, captures frames, and displays real-time analytics on a canvas.
    No subtitles are used in this version.
    """
    model_path = r"C:\Users\joao.miranda\Documents\POC\POC\Neural Network [POC]\Model_protD.keras"

    model = tf.keras.models.load_model(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    print(f"Processing video: {video_path}")

    # Create a blank canvas with sections for different plots
    canvas, positions, camera_space, scatter_space, line_chart_space = gf.create_canvas(
        camera_space=(600, 400), scatter_space=(600, 400), line_chart_space=(1200, 200)
    )

    # Adjust canvas to include an additional line chart for raw (non-normalized) measures
    canvas_height = canvas.shape[0] + line_chart_space[1]
    canvas = cv2.resize(canvas, (canvas.shape[1], canvas_height))
    positions["line_chart_raw"] = (positions["line_chart"][0], positions["line_chart"][1] + line_chart_space[1])
    
    # Initialize buffers for the time-series history
  
    time_series_buffer_normalized = deque(maxlen=gf.PLOT_SIZE)
    time_series_buffer_raw = deque(maxlen=gf.PLOT_SIZE)

    # Inicializa o buffer de cores vazio
    color_buffer = deque(maxlen=gf.PLOT_SIZE)
    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        if faces:
            face = faces[0]
            landmarks = face_utils.shape_to_np(predictor(gray_frame, face))

             # Overlay landmarks on the original frame before resizing
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Normalized and raw points
            normalized_points = gf.normalization(landmarks, (gf.normal_height_img, gf.normal_width_img))
            raw_points = landmarks  # Direct raw points without normalization

            # Resize and place the video frame on the canvas
            resized_frame = cv2.resize(frame, (camera_space[0], camera_space[1]))
            canvas[positions["camera"][1]:positions["camera"][1] + camera_space[1],
                   positions["camera"][0]:positions["camera"][0] + camera_space[0]] = resized_frame

            # Plot scatter points
            canvas = gf.plot_scatter(canvas, scatter_space, positions["scatter"], normalized_points)

            # Calculate measures and update buffers
            measures_normalized = gf.calculate_measures_distances(normalized_points)
            measures_raw = gf.calculate_measures_distances(raw_points)

            new_value_norm = measures_normalized["m3"] if measures_normalized else None
            new_value_raw = measures_raw["m3"] if measures_raw else None

            time_series_buffer_normalized.append(new_value_norm)
            time_series_buffer_raw.append(new_value_raw)
             
            #FIFO
            FIFO_SIZE = 150  # Definição do tamanho da FIFO

            # Inicializa os FIFOs com o tamanho definido
            if not fifo.measures_fifos:
                fifo.initialize_fifos(measures_normalized.keys(), FIFO_SIZE)

            fifo.update_fifos(measures_normalized)

            # validação de subfifo e fifo
            #fifo.check_subfifos_shiftando()

            # Obter a matriz no formato (32, 22)
            #fifo_matrix = fifo.get_fifo_matrix()

            #Matriz para inferencia

            # Preparar a matriz de entrada para as subFIFOs
            matrices_subfifos = fifo.prepare_subfifo_matrix()
           
            emotion_classes = []
            probs = []

            global SEAL_COLOR, SEAL_COUNTER

            # Verifica se a preparação das subFIFOs foi bem-sucedida
            if matrices_subfifos is not None:
                # Faz a inferência para as 3 subFIFOs
                results = fifo.infer_emotions_for_subfifos(model, matrices_subfifos)
                
                # Processa os resultados para cada subFIFO (A, B, C)
                for result in results:
                    subfifo_name, emotion_class, prob = result  # Desempacota corretamente os 3 valores
                    
                    # Define a cor com base na classe
                    new_color = COLOR_STD # cor padrão
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
                    # Exibe a inferência
                    print(f"SubFIFO {subfifo_name} - {emotion_class}, Probabilidade: {prob:.2f}")

                    # Ativa o selamento por WINDOW_SIZE frames
                    SEAL_COLOR = new_color
                    SEAL_COUNTER = WINDOW_SIZE

                    emotion_classes.append(emotion_class)
                    probs.append(prob)

            # Atualiza o buffer de cores dinamicamente
            if SEAL_COUNTER > 0:
                color_buffer.append(SEAL_COLOR)
                SEAL_COUNTER -= 1
            else:
                color_buffer.append(COLOR_STD)

            # Plota o gráfico normalizado com as cores atualizadas
            canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart"], list(time_series_buffer_normalized), color_buffer=color_buffer)
            
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

    # DLIB
    LANDMARK_FILE = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_FILE)

    if args.video:
        print(f"Processing video: {args.video}")
        offline_video_capture_with_canvas(args.video, detector, predictor, args.display)
    else:
        print("Starting camera capture...")
        camera_capture_with_canvas(detector, predictor, args.display)

if __name__ == "__main__":
    main()
