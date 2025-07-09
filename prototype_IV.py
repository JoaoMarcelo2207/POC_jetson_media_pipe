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



# Configurações do gráfico
COLOR_STD = (0, 0, 255)    # Vermelho - padrão
COLOR_ABOUT = (255, 255, 0)  # Azul - classe "about"
COLOR_INTERVIEW = (0, 255, 0) # Verde - classe "interview"
COLOR_HAVE = (255,0,0)
color_buffer = deque(maxlen=gf.PLOT_SIZE)
for _ in range(gf.PLOT_SIZE):
    color_buffer.append(COLOR_STD)  # Buffer de cores
WINDOW_SIZE = 10  # Janela de pontos a serem coloridos após a predição
SEAL_COLOR = None
LAST_COLOR = COLOR_STD
SEAL_COUNTER = 0


def video_capture_with_canvas(video_path, display):
    """
    Captures frames from the camera or video and displays them on a canvas with sections for the camera, scatter plot, and line chart.
    Adds an additional line chart below the existing one, showing raw (non-normalized) measures.
    Overlays facial landmarks correctly aligned to the face.
    """
    
    # Abrir vídeo ou webcam
    if video_path:
        print(f"Processing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps_cv2 = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS original do vídeo: {fps_cv2:.2f}")
        frame_duration = 1.0 / fps_cv2
    else:
        print("Capturing from webcam...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        # Força resolução
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

    #Monitorar Ram
    pid = os.getpid()
    process = psutil.Process(pid)
    
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
        
        # Envia o frame para a thread processar
        landmark_thread.set_frame(img_all)
        # Pega o resultado da thread (da iteração anterior)
        landmarks = landmark_thread.get_landmarks()
        print(f"RAM usada (MB): {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f}")

        if landmarks is None:
            continue

        if landmarks is not None:

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
            
            #FIFO
            FIFO_SIZE = 150  # Definição do tamanho da FIFO

            # Inicializa os FIFOs com o tamanho definido
            if not fifo.measures_fifos:
                fifo.initialize_fifos(measures_normalized.keys(), FIFO_SIZE)

            fifo.update_fifos(measures_normalized)


            #Matriz para inferencia            
            matrices_subfifos = fifo.prepare_subfifo_matrix()

            emotion_classes = []
            probs = []
            
            global SEAL_COLOR, SEAL_COUNTER
            # Verifica se a preparação das subFIFOs foi bem-sucedida
            if matrices_subfifos is not None:
                # Faz a inferência para as 3 subFIFOs
                results = fifo.infer_emotions_for_subfifos(matrices_subfifos)
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

            # Plot the time-series   
            canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart"], list(time_series_buffer_normalized), color_buffer=color_buffer)
            
            # Plot the raw time-series in the additional line chart section
            canvas = gf.plot_line_chart(canvas, line_chart_space, positions["line_chart_raw"], list(time_series_buffer_raw), color=(255, 0, 0))

            # 10. FPS + uso de RAM
            loop_end = time.perf_counter()
            ram_usage = process.memory_info().rss / 1024 / 1024  # em MB
            if video_path: # Sincronizar a execução com o tempo real do vídeo (reprodução normalizada)
                    loop_duration = loop_end - loop_start
                    # Espera o tempo necessário para manter frame_duration (ex: 1/30s = 0.033s)
                    sleep_time = max(0, frame_duration - loop_duration)
                    time.sleep(sleep_time)
                    # Agora mede o FPS após o sleep
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
