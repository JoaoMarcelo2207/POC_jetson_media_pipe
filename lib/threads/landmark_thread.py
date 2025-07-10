import threading
import cv2
import mediapipe as mp
import time
from queue import Queue, Empty

class LandmarkProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True

        # Fila com tamanho 1: sempre mantém o frame mais recente
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def set_frame(self, frame):
        # Substitui o frame atual na fila (se houver)
        if not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
        self.input_queue.put_nowait(frame)

    def get_landmarks(self):
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None

    def run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1)
            except Empty:
                continue
            
            loop_thread_start = time.perf_counter()
            # Salva resolução original
            orig_h, orig_w, _ = frame.shape

            # Redimensiona para acelerar a inferência
            resized_frame = cv2.resize(frame, (320, 180))
            resized_h, resized_w = resized_frame.shape[:2]

            # Converte pra RGB e roda inferência
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                landmarks = []
                # Reescala os pontos para o frame original
                scale_x = orig_w / resized_w
                scale_y = orig_h / resized_h

                for lm in results.multi_face_landmarks[0].landmark:
                    x = int(lm.x * resized_w * scale_x)
                    y = int(lm.y * resized_h * scale_y)
                    landmarks.append((x, y))

                # Substitui landmarks anteriores na fila
                if not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except Empty:
                        pass
                self.output_queue.put_nowait(landmarks)

                loop_thread_end = time.perf_counter()
                loop_duration_thread = loop_thread_end - loop_thread_start
                int(f"Tempo de processamento: {loop_duration_thread * 1000:.2f} ms")

            else:
                # Também limpa a fila se não houve detecção
                if not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except Empty:
                        pass
                self.output_queue.put_nowait(None)

    def stop(self):
        self.running = False
