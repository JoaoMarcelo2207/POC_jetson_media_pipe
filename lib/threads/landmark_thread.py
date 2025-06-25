import threading
import cv2
import mediapipe as mp

class LandmarkProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.input_frame = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.results = None
        self.landmarks = None

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def set_frame(self, frame):
        with self.lock:
            self.input_frame = frame
        self.new_frame_event.set()

    def get_landmarks(self):
        with self.lock:
            return self.landmarks

    def run(self):
        while self.running:
            self.new_frame_event.wait()
            self.new_frame_event.clear()

            with self.lock:
                frame = self.input_frame.copy() if self.input_frame is not None else None

            if frame is None:
                continue

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
                # Faz a reescala dos pontos detectados pro frame original
                scale_x = orig_w / resized_w
                scale_y = orig_h / resized_h

                for lm in results.multi_face_landmarks[0].landmark:
                    x = int(lm.x * resized_w * scale_x)
                    y = int(lm.y * resized_h * scale_y)
                    landmarks.append((x, y))

                with self.lock:
                    self.landmarks = landmarks
            else:
                with self.lock:
                    self.landmarks = None



    def stop(self):
        self.running = False
        self.new_frame_event.set()
