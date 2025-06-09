from queue import Queue, Empty
import threading
import cv2
import mediapipe as mp

class LandmarkProcessor(threading.Thread):
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
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
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)  # espera por novo frame
            except Empty:
                continue

            if frame is None:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
                self.output_queue.put(landmarks)
            else:
                self.output_queue.put(None)
 
    def stop(self):
        self.running = False
