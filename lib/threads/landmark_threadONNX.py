import threading
import cv2
import numpy as np
import onnxruntime as ort

class LandmarkProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.input_frame = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.landmarks = None

        facemesh_path = "/home/joaomarcelohpc/Documents/POC_jetson_media_pipe/07 - Neural Network/better-facemesh/face_mesh_Nx3x192x192_post.onnx"
        yoloV4_path = "/home/joaomarcelohpc/Documents/POC_jetson_media_pipe/07 - Neural Network/better-facemesh/yolov4_headdetection_480x640_post.onnx"
        # Session do modelo de FaceMesh (PINTO0309)
        self.session = ort.InferenceSession(facemesh_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Assume-se que o detector facial já foi inicializado fora dessa thread
        self.face_detector = cv2.dnn.readNetFromONNX(yoloV4_path)

    def set_frame(self, frame):
        with self.lock:
            self.input_frame = frame.copy()
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

            img_height, img_width = frame.shape[:2]

            # --- 1. DETECÇÃO FACIAL COM YOLO ---
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(480, 640), swapRB=True)
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()

            if detections.shape[1] == 0:
                with self.lock:
                    self.landmarks = None
                continue

            # Pegamos o primeiro rosto detectado
            box = detections[0, 0]  # ou ajuste se o output for diferente
            conf = box[2]
            if conf < 0.4:
                with self.lock:
                    self.landmarks = None
                continue

            x1, y1, x2, y2 = box[3:7] * np.array([img_width, img_height, img_width, img_height])
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # --- 2. MARGEM DE 25% ---
            w, h = x2 - x1, y2 - y1
            margin_x = int(0.25 * w)
            margin_y = int(0.25 * h)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(img_width, x2 + margin_x)
            y2 = min(img_height, y2 + margin_y)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                with self.lock:
                    self.landmarks = None
                continue

            # --- 3. PRÉ-PROCESSAMENTO ---
            resized = cv2.resize(crop, (192, 192))
            img_input = resized.astype(np.float32) / 255.0
            img_input = img_input.transpose(2, 0, 1)  # HWC -> CHW
            img_input = img_input[np.newaxis, :]  # [1, 3, 192, 192]

            # --- 4. INFERÊNCIA ---
            ort_inputs = {self.session.get_inputs()[0].name: img_input}
            outputs = self.session.run(None, ort_inputs)
            landmarks = outputs[0][0]  # shape (468, 3)

            # --- 5. PÓS-PROCESSAMENTO (coordenadas absolutas no frame original) ---
            landmarks[:, 0] = landmarks[:, 0] * (x2 - x1) + x1  # x
            landmarks[:, 1] = landmarks[:, 1] * (y2 - y1) + y1  # y
            landmarks = [(int(x), int(y)) for x, y, _ in landmarks]

            with self.lock:
                self.landmarks = landmarks

    def stop(self):
        self.running = False
        self.new_frame_event.set()
