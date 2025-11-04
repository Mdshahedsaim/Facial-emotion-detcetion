import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
import threading
from queue import Queue

class OptimizedEmotionDetector:
    def __init__(self):
        self.load_model()
        self.setup_camera()
        self.setup_detection()
        
        # Threading for async processing
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.stop_processing = False
        
    def load_model(self):
        try:
            self.model = load_model('best_emotion_model.h5')
            print("Model loaded successfully!")
        except:
            try:
                self.model = load_model('emotion_model_improved.h5')
                print("Fallback model loaded successfully!")
            except:
                self.model = load_model('emotion_model.h5')
                print("Original model loaded successfully!")

        try:
            self.emotion_labels = []
            with open('emotion_labels_improved.txt', 'r') as f:
                for line in f:
                    parts = line.strip().split(': ')
                    if len(parts) == 2:
                        self.emotion_labels.append(parts[1])
        except:
            self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            exit()
            
        # Better balanced camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Keep good resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def setup_detection(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prediction_buffer = deque(maxlen=4)  # Optimal smoothing
        self.confidence_threshold = 0.3
        
        # Frame processing for target FPS
        self.frame_skip = 2  # Process every 2nd frame for 20-28 FPS
        self.frame_count = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_display = 0
        self.target_fps = 25  # Target FPS
        self.frame_time = 1.0 / self.target_fps  # Target frame time
        
        # Cache last detection
        self.last_detection = None
        self.detection_age = 0
        self.max_detection_age = 4  # Cache for 4 frames

    def preprocess_face(self, roi_gray):
        """Better balanced preprocessing"""
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_LINEAR)
        roi = roi_gray.astype('float32') / 255.0
        
        # Keep histogram equalization for better accuracy
        roi = cv2.equalizeHist((roi * 255).astype('uint8')).astype('float32') / 255.0
        
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        return roi

    def process_frame_async(self):
        """Async frame processing thread optimized for 20-28 FPS"""
        while not self.stop_processing:
            try:
                if not self.frame_queue.empty():
                    gray_frame = self.frame_queue.get_nowait()
                    
                    # Optimized face detection for target FPS
                    faces = self.face_classifier.detectMultiScale(
                        gray_frame,
                        scaleFactor=1.15,   # Balanced speed/accuracy
                        minNeighbors=4,     # Good detection quality
                        minSize=(40, 40),   # Reasonable minimum size
                        maxSize=(280, 280),
                        flags=cv2.CASCADE_SCALE_IMAGE  # Optimization flag
                    )
                    
                    results = []
                    # Limit to 2 faces max for consistent performance
                    for (x, y, w, h) in faces[:2]:  
                        roi_gray = gray_frame[y:y+h, x:x+w]
                        roi = self.preprocess_face(roi_gray)
                        
                        try:
                            prediction = self.model.predict(roi, verbose=0)[0]
                            max_confidence = np.max(prediction)
                            
                            if max_confidence > self.confidence_threshold:
                                emotion_idx = np.argmax(prediction)
                                emotion = self.emotion_labels[emotion_idx]
                                
                                # Simplified smoothing for performance
                                self.prediction_buffer.append((emotion, max_confidence))
                                
                                if len(self.prediction_buffer) >= 2:
                                    # Get most recent emotions for quick smoothing
                                    recent_emotions = [pred[0] for pred in list(self.prediction_buffer)[-3:]]
                                    if recent_emotions.count(emotion) >= 2:
                                        display_emotion = emotion
                                        display_confidence = max_confidence
                                    else:
                                        # Use most common from recent predictions
                                        display_emotion = max(set(recent_emotions), key=recent_emotions.count)
                                        display_confidence = max_confidence
                                else:
                                    display_emotion = emotion
                                    display_confidence = max_confidence
                                
                                results.append((x, y, w, h, display_emotion, display_confidence))
                        except:
                            pass
                    
                    if not self.result_queue.full():
                        self.result_queue.put_nowait(results)
                        
            except:
                pass
            
            # Control processing speed for target FPS
            time.sleep(0.008)  # 8ms delay for ~25 FPS processing

    def get_emotion_color(self, emotion):
        """Fast color mapping"""
        color_map = {
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Angry': (0, 0, 255),
            'Fear': (255, 0, 255),
            'Surprise': (0, 255, 255),
            'Disgust': (128, 0, 128),
            'Neutral': (128, 128, 128)
        }
        return color_map.get(emotion, (0, 255, 0))

    def draw_detection(self, frame, x, y, w, h, emotion, confidence):
        """Optimized drawing"""
        color = self.get_emotion_color(emotion)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label with background
        label = f"{emotion} ({confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x, y-text_height-8), (x+text_width, y), color, -1)
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Simplified confidence bar
        bar_width = w
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(frame, (x, y+h+2), (x+confidence_width, y+h+6), color, -1)

    def run(self):
        """Main detection loop"""
        print("Starting optimized emotion detection... Press 'q' to quit")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frame_async)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        current_results = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            
            # Process frames asynchronously
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Add frame to queue for async processing
            if not self.frame_queue.full():
                try:
                    self.frame_queue.put_nowait(gray.copy())
                except:
                    pass
            
            # Get results from processing thread
            try:
                if not self.result_queue.empty():
                    current_results = self.result_queue.get_nowait()
                    self.detection_age = 0
                else:
                    self.detection_age += 1
            except:
                pass
            
            # Use cached results if recent
            if self.detection_age < self.max_detection_age:
                for result in current_results:
                    x, y, w, h, emotion, confidence = result
                    self.draw_detection(frame, x, y, w, h, emotion, confidence)
            
            # Update FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.fps_display = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Draw UI info
            cv2.putText(frame, f"FPS: {self.fps_display}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(current_results)}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Emotion Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped and resources cleaned up.")

if __name__ == "__main__":
    detector = OptimizedEmotionDetector()
    detector.run()