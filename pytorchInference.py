# import cv2
# import torch
# import torch.nn.functional as F
# import numpy as np
# import mediapipe as mp
# import pickle
# import json
# import os
# import time
# from collections import deque
# import threading
# from datetime import datetime
# import argparse
# from pytorchModelBuilder import ISLViTModel 
# from config import ISLConfig

# # Import your model classes (assuming they're in separate files)
# # from model import ISLViTModel  # Uncomment if importing from separate file
# # from config import ISLConfig   # Uncomment if importing from separate file

# class ISLRealTimeInference:
#     """Real-time ISL gesture detection with camera feed"""
    
#     def __init__(self, model_path, config_path=None, device=None):
#         self.model_path = model_path
#         self.device = self._setup_device(device)
        
#         # Load model and configuration
#         self._load_model()
        
#         # Initialize MediaPipe
#         self._setup_mediapipe()
        
#         # Initialize frame buffer for temporal sequences
#         self.frame_buffer = deque(maxlen=self.config.SEQUENCE_LENGTH)
#         self.landmark_buffer = deque(maxlen=self.config.SEQUENCE_LENGTH)
        
#         # Prediction smoothing
#         self.prediction_buffer = deque(maxlen=5)  # Last 5 predictions
#         self.confidence_threshold = 0.6
        
#         # UI settings
#         self.font = cv2.FONT_HERSHEY_SIMPLEX
#         self.font_scale = 1.2
#         self.font_thickness = 2
        
#         # Colors (BGR format)
#         self.colors = {
#             'primary': (255, 87, 34),      # Deep Orange
#             'secondary': (76, 175, 80),     # Green
#             'background': (33, 33, 33),     # Dark Gray
#             'text': (255, 255, 255),        # White
#             'warning': (255, 193, 7),       # Amber
#             'error': (244, 67, 54)          # Red
#         }
        
#         # Performance tracking
#         self.fps_counter = 0
#         self.fps_start_time = time.time()
#         self.current_fps = 0
        
#         print(f"✅ ISL Real-time Inference initialized on {self.device}")
    
#     def _setup_device(self, device):
#         """Setup computation device"""
#         if device is not None:
#             return torch.device(device)
#         elif torch.backends.mps.is_available():
#             return torch.device("mps")
#         elif torch.cuda.is_available():
#             return torch.device("cuda")
#         else:
#             return torch.device("cpu")
    
#     def _load_model(self):
#         """Load the trained model and configuration"""
#         print(f"Loading model from {self.model_path}...")
        
#         # Load checkpoint
#         checkpoint = torch.load(self.model_path, map_location=self.device)
        
#         # Reconstruct config from checkpoint or use default
#         if 'config' in checkpoint:
#             config_dict = checkpoint['config']
#             # Create a simple config object
#             class Config:
#                 def __init__(self, **kwargs):
#                     for key, value in kwargs.items():
#                         setattr(self, key, value)
            
#             self.config = Config(**config_dict)
#         else:
#             # Default configuration if not found in checkpoint
#             print("We are shifting to default config")
#             class Config:
#                 SEQUENCE_LENGTH = 16
#                 IMG_SIZE = (224, 224)
#                 EMBED_DIM = 512
#                 NUM_HEADS = 8
#                 DROPOUT_RATE = 0.1
#                 MAX_NUM_HANDS = 2
#                 MIN_DETECTION_CONFIDENCE = 0.7
#                 MIN_TRACKING_CONFIDENCE = 0.5
            
#             self.config = Config()
        
#         # Load label encoder
#         label_encoder_path = os.path.join(os.path.dirname(self.model_path), 'label_encoder.pkl')
#         if os.path.exists(label_encoder_path):
#             with open(label_encoder_path, 'rb') as f:
#                 self.label_encoder = pickle.load(f)
#         else:
#             print("Warning: Label encoder not found. Using default class names.")
#             # Create dummy label encoder for demo
#             class DummyEncoder:
#                 def __init__(self):
#                     self.classes_ = [f'Gesture_{i}' for i in range(10)]
                
#                 def inverse_transform(self, labels):
#                     return [self.classes_[i] if i < len(self.classes_) else 'Unknown' for i in labels]
            
#             self.label_encoder = DummyEncoder()
        
#         self.num_classes = len(self.label_encoder.classes_)
        
#         # Initialize and load model
        
#         self.model = ISLViTModel(self.config, self.num_classes)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.to(self.device)
#         self.model.eval()
        
#         print(f"✅ Model loaded with {self.num_classes} classes")
#         print(f"📋 Classes: {self.label_encoder.classes_[:5]}..." if len(self.label_encoder.classes_) > 5 else f"📋 Classes: {self.label_encoder.classes_}")
    
#     def _setup_mediapipe(self):
#         """Initialize MediaPipe components"""
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=self.config.MAX_NUM_HANDS,
#             min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
#             min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
#         )
        
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=False,
#             min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
#             min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
#         )
        
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.mp_drawing_styles = mp.solutions.drawing_styles
    
#     def extract_landmarks(self, frame):
#         """Extract hand and pose landmarks from frame"""
#         landmarks = []
        
#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Hand landmarks
#         hand_results = self.hands.process(rgb_frame)
#         hand_landmarks_list = [np.zeros(63), np.zeros(63)]
        
#         if hand_results.multi_hand_landmarks:
#             for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
#                 if i >= 2:
#                     break
#                 coords = []
#                 for landmark in hand_landmarks.landmark:
#                     coords.extend([landmark.x, landmark.y, landmark.z])
#                 hand_landmarks_list[i] = np.array(coords)
        
#         landmarks.extend(hand_landmarks_list[0])
#         landmarks.extend(hand_landmarks_list[1])
        
#         # Pose landmarks
#         pose_results = self.pose.process(rgb_frame)
#         if pose_results.pose_landmarks:
#             for i in range(11):
#                 landmark = pose_results.pose_landmarks.landmark[i]
#                 landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
#         else:
#             landmarks.extend([0.0] * 44)
        
#         return np.array(landmarks), hand_results, pose_results
    
#     def draw_landmarks(self, frame, hand_results, pose_results):
#         """Draw MediaPipe landmarks on frame"""
#         # Draw hand landmarks
#         if hand_results.multi_hand_landmarks:
#             for hand_landmarks in hand_results.multi_hand_landmarks:
#                 self.mp_drawing.draw_landmarks(
#                     frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
#                     self.mp_drawing_styles.get_default_hand_landmarks_style(),
#                     self.mp_drawing_styles.get_default_hand_connections_style()
#                 )
        
#         # Draw pose landmarks
#         if pose_results.pose_landmarks:
#             self.mp_drawing.draw_landmarks(
#                 frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
#             )
    
#     def preprocess_frame(self, frame):
#         """Preprocess frame for model input"""
#         # Resize to model input size
#         frame_resized = cv2.resize(frame, self.config.IMG_SIZE)
#         # Normalize to [0, 1]
#         frame_normalized = frame_resized.astype(np.float32) / 255.0
#         return frame_normalized
    
#     def predict_gesture(self):
#         """Predict gesture from current buffer"""
#         if len(self.frame_buffer) < self.config.SEQUENCE_LENGTH:
#             return None, 0.0
        
#         # Prepare input tensors
#         frames = np.stack(list(self.frame_buffer))  # (T, H, W, C)
#         landmarks = np.stack(list(self.landmark_buffer))  # (T, 170)
        
#         # Convert to PyTorch tensors
#         frames_tensor = torch.from_numpy(frames).float()
#         landmarks_tensor = torch.from_numpy(landmarks).float()
        
#         # Add batch dimension and permute frames
#         frames_tensor = frames_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)
#         landmarks_tensor = landmarks_tensor.unsqueeze(0)  # (1, T, 170)
        
#         # Move to device
#         frames_tensor = frames_tensor.to(self.device)
#         landmarks_tensor = landmarks_tensor.to(self.device)
        
#         # Predict
#         with torch.no_grad():
#             logits = self.model(frames_tensor, landmarks_tensor)
#             probabilities = F.softmax(logits, dim=1)
#             confidence, predicted = torch.max(probabilities, 1)
            
#             predicted_class = predicted.item()
#             confidence_score = confidence.item()
        
#         return predicted_class, confidence_score
    
#     def smooth_predictions(self, prediction, confidence):
#         """Smooth predictions over time"""
#         if prediction is None:
#             return None, 0.0
        
#         self.prediction_buffer.append((prediction, confidence))
        
#         if len(self.prediction_buffer) < 3:
#             return prediction, confidence
        
#         # Get most common prediction from recent buffer
#         recent_predictions = [p[0] for p in list(self.prediction_buffer)]
#         recent_confidences = [p[1] for p in list(self.prediction_buffer)]
        
#         # Simple majority vote
#         prediction_counts = {}
#         for pred in recent_predictions:
#             prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
#         most_common_pred = max(prediction_counts, key=prediction_counts.get)
#         avg_confidence = np.mean([conf for pred, conf in self.prediction_buffer if pred == most_common_pred])
        
#         return most_common_pred, avg_confidence
    
#     def draw_ui(self, frame, prediction, confidence, fps):
#         """Draw beautiful UI overlay"""
#         h, w = frame.shape[:2]
        
#         # Create overlay
#         overlay = frame.copy()
        
#         # Top bar
#         cv2.rectangle(overlay, (0, 0), (w, 80), self.colors['background'], -1)
#         cv2.rectangle(overlay, (0, 0), (w, 80), self.colors['primary'], 3)
        
#         # Title
#         title = "ISL Gesture Recognition"
#         title_size = cv2.getTextSize(title, self.font, 0.8, 2)[0]
#         title_x = (w - title_size[0]) // 2
#         cv2.putText(overlay, title, (title_x, 35), self.font, 0.8, self.colors['text'], 2)
        
#         # FPS counter
#         fps_text = f"FPS: {fps:.1f}"
#         cv2.putText(overlay, fps_text, (w - 120, 60), self.font, 0.5, self.colors['text'], 1)
        
#         # Prediction panel
#         if prediction is not None and confidence > self.confidence_threshold:
#             gesture_name = self.label_encoder.inverse_transform([prediction])[0]
            
#             # Main prediction box
#             box_height = 120
#             box_y = h - box_height - 20
#             cv2.rectangle(overlay, (20, box_y), (w - 20, h - 20), self.colors['background'], -1)
#             cv2.rectangle(overlay, (20, box_y), (w - 20, h - 20), self.colors['secondary'], 3)
            
#             # Gesture name
#             gesture_text = f"Gesture: {gesture_name}"
#             text_size = cv2.getTextSize(gesture_text, self.font, self.font_scale, self.font_thickness)[0]
#             text_x = (w - text_size[0]) // 2
#             cv2.putText(overlay, gesture_text, (text_x, box_y + 45), 
#                        self.font, self.font_scale, self.colors['text'], self.font_thickness)
            
#             # Confidence bar
#             conf_text = f"Confidence: {confidence:.2%}"
#             conf_size = cv2.getTextSize(conf_text, self.font, 0.7, 2)[0]
#             conf_x = (w - conf_size[0]) // 2
#             cv2.putText(overlay, conf_text, (conf_x, box_y + 80), 
#                        self.font, 0.7, self.colors['text'], 2)
            
#             # Confidence progress bar
#             bar_width = 300
#             bar_height = 15
#             bar_x = (w - bar_width) // 2
#             bar_y = box_y + 90
            
#             # Background bar
#             cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
#                          self.colors['text'], 1)
            
#             # Filled bar
#             filled_width = int(bar_width * confidence)
#             bar_color = self.colors['secondary'] if confidence > 0.8 else self.colors['warning']
#             cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
#                          bar_color, -1)
        
#         else:
#             # No prediction or low confidence
#             status_text = "Show gesture to camera..."
#             if len(self.frame_buffer) < self.config.SEQUENCE_LENGTH:
#                 status_text = f"Collecting frames... ({len(self.frame_buffer)}/{self.config.SEQUENCE_LENGTH})"
            
#             text_size = cv2.getTextSize(status_text, self.font, 0.8, 2)[0]
#             text_x = (w - text_size[0]) // 2
#             cv2.putText(overlay, status_text, (text_x, h - 40), 
#                        self.font, 0.8, self.colors['warning'], 2)
        
#         # Instructions
#         instructions = [
#             "Press 'q' to quit",
#             "Press 'r' to reset buffer",
#             "Press 's' to save screenshot"
#         ]
        
#         for i, instruction in enumerate(instructions):
#             cv2.putText(overlay, instruction, (10, h - 120 + i * 25), 
#                        self.font, 0.4, self.colors['text'], 1)
        
#         # Blend overlay
#         alpha = 0.8
#         frame_with_ui = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
#         return frame_with_ui
    
#     def calculate_fps(self):
#         """Calculate current FPS"""
#         self.fps_counter += 1
#         if self.fps_counter % 30 == 0:  # Update every 30 frames
#             current_time = time.time()
#             self.current_fps = 30 / (current_time - self.fps_start_time)
#             self.fps_start_time = current_time
        
#         return self.current_fps
    
#     def run(self, camera_id=0, show_landmarks=True, save_screenshots=False):
#         """Main inference loop"""
#         print(f"🎥 Starting camera feed (Camera ID: {camera_id})...")
#         print("📋 Controls:")
#         print("   'q' - Quit")
#         print("   'r' - Reset frame buffer") 
#         print("   's' - Save screenshot")
#         print("   'l' - Toggle landmark display")
        
#         cap = cv2.VideoCapture(camera_id)
        
#         # Set camera properties for better performance
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap.set(cv2.CAP_PROP_FPS, 30)
        
#         if not cap.isOpened():
#             print("❌ Error: Could not open camera")
#             return
        
#         print("✅ Camera opened successfully!")
#         print("🚀 Starting real-time inference...")
        
#         try:
#             i = 0
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("❌ Error reading frame")
#                     break
                
#                 # Flip frame horizontally for mirror effect
#                 frame = cv2.flip(frame, 1)
#                 print("We are flipping", i)
#                 i = i + 1
                
#                 # Extract landmarks and preprocess
#                 landmarks, hand_results, pose_results = self.extract_landmarks(frame)
#                 processed_frame = self.preprocess_frame(frame)
                
#                 # Add to buffers
#                 self.frame_buffer.append(processed_frame)
#                 self.landmark_buffer.append(landmarks)
                
#                 # Draw landmarks if enabled
#                 if show_landmarks:
#                     self.draw_landmarks(frame, hand_results, pose_results)
                
#                 # Predict gesture
#                 prediction, confidence = self.predict_gesture()
#                 smoothed_prediction, smoothed_confidence = self.smooth_predictions(prediction, confidence)


#                 if smoothed_prediction is not None:
#                     gesture_name = self.label_encoder.inverse_transform([smoothed_prediction])[0]
#                     print(gesture_name, smoothed_confidence)
#                 else:
#                     print("Waiting for enough frames...")


                
#                 # Calculate FPS
#                 fps = self.calculate_fps()
                
#                 # Draw UI
#                 frame_with_ui = self.draw_ui(frame, smoothed_prediction, smoothed_confidence, fps)
                
#                 # Display frame
#                 cv2.imshow('ISL Gesture Recognition', frame_with_ui)
                
#                 # Handle key presses
#                 key = cv2.waitKey(1) & 0xFF
                
#                 if key == ord('q'):
#                     print("👋 Quitting...")
#                     break
#                 elif key == ord('r'):
#                     self.frame_buffer.clear()
#                     self.landmark_buffer.clear()
#                     self.prediction_buffer.clear()
#                     print("🔄 Buffers reset")
#                 elif key == ord('s') and save_screenshots:
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"isl_screenshot_{timestamp}.jpg"
#                     cv2.imwrite(filename, frame_with_ui)
#                     print(f"📸 Screenshot saved: {filename}")
#                 elif key == ord('l'):
#                     show_landmarks = not show_landmarks
#                     print(f"👆 Landmarks display: {'ON' if show_landmarks else 'OFF'}")
        
#         except KeyboardInterrupt:
#             print("\n👋 Interrupted by user")
        
#         finally:
#             cap.release()
#             cv2.destroyAllWindows()
#             print("🔚 Camera released and windows closed")

# def main():
#     """Main function with argument parsing"""
#     parser = argparse.ArgumentParser(description='Real-time ISL Gesture Recognition')
#     parser.add_argument('--model_path', type=str, default='best_model.pth',
#                        help='Path to the trained model file')
#     parser.add_argument('--camera_id', type=int, default=0,
#                        help='Camera ID (default: 0)')
#     parser.add_argument('--device', type=str, default=None,
#                        help='Device to use (cuda/mps/cpu, default: auto)')
#     parser.add_argument('--no_landmarks', action='store_true',
#                        help='Disable landmark visualization')
#     parser.add_argument('--save_screenshots', action='store_true',
#                        help='Enable screenshot saving with "s" key')
    
#     args = parser.parse_args()
    
#     # Check if model file exists
#     if not os.path.exists(args.model_path):
#         print(f"❌ Error: Model file '{args.model_path}' not found!")
#         return
    
#     try:
#         # Initialize inference system
#         inference = ISLRealTimeInference(
#             model_path=args.model_path,
#             device=args.device
#         )
        
#         # Run inference
#         inference.run(
#             camera_id=args.camera_id,
#             show_landmarks=not args.no_landmarks,
#             save_screenshots=args.save_screenshots
#         )
        
#     except Exception as e:
#         print(f"❌ Error during inference: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()
























































































import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import pickle
import json
import os
import time
from collections import deque
import threading
from datetime import datetime
import argparse
from pytorchModelBuilder import ISLViTModel 
from config import ISLConfig

class GestureState:
    """Enum for gesture detection states"""
    IDLE = 0
    MOTION_DETECTED = 1
    RECORDING = 2
    PREDICTION = 3

class ISLRealTimeInference:
    """Real-time ISL gesture detection with motion-based triggering"""
    
    def __init__(self, model_path, config_path=None, device=None):
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        # Load model and configuration
        self._load_model()
        
        # Initialize MediaPipe
        self._setup_mediapipe()
        
        # Motion detection parameters - FIXED VALUES
        self.motion_threshold = 0.03  # Reduced threshold for better sensitivity
        self.stillness_threshold = 0.01  # Lower stillness threshold
        self.min_motion_frames = 5  # Reduced required motion frames
        self.stillness_frames_required = 8  # Reduced stillness requirement
        self.max_recording_frames = 60  # Maximum frames to record
        self.min_gesture_frames = 12  # Reduced minimum gesture frames
        
        # State management
        self.gesture_state = GestureState.IDLE
        self.motion_frame_count = 0
        self.stillness_frame_count = 0
        self.recording_frame_count = 0
        
        # Buffers
        self.gesture_frames = []
        self.gesture_landmarks = []
        self.previous_landmarks = None
        
        # Motion history for better detection
        self.motion_history = deque(maxlen=10)
        self.landmark_history = deque(maxlen=5)
        
        # Prediction smoothing
        self.recent_predictions = deque(maxlen=3)
        self.confidence_threshold = 0.6
        self.prediction_cooldown = 0
        self.cooldown_duration = 30
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.2
        self.font_thickness = 2
        
        # Colors (BGR format)
        self.colors = {
            'idle': (128, 128, 128),        # Gray
            'motion': (255, 193, 7),        # Amber
            'recording': (255, 87, 34),     # Deep Orange
            'predicting': (76, 175, 80),    # Green
            'background': (33, 33, 33),     # Dark Gray
            'text': (255, 255, 255),        # White
            'warning': (255, 193, 7),       # Amber
            'error': (244, 67, 54)          # Red
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Results
        self.last_prediction = None
        self.last_confidence = 0.0
        self.prediction_timestamp = None
        
        print(f"✅ ISL Real-time Inference initialized on {self.device}")
        print(f"🎯 Motion threshold: {self.motion_threshold}")
        print(f"📊 Sequence length: {self.config.SEQUENCE_LENGTH}")
    
    def _setup_device(self, device):
        """Setup computation device"""
        if device is not None:
            return torch.device(device)
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_model(self):
        """Load the trained model and configuration"""
        print(f"Loading model from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Reconstruct config from checkpoint or use default
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            class Config:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            self.config = Config(**config_dict)
        else:
            print("Using default config")
            class Config:
                SEQUENCE_LENGTH = 16
                IMG_SIZE = (224, 224)
                EMBED_DIM = 512
                NUM_HEADS = 8
                DROPOUT_RATE = 0.1
                MAX_NUM_HANDS = 2
                MIN_DETECTION_CONFIDENCE = 0.7
                MIN_TRACKING_CONFIDENCE = 0.5
            self.config = Config()
        
        # Load label encoder
        label_encoder_path = os.path.join(os.path.dirname(self.model_path), 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            print("Warning: Label encoder not found. Using default class names.")
            class DummyEncoder:
                def __init__(self):
                    self.classes_ = [f'Gesture_{i}' for i in range(10)]
                def inverse_transform(self, labels):
                    return [self.classes_[i] if i < len(self.classes_) else 'Unknown' for i in labels]
            self.label_encoder = DummyEncoder()
        
        self.num_classes = len(self.label_encoder.classes_)
        
        # Initialize and load model
        self.model = ISLViTModel(self.config, self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded with {self.num_classes} classes")
        print(f"📋 Classes: {self.label_encoder.classes_[:5]}..." if len(self.label_encoder.classes_) > 5 else f"📋 Classes: {self.label_encoder.classes_}")
    
    def _setup_mediapipe(self):
        """Initialize MediaPipe components"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.MAX_NUM_HANDS,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def extract_landmarks(self, frame):
        """Extract hand and pose landmarks from frame - FIXED"""
        landmarks = []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand landmarks
        hand_results = self.hands.process(rgb_frame)
        
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]
        
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if i >= 2:
                    break
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                hand_landmarks_list[i] = np.array(coords)
        
        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])
        
        # Pose landmarks (upper body only)
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            for i in range(11):  # Upper body landmarks
                landmark = pose_results.pose_landmarks.landmark[i]
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            landmarks.extend([0.0] * 44)
        
        # Debug output for troubleshooting
        hand_sum = np.sum(landmarks[:126])
        pose_sum = np.sum(landmarks[126:])
        if hand_sum > 0 or pose_sum > 0:
            print(f"[DEBUG] Landmarks extracted. Hand sum: {hand_sum:.4f}, Pose sum: {pose_sum:.4f}")
        
        return np.array(landmarks), hand_results, pose_results
    
    def calculate_motion(self, current_landmarks):
        """Calculate motion magnitude from landmark changes - IMPROVED"""
        if self.previous_landmarks is None:
            self.previous_landmarks = current_landmarks.copy()
            return 0.0
        
        # Focus on hand landmarks (first 126 values) for motion detection
        hand_landmarks_current = current_landmarks[:126]
        hand_landmarks_previous = self.previous_landmarks[:126]
        
        # Check if we have valid hand landmarks
        current_has_hands = np.sum(np.abs(hand_landmarks_current)) > 0
        previous_has_hands = np.sum(np.abs(hand_landmarks_previous)) > 0
        
        if not current_has_hands and not previous_has_hands:
            motion_magnitude = 0.0
        elif not current_has_hands or not previous_has_hands:
            # Hand appeared or disappeared - significant motion
            motion_magnitude = 0.1
        else:
            # Calculate motion for detected hands
            # Reshape to (num_landmarks, 3) for x,y,z coordinates
            current_points = hand_landmarks_current.reshape(-1, 3)
            previous_points = hand_landmarks_previous.reshape(-1, 3)
            
            # Calculate differences
            differences = current_points - previous_points
            
            # Calculate motion as RMS of all landmark movements
            motion_magnitude = np.sqrt(np.mean(np.sum(differences**2, axis=1)))
            
            # Scale motion for better sensitivity
            motion_magnitude *= 2.0
        
        # Update previous landmarks
        self.previous_landmarks = current_landmarks.copy()
        
        if motion_magnitude > 0.001:  # Only print non-zero motion
            print(f"[DEBUG] Motion magnitude: {motion_magnitude:.5f}")
        
        return motion_magnitude
    
    def update_motion_state(self, motion_magnitude):
        """Update gesture detection state based on motion - IMPROVED"""
        self.motion_history.append(motion_magnitude)
        avg_motion = np.mean(list(self.motion_history))
        
        if self.prediction_cooldown > 0:
            self.prediction_cooldown -= 1
        
        if self.gesture_state == GestureState.IDLE:
            if avg_motion > self.motion_threshold:
                self.motion_frame_count += 1
                print(f"[DEBUG] Motion detected! Count: {self.motion_frame_count}/{self.min_motion_frames}")
                if self.motion_frame_count >= self.min_motion_frames:
                    self.gesture_state = GestureState.MOTION_DETECTED
                    print("🎯 Motion detected! Preparing to record...")
            else:
                if self.motion_frame_count > 0:
                    print(f"[DEBUG] Motion lost, resetting count from {self.motion_frame_count}")
                self.motion_frame_count = 0
                
        elif self.gesture_state == GestureState.MOTION_DETECTED:
            if avg_motion > self.motion_threshold:
                self.gesture_state = GestureState.RECORDING
                self.recording_frame_count = 0
                self.gesture_frames = []
                self.gesture_landmarks = []
                self.stillness_frame_count = 0
                print("🎬 Recording gesture...")
            else:
                # False alarm, go back to idle
                self.gesture_state = GestureState.IDLE
                self.motion_frame_count = 0
                print("[DEBUG] False alarm, back to idle")
                
        elif self.gesture_state == GestureState.RECORDING:
            self.recording_frame_count += 1
            
            if avg_motion < self.stillness_threshold:
                self.stillness_frame_count += 1
                print(f"[DEBUG] Still frame: {self.stillness_frame_count}/{self.stillness_frames_required}")
            else:
                if self.stillness_frame_count > 0:
                    print(f"[DEBUG] Motion resumed, resetting stillness count")
                self.stillness_frame_count = 0
            
            # Check if gesture is complete
            if (self.stillness_frame_count >= self.stillness_frames_required and 
                len(self.gesture_frames) >= self.min_gesture_frames):
                
                if self.prediction_cooldown == 0:
                    self.gesture_state = GestureState.PREDICTION
                    print(f"✅ Gesture recorded! {len(self.gesture_frames)} frames")
                else:
                    print("⏳ Prediction cooldown active, returning to idle...")
                    self._reset_to_idle()
            
            # Prevent infinite recording
            elif self.recording_frame_count >= self.max_recording_frames:
                print("⚠️ Maximum recording length reached, resetting...")
                self._reset_to_idle()
                
        elif self.gesture_state == GestureState.PREDICTION:
            # This state is handled in the main loop
            pass
    
    def _reset_to_idle(self):
        """Reset state machine to idle"""
        print("[DEBUG] Resetting to idle state")
        self.gesture_state = GestureState.IDLE
        self.motion_frame_count = 0
        self.stillness_frame_count = 0
        self.recording_frame_count = 0
        self.gesture_frames = []
        self.gesture_landmarks = []
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        frame_resized = cv2.resize(frame, self.config.IMG_SIZE)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        return frame_normalized
    
    def predict_gesture(self):
        """Predict gesture from recorded sequence"""
        if len(self.gesture_frames) < self.config.SEQUENCE_LENGTH:
            # If we have fewer frames than required, interpolate or repeat
            frames_to_use, landmarks_to_use = self._prepare_sequence(self.gesture_frames, self.gesture_landmarks)
        else:
            # If we have more frames, sample uniformly
            indices = np.linspace(0, len(self.gesture_frames) - 1, self.config.SEQUENCE_LENGTH, dtype=int)
            frames_to_use = [self.gesture_frames[i] for i in indices]
            landmarks_to_use = [self.gesture_landmarks[i] for i in indices]
        
        if len(frames_to_use) != self.config.SEQUENCE_LENGTH:
            print(f"⚠️ Sequence preparation failed: {len(frames_to_use)} vs {self.config.SEQUENCE_LENGTH}")
            return None, 0.0
        
        # Prepare input tensors
        frames = np.stack(frames_to_use)
        landmarks = np.stack(landmarks_to_use)
        
        # Convert to PyTorch tensors
        frames_tensor = torch.from_numpy(frames).float()
        landmarks_tensor = torch.from_numpy(landmarks).float()
        
        # Add batch dimension and permute frames
        frames_tensor = frames_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)
        landmarks_tensor = landmarks_tensor.unsqueeze(0)  # (1, T, 170)
        
        # Move to device
        frames_tensor = frames_tensor.to(self.device)
        landmarks_tensor = landmarks_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(frames_tensor, landmarks_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def _prepare_sequence(self, frames, landmarks):
        """Prepare sequence to match required length"""
        if len(frames) == 0:
            return [], []
        
        if len(frames) < self.config.SEQUENCE_LENGTH:
            # Repeat frames to reach required length
            repeat_factor = self.config.SEQUENCE_LENGTH // len(frames) + 1
            extended_frames = (frames * repeat_factor)[:self.config.SEQUENCE_LENGTH]
            extended_landmarks = (landmarks * repeat_factor)[:self.config.SEQUENCE_LENGTH]
            return extended_frames, extended_landmarks
        else:
            # Sample frames uniformly
            indices = np.linspace(0, len(frames) - 1, self.config.SEQUENCE_LENGTH, dtype=int)
            return [frames[i] for i in indices], [landmarks[i] for i in indices]
    
    def draw_landmarks(self, frame, hand_results, pose_results):
        """Draw MediaPipe landmarks on frame"""
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
    
    def get_state_info(self):
        """Get current state information for UI"""
        motion_avg = np.mean(list(self.motion_history)) if self.motion_history else 0.0
        
        if self.gesture_state == GestureState.IDLE:
            if self.motion_frame_count > 0:
                return f"Detecting motion... ({self.motion_frame_count}/{self.min_motion_frames})", self.colors['motion'], motion_avg
            else:
                return "Ready - Make a gesture", self.colors['idle'], motion_avg
                
        elif self.gesture_state == GestureState.MOTION_DETECTED:
            return "Motion detected! Starting recording...", self.colors['motion'], motion_avg
            
        elif self.gesture_state == GestureState.RECORDING:
            status = f"Recording... ({len(self.gesture_frames)} frames)"
            if self.stillness_frame_count > 0:
                status += f" - Still: {self.stillness_frame_count}/{self.stillness_frames_required}"
            return status, self.colors['recording'], motion_avg
            
        elif self.gesture_state == GestureState.PREDICTION:
            return "Processing gesture...", self.colors['predicting'], motion_avg
        
        return "Unknown state", self.colors['error'], motion_avg
    
    def draw_ui(self, frame, fps):
        """Draw comprehensive UI overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 100), self.colors['background'], -1)
        cv2.rectangle(overlay, (0, 0), (w, 100), self.colors['predicting'], 3)
        
        # Title
        title = "ISL Gesture Recognition - Motion Triggered"
        title_size = cv2.getTextSize(title, self.font, 0.7, 2)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(overlay, title, (title_x, 35), self.font, 0.7, self.colors['text'], 2)
        
        # FPS and frame count
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(overlay, fps_text, (w - 120, 25), self.font, 0.5, self.colors['text'], 1)
        
        # Get state information
        state_text, state_color, motion_magnitude = self.get_state_info()
        
        # State indicator
        cv2.putText(overlay, state_text, (20, 65), self.font, 0.6, state_color, 2)
        
        # Motion indicator bar
        motion_bar_width = 200
        motion_bar_height = 10
        motion_bar_x = 20
        motion_bar_y = 75
        
        cv2.rectangle(overlay, (motion_bar_x, motion_bar_y), 
                     (motion_bar_x + motion_bar_width, motion_bar_y + motion_bar_height), 
                     self.colors['text'], 1)
        
        motion_fill = int(min(motion_magnitude / self.motion_threshold, 1.0) * motion_bar_width)
        if motion_fill > 0:
            cv2.rectangle(overlay, (motion_bar_x, motion_bar_y), 
                         (motion_bar_x + motion_fill, motion_bar_y + motion_bar_height), 
                         state_color, -1)
        
        # Motion value
        cv2.putText(overlay, f"Motion: {motion_magnitude:.3f}", (motion_bar_x + motion_bar_width + 10, motion_bar_y + 8), 
                   self.font, 0.4, self.colors['text'], 1)
        
        # Prediction results
        if (self.last_prediction is not None and 
            self.last_confidence > self.confidence_threshold and 
            self.prediction_timestamp and 
            time.time() - self.prediction_timestamp < 3.0):
            
            gesture_name = self.label_encoder.inverse_transform([self.last_prediction])[0]
            
            # Prediction box
            box_height = 120
            box_y = h - box_height - 20
            cv2.rectangle(overlay, (20, box_y), (w - 20, h - 20), self.colors['background'], -1)
            cv2.rectangle(overlay, (20, box_y), (w - 20, h - 20), self.colors['predicting'], 3)
            
            # Gesture name
            gesture_text = f"Detected: {gesture_name}"
            text_size = cv2.getTextSize(gesture_text, self.font, self.font_scale, self.font_thickness)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay, gesture_text, (text_x, box_y + 45), 
                       self.font, self.font_scale, self.colors['text'], self.font_thickness)
            
            # Confidence
            conf_text = f"Confidence: {self.last_confidence:.1%}"
            conf_size = cv2.getTextSize(conf_text, self.font, 0.7, 2)[0]
            conf_x = (w - conf_size[0]) // 2
            cv2.putText(overlay, conf_text, (conf_x, box_y + 80), 
                       self.font, 0.7, self.colors['text'], 2)
            
            # Confidence bar
            bar_width = 300
            bar_height = 15
            bar_x = (w - bar_width) // 2
            bar_y = box_y + 90
            
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         self.colors['text'], 1)
            filled_width = int(bar_width * self.last_confidence)
            bar_color = self.colors['predicting'] if self.last_confidence > 0.8 else self.colors['warning']
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                         bar_color, -1)
        
        # Instructions
        instructions = [
            "Make a gesture and hold still to trigger detection",
            "Press 'q' to quit | 'r' to reset | 's' for screenshot | 'm' for manual trigger"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(overlay, instruction, (10, h - 40 + i * 20), 
                       self.font, 0.4, self.colors['text'], 1)
        
        # Cooldown indicator
        if self.prediction_cooldown > 0:
            cooldown_text = f"Cooldown: {self.prediction_cooldown}"
            cv2.putText(overlay, cooldown_text, (w - 150, h - 40), 
                       self.font, 0.5, self.colors['warning'], 1)
        
        # Blend overlay
        alpha = 0.85
        frame_with_ui = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame_with_ui
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
        return self.current_fps
    
    def run(self, camera_id=0, show_landmarks=True):
        """Main inference loop with motion-triggered prediction - SIMPLIFIED"""
        print(f"🎥 Starting camera feed (Camera ID: {camera_id})...")
        print("📋 Controls:")
        print("   'q' - Quit")
        print("   'r' - Reset state") 
        print("   's' - Save screenshot")
        print("   'l' - Toggle landmarks")
        print("   'm' - Manual trigger recording")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("✅ Camera opened successfully!")
        print("🚀 Starting motion-triggered inference...")
        print(f"🎯 Using fixed motion threshold: {self.motion_threshold}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks, hand_results, pose_results = self.extract_landmarks(frame)
                
                # Calculate motion and update state
                motion_magnitude = self.calculate_motion(landmarks)
                self.update_motion_state(motion_magnitude)
                
                # Handle recording state
                if self.gesture_state == GestureState.RECORDING:
                    processed_frame = self.preprocess_frame(frame)
                    self.gesture_frames.append(processed_frame)
                    self.gesture_landmarks.append(landmarks)
                
                # Handle prediction state
                elif self.gesture_state == GestureState.PREDICTION:
                    print("🔮 Making prediction...")
                    prediction, confidence = self.predict_gesture()
                    
                    if prediction is not None and confidence > self.confidence_threshold:
                        gesture_name = self.label_encoder.inverse_transform([prediction])[0]
                        print(f"✅ Prediction: {gesture_name} (confidence: {confidence:.2%})")
                        
                        self.last_prediction = prediction
                        self.last_confidence = confidence
                        self.prediction_timestamp = time.time()
                        self.prediction_cooldown = self.cooldown_duration
                    else:
                        gesture_name = self.label_encoder.inverse_transform([prediction])[0]
                        print(f"❌ Low confidence prediction: {confidence:.2%}")
                        print(f"✅ Prediction: {gesture_name} (confidence: {confidence:.2%})")
                    
                    # Reset to idle
                    self._reset_to_idle()
                
                # Draw landmarks if enabled
                if show_landmarks:
                    self.draw_landmarks(frame, hand_results, pose_results)
                
                # Calculate FPS and draw UI
                fps = self.calculate_fps()
                frame_with_ui = self.draw_ui(frame, fps)
                
                # Display frame
                cv2.imshow('ISL Motion-Triggered Recognition', frame_with_ui)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 Quitting...")
                    break
                elif key == ord('r'):
                    self._reset_to_idle()
                    self.last_prediction = None
                    print("🔄 State reset")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"isl_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_with_ui)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('l'):
                    show_landmarks = not show_landmarks
                    print(f"👆 Landmarks display: {'ON' if show_landmarks else 'OFF'}")
                elif key == ord('m'):
                    # Manual trigger for testing
                    if self.gesture_state == GestureState.IDLE:
                        print("🎯 Manual trigger - starting recording...")
                        self.gesture_state = GestureState.RECORDING
                        self.recording_frame_count = 0
                        self.gesture_frames = []
                        self.gesture_landmarks = []
                        self.stillness_frame_count = 0
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🔚 Camera released and windows closed")

def main():
    """Main function to run ISL real-time inference with command line arguments"""
    parser = argparse.ArgumentParser(description='ISL Real-time Gesture Recognition')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file (.pth)')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use for inference (auto-detect if not specified)')
    parser.add_argument('--no-landmarks', action='store_true',
                        help='Disable landmark visualization')
    parser.add_argument('--motion-threshold', type=float, default=0.03,
                        help='Motion detection threshold (default: 0.03)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                        help='Minimum confidence for predictions (default: 0.6)')
    parser.add_argument('--cooldown', type=int, default=30,
                        help='Cooldown frames between predictions (default: 30)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Validate config path if provided
    if args.config and not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        return
    
    print("🚀 Initializing ISL Real-time Inference...")
    print(f"📁 Model: {args.model}")
    print(f"📷 Camera ID: {args.camera}")
    print(f"💻 Device: {args.device if args.device else 'auto-detect'}")
    print(f"🎯 Motion threshold: {args.motion_threshold}")
    print(f"📊 Confidence threshold: {args.confidence_threshold}")
    print(f"⏰ Cooldown: {args.cooldown} frames")
    print("-" * 50)
    
    try:
        # Initialize the inference system
        isl_inference = ISLRealTimeInference(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
        
        # Override thresholds if specified
        if args.motion_threshold != 0.03:
            isl_inference.motion_threshold = args.motion_threshold
            print(f"🎯 Motion threshold updated to: {args.motion_threshold}")
        
        if args.confidence_threshold != 0.6:
            isl_inference.confidence_threshold = args.confidence_threshold
            print(f"📊 Confidence threshold updated to: {args.confidence_threshold}")
        
        if args.cooldown != 30:
            isl_inference.cooldown_duration = args.cooldown
            print(f"⏰ Cooldown updated to: {args.cooldown} frames")
        
        # Start the inference loop
        isl_inference.run(
            camera_id=args.camera,
            show_landmarks=not args.no_landmarks
        )
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        print("💡 Please check your model file and dependencies")
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    print("✅ Program finished")

if __name__ == "__main__":
    main()