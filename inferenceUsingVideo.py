import torch
import cv2
import numpy as np
import mediapipe as mp
import pickle
import argparse
import os
from collections import deque

# Import necessary classes from your existing project files
from config import ISLConfig
from pytorchModelBuilder import ISLViTModel # Assumes ISLViTModel is in this file

# ====================================================================================
# 1. INFERENCE ENGINE
# ====================================================================================

class InferenceEngine:
    """Handles the full inference pipeline by processing video and using the trained model."""

    def __init__(self, config, checkpoint_path, label_encoder_path, device):
        self.config = config
        self.device = device

        # Load the label encoder
        print(f"🔄 Loading label encoder from {label_encoder_path}...")
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Label encoder not found at '{label_encoder_path}'. Please check the path.")
            exit()
        num_classes = len(self.label_encoder.classes_)
        print(f"✅ Found {num_classes} classes.")

        # Build and load the model
        print("🛠️  Building model architecture...")
        self.model = ISLViTModel(config, num_classes=num_classes)
        
        print(f"🔄 Loading model checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print(f"❌ ERROR: Model checkpoint not found at '{checkpoint_path}'. Please check the path.")
            exit()
        except KeyError:
            print("❌ ERROR: The checkpoint file seems to be corrupt or missing 'model_state_dict'.")
            exit()
            
        self.model.to(device)
        self.model.eval()
        print("✅ Model loaded successfully and set to evaluation mode.")

        # Initialize MediaPipe for landmark extraction
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=config.MAX_NUM_HANDS, 
            min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Data buffers to hold the sequence of frames and landmarks
        self.frame_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.landmark_buffer = deque(maxlen=config.SEQUENCE_LENGTH)

    def extract_landmarks(self, frame):
        """Extracts and draws hand and pose landmarks from a single frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = []
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        hand_landmarks_list = [np.zeros(63), np.zeros(63)] # 21 landmarks * 3 coords = 63
        if hand_results.multi_hand_landmarks:
            for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                if i >= self.config.MAX_NUM_HANDS: break
                hand_landmarks_list[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]).flatten()
                self.mp_drawing.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])
        
        # Process pose (upper body)
        pose_results = self.pose.process(rgb_frame)
        pose_lm_list = np.zeros(44) # 11 landmarks * 4 coords (x,y,z,vis) = 44
        if pose_results.pose_landmarks:
            pose_lm_list = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_results.pose_landmarks.landmark[:11]]).flatten()
            self.mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        landmarks.extend(pose_lm_list)
        
        return np.array(landmarks)

    def normalize_landmarks(self, landmarks):
        """Normalizes a sequence of landmarks to be translation and scale invariant."""
        landmarks_norm = landmarks.copy()
        for t in range(landmarks_norm.shape[0]):
            frame_landmarks_flat = landmarks_norm[t]
            hand1 = frame_landmarks_flat[:63].reshape(21, 3)
            hand2 = frame_landmarks_flat[63:126].reshape(21, 3)
            pose = frame_landmarks_flat[126:].reshape(11, 4)
            all_xyz = np.concatenate([hand1, hand2, pose[:, :3]], axis=0) # Shape (53, 3)
            
            valid_mask = np.any(all_xyz != 0, axis=1)
            if np.any(valid_mask):
                valid_coords = all_xyz[valid_mask]
                mean_pos = np.mean(valid_coords[:, :2], axis=0)
                all_xyz[:, :2] -= mean_pos
                
                max_dist = np.max(np.linalg.norm(all_xyz[valid_mask, :2], axis=1))
                if max_dist > 1e-6:
                    all_xyz[:, :2] /= max_dist

            pose[:, :3] = all_xyz[42:]
            landmarks_norm[t] = np.concatenate([all_xyz[:21].flatten(), all_xyz[21:42].flatten(), pose.flatten()])
        return landmarks_norm

    def predict(self):
        """Performs a prediction and returns the top 5 results if the buffer is full."""
        if len(self.frame_buffer) < self.config.SEQUENCE_LENGTH:
            return None

        # Prepare data from buffers
        frames = np.array(self.frame_buffer)
        landmarks = np.array(self.landmark_buffer)
        
        landmarks_normalized = self.normalize_landmarks(landmarks)

        # Convert to tensors and move to the correct device
        frames_tensor = torch.from_numpy(frames).float().to(self.device).permute(0, 3, 1, 2)
        landmarks_tensor = torch.from_numpy(landmarks_normalized).float().to(self.device)

        # Add batch dimension for model input
        frames_tensor = frames_tensor.unsqueeze(0)
        landmarks_tensor = landmarks_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            logits = self.model(frames_tensor, landmarks_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # ⭐ Get top 5 predictions using torch.topk
            top5_conf, top5_idx = torch.topk(probabilities, 5, dim=1)
        
        # Decode the top 5 indices to labels
        indices = top5_idx.cpu().numpy().flatten()
        confidences = top5_conf.cpu().numpy().flatten()
        labels = self.label_encoder.inverse_transform(indices)

        # Combine into a list of (label, confidence) tuples
        top_predictions = list(zip(labels, confidences))
        
        return top_predictions

    def process_and_predict(self, frame):
        """High-level method to process a frame, update buffers, and get a prediction."""
        # 1. Preprocess frame for model input
        frame_resized = cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # 2. Extract landmarks and draw on the original frame for display
        landmarks = self.extract_landmarks(frame)
        
        # 3. Update internal buffers
        self.frame_buffer.append(frame_normalized)
        self.landmark_buffer.append(landmarks)
        
        # 4. Get prediction from the model
        return self.predict()

# ====================================================================================
# 2. MAIN EXECUTION BLOCK
# ====================================================================================

def main(args):
    """Main function to set up and run the inference loop."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Initialize components
    config = ISLConfig()
    engine = InferenceEngine(config, args.checkpoint, args.label_encoder, device)

    # Setup video capture
    video_source = args.video_path if args.video_path else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{video_source}'.")
        return

    # Initialize a placeholder for the top predictions
    top_predictions = [("Initializing...", 0.0)] * 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⏹️  End of video or camera feed.")
            break

        display_frame = frame.copy()
        
        # Process the frame and get predictions
        predictions = engine.process_and_predict(display_frame)
        
        # Update the list of predictions if the model returns a new set
        if predictions is not None:
            top_predictions = predictions

        # ⭐ Display the top 5 predictions on the frame
        # Make the background rectangle larger to fit 5 lines
        cv2.rectangle(display_frame, (0, 0), (700, 170), (0, 0, 0), -1) 
        
        for i, (label, conf) in enumerate(top_predictions):
            text = f"{i+1}. {label} ({conf:.2f})"
            y_pos = 30 + i * 30 # Calculate y position for each line
            cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Indian Sign Language Inference', display_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Application closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time Indian Sign Language Inference")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to the model checkpoint file.')
    parser.add_argument('--label_encoder', type=str, default='chunked_data/label_encoder.pkl', help='Path to the label encoder pickle file.')
    parser.add_argument('--video_path', type=str, default=None, help='Path to a video file. If not provided, uses the webcam.')
    
    args = parser.parse_args()
    main(args)
