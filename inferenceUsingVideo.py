import torch
import cv2
import numpy as np
import mediapipe as mp
import pickle
import argparse
import os
from collections import deque

# Import necessary classes from your project
from config import ISLConfig
from pytorchModelBuilder import ISLViTModel
from label_encoder import CustomLabelEncoder

# ====================================================================================
# 1. INFERENCE ENGINE
# ====================================================================================

class InferenceEngine:
    """Handles video-based inference by processing full gesture video and predicting a class."""

    def __init__(self, config, checkpoint_path, label_encoder_path, device):
        self.config = config
        self.device = device

        # Load the label encoder
        print(f"🔄 Loading label encoder from {label_encoder_path}...")
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Label encoder not found at '{label_encoder_path}'.")
            exit()
        num_classes = len(self.label_encoder.classes_)
        print(f"✅ Found {num_classes} classes.")

        # Build and load the model
        print("🛠️  Building model architecture...")
        self.model = ISLViTModel(config, num_classes=num_classes)
        print(f"🔄 Loading model checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print(f"❌ ERROR: Model checkpoint not found at '{checkpoint_path}'.")
            exit()
        except KeyError:
            print("❌ ERROR: The checkpoint is missing 'model_state_dict'.")
            exit()

        self.model.to(device)
        self.model.eval()
        print("✅ Model loaded and set to evaluation mode.")

        # MediaPipe for landmark extraction
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

        self.frame_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.landmark_buffer = deque(maxlen=config.SEQUENCE_LENGTH)

    def extract_landmarks(self, frame):
        """Extracts hand and pose landmarks from a single frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = []

        # Process hands
        hand_results = self.hands.process(rgb_frame)
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]
        if hand_results.multi_hand_landmarks:
            for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                if i >= self.config.MAX_NUM_HANDS:
                    break
                hand_landmarks_list[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]).flatten()
        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])

        # Process pose (first 11 upper-body keypoints)
        pose_results = self.pose.process(rgb_frame)
        pose_lm_list = np.zeros(44)
        if pose_results.pose_landmarks:
            pose_lm_list = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_results.pose_landmarks.landmark[:11]]).flatten()
        landmarks.extend(pose_lm_list)

        return np.array(landmarks)

    def normalize_landmarks(self, landmarks):
        """Normalizes landmarks to be translation and scale invariant."""
        landmarks_norm = landmarks.copy()
        for t in range(landmarks.shape[0]):
            frame_landmarks = landmarks[t]
            hand1 = frame_landmarks[:63].reshape(21, 3)
            hand2 = frame_landmarks[63:126].reshape(21, 3)
            pose = frame_landmarks[126:].reshape(11, 4)

            all_xyz = np.concatenate([hand1, hand2, pose[:, :3]], axis=0)
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
        """Predicts class from buffered frames and landmarks."""
        if len(self.frame_buffer) < self.config.SEQUENCE_LENGTH:
            return None, None

        frames = np.array(self.frame_buffer)
        landmarks = np.array(self.landmark_buffer)
        landmarks_normalized = self.normalize_landmarks(landmarks)

        frames_tensor = torch.from_numpy(frames).float().to(self.device).permute(0, 3, 1, 2)
        landmarks_tensor = torch.from_numpy(landmarks_normalized).float().to(self.device)

        frames_tensor = frames_tensor.unsqueeze(0)
        landmarks_tensor = landmarks_tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(frames_tensor, landmarks_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)

        pred_label = self.label_encoder.inverse_transform([pred_idx.item()])[0]
        return pred_label, confidence.item()

    def process_video(self, video_path):
        """Reads and processes the entire gesture video for prediction."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file '{video_path}'.")
            return None, None

        self.frame_buffer.clear()
        self.landmark_buffer.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            landmarks = self.extract_landmarks(frame)

            self.frame_buffer.append(frame_normalized)
            self.landmark_buffer.append(landmarks)

        cap.release()

        if len(self.frame_buffer) < self.config.SEQUENCE_LENGTH:
            print(f"⚠️ Not enough frames ({len(self.frame_buffer)}) for sequence length = {self.config.SEQUENCE_LENGTH}")
            return None, None

        return self.predict()

# ====================================================================================
# 2. MAIN EXECUTION BLOCK
# ====================================================================================

def main(args):
    """Main function to predict a gesture from a video file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    config = ISLConfig()
    engine = InferenceEngine(config, args.checkpoint, args.label_encoder, device)

    if args.video_path is None:
        print("❌ Please provide a video file using --video_path.")
        return

    pred_label, confidence = engine.process_video(args.video_path)

    if pred_label:
        print(f"\n🧠 Final Prediction: {pred_label} (Confidence: {confidence:.2f})\n")
    else:
        print("❌ Could not predict from the given video.")

# ====================================================================================
# 3. ENTRY POINT
# ====================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict a single gesture from a video using ISLViTModel")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to model checkpoint.')
    parser.add_argument('--label_encoder', type=str, default='processed_data/chunked/label_encoder.pkl', help='Path to label encoder.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the gesture video.')

    args = parser.parse_args()
    main(args)
