import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import mediapipe as mp
from pytorchModelBuilder import ISLViTModel
from config import ISLConfig

class InferencePreprocessor:
    """Preprocessor for inference that mimics DataPreprocessor's landmark extraction and frame processing"""

    def __init__(self, config: ISLConfig):
        self.config = config
        
        # MediaPipe Hands and Pose models
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )

    def extract_landmarks(self, frame):
        landmarks = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(rgb_frame)
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]  # 2 hands, 21 landmarks * 3 coords = 63

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

        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            for i in range(11):  # upper body only
                lm = pose_results.pose_landmarks.landmark[i]
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 44)  # 11 landmarks * 4 coords

        return np.array(landmarks)

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        landmarks_sequence = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > self.config.SEQUENCE_LENGTH:
            frame_indices = np.linspace(0, total_frames - 1, self.config.SEQUENCE_LENGTH, dtype=int)
        else:
            frame_indices = list(range(total_frames))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_indices:
                frame_resized = cv2.resize(frame, self.config.IMG_SIZE)
                frames.append(frame_resized)

                landmarks = self.extract_landmarks(frame)
                landmarks_sequence.append(landmarks)

            frame_idx += 1

        cap.release()

        # Pad sequences if needed
        while len(frames) < self.config.SEQUENCE_LENGTH:
            if frames:
                frames.append(frames[-1])
                landmarks_sequence.append(landmarks_sequence[-1])
            else:
                blank_frame = np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)
                frames.append(blank_frame)
                landmarks_sequence.append(np.zeros(170))  # 63 + 63 + 44

        return np.array(frames[:self.config.SEQUENCE_LENGTH]), np.array(landmarks_sequence[:self.config.SEQUENCE_LENGTH])

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        config = Config(**config_dict)
    else:
        config = ISLConfig()

    label_path = os.path.join(os.path.dirname(model_path), "label_encoder.pkl")
    if os.path.exists(label_path):
        with open(label_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        class DummyEncoder:
            def __init__(self):
                self.classes_ = [f'Gesture_{i}' for i in range(10)]
            def inverse_transform(self, labels):
                return [self.classes_[i] if i < len(self.classes_) else 'Unknown' for i in labels]
        label_encoder = DummyEncoder()

    num_classes = len(label_encoder.classes_)
    model = ISLViTModel(config, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config, label_encoder

def infer_from_video(video_path, model_path, device="mps" if torch.backends.mps.is_available() else "cpu"):
    device = torch.device(device)
    model, config, label_encoder = load_model(model_path, device)

    preprocessor = InferencePreprocessor(config)

    print("🔁 Preprocessing video frames and extracting landmarks...")
    frames, landmarks = preprocessor.preprocess_video(video_path)

    # Normalize frames as in training
    frames = frames.astype(np.float32) / 255.0

    # Convert to tensors and permute
    video_tensor = torch.from_numpy(frames).unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
    landmark_tensor = torch.from_numpy(landmarks.astype(np.float32)).unsqueeze(0).to(device)


    with torch.no_grad():
        logits = model(video_tensor, landmark_tensor)
        probs = F.softmax(logits, dim=1)

        top5_confidence, top5_pred = torch.topk(probs, 5, dim=1)
        top5_confidence = top5_confidence.squeeze(0).cpu().numpy()
        top5_pred = top5_pred.squeeze(0).cpu().numpy()

        top5_labels = label_encoder.inverse_transform(top5_pred.tolist())

    print("✅ Top 5 Predictions:")
    for i, (label, conf) in enumerate(zip(top5_labels, top5_confidence), 1):
        print(f"  {i}. {label} - Confidence: {conf:.2%}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict gesture from video file")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device to run on")

    args = parser.parse_args()

    device_to_use = args.device if args.device else ("mps" if torch.backends.mps.is_available() else "cpu")
    infer_from_video(args.video, args.model, device=device_to_use)
