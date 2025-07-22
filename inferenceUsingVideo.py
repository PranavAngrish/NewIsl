import torch
import cv2
import numpy as np
import mediapipe as mp
import pickle
import argparse
import os

# Assuming your ISLConfig and ISLViTModel classes are importable
# If they are in files named 'config.py' and 'pytorchModelBuilder.py':
from config import ISLConfig
from pytorchModelBuilder import ISLViTModel

# =============================================================================
# 1. INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Handles video-based inference by replicating the exact preprocessing
    pipeline from training to ensure model accuracy.
    """

    def __init__(self, config, checkpoint_path, label_encoder_path, device):
        """
        Initializes the model, preprocessors, and loads necessary assets.
        """
        self.config = config
        self.device = device

        # --- Load Label Encoder ---
        print(f"🔄 Loading label encoder from {label_encoder_path}...")
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Label encoder not found at '{label_encoder_path}'.")
            exit()
        num_classes = len(self.label_encoder.classes_)
        print(f"✅ Found {num_classes} classes.")

        # --- Build and Load Model ---
        print("🛠️  Building model architecture...")
        self.model = ISLViTModel(config, num_classes=num_classes)
        print(f"🔄 Loading model checkpoint from {checkpoint_path}...")
        try:
            # Use map_location to load model on the correct device
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print(f"❌ ERROR: Model checkpoint not found at '{checkpoint_path}'.")
            exit()
        except KeyError:
            print("❌ ERROR: The checkpoint is missing 'model_state_dict'. It might be an old format.")
            exit()

        self.model.to(device)
        self.model.eval()
        print("✅ Model loaded and set to evaluation mode.")

        # --- Initialize MediaPipe ---
        # These are the same settings used in the training preprocessor
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # -------------------------------------------------------------------------
    # PREPROCESSING FUNCTIONS (Mirrored from Training Script)
    # -------------------------------------------------------------------------

    def preprocess_frame(self, frame):
        """
        Applies histogram equalization and blur.
        Identical to the training preprocessor.
        """
        if len(frame.shape) == 3:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return cv2.GaussianBlur(frame, (3, 3), 0)

    def extract_landmarks(self, frame):
        """
        Extracts hand and pose landmarks from a single frame using MediaPipe.
        Includes the same fallback logic as the training preprocessor.
        """
        landmarks = []
        # Preprocess frame before sending to MediaPipe
        rgb_frame = cv2.cvtColor(self.preprocess_frame(frame), cv2.COLOR_BGR2RGB)

        # --- Hand Landmark Extraction ---
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]
        hand_results = self.hands.process(rgb_frame)
        # Fallback with increased contrast if no hands are found
        if not hand_results.multi_hand_landmarks:
            enhanced_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=20)
            hand_results = self.hands.process(enhanced_frame)

        if hand_results.multi_hand_landmarks:
            for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                if i >= self.config.MAX_NUM_HANDS:
                    break
                hand_landmarks_list[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]).flatten()
        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])

        # --- Pose Landmark Extraction ---
        pose_lm_list = np.zeros(44)
        pose_results = self.pose.process(rgb_frame)
        # Fallback with increased contrast if no pose is found
        if not pose_results.pose_landmarks:
            enhanced_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.1, beta=15)
            pose_results = self.pose.process(enhanced_frame)

        if pose_results.pose_landmarks:
            # Extract the first 11 upper-body keypoints
            pose_lm_list = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_results.pose_landmarks.landmark[:11]]).flatten()
        landmarks.extend(pose_lm_list)

        return np.array(landmarks)

    def normalize_landmarks(self, landmarks):
        """
        Normalizes landmarks to be translation, scale, and rotation invariant.
        This is the corrected version from the training Dataset, including Z-standardization.
        """
        landmarks = landmarks.copy() # Shape (T, 170)
        for t in range(landmarks.shape[0]): # For each frame
            frame_landmarks_flat = landmarks[t]

            # Deconstruct the flat array into structured landmarks
            hand1 = frame_landmarks_flat[:63].reshape(21, 3)
            hand2 = frame_landmarks_flat[63:126].reshape(21, 3)
            pose = frame_landmarks_flat[126:].reshape(11, 4)

            # Combine all (x,y,z) coordinates for normalization
            all_xyz = np.concatenate([hand1, hand2, pose[:, :3]], axis=0)

            # Find valid landmarks (non-zero) to base normalization on
            valid_mask = np.any(all_xyz != 0, axis=1)
            
            if np.any(valid_mask):
                valid_coords = all_xyz[valid_mask]
                
                # Center normalization for x,y coordinates
                mean_pos = np.mean(valid_coords[:, :2], axis=0)
                all_xyz[:, :2] -= mean_pos
                
                # Scale normalization for x,y coordinates
                distances = np.linalg.norm(all_xyz[valid_mask, :2], axis=1)
                max_dist = np.max(distances)
                if max_dist > 1e-6: # Avoid division by zero
                    all_xyz[:, :2] /= max_dist
                
                # Z-coordinate normalization (standardization)
                z_coords = all_xyz[valid_mask, 2]
                if len(z_coords) > 1:
                    z_mean = np.mean(z_coords)
                    z_std = np.std(z_coords)
                    if z_std > 1e-6: # Avoid division by zero
                        all_xyz[:, 2] = (all_xyz[:, 2] - z_mean) / z_std
            
            # Put the normalized coordinates back into their original structures
            pose[:, :3] = all_xyz[42:]

            # Reconstruct the flat 170-dimensional array
            landmarks[t] = np.concatenate([
                all_xyz[:21].flatten(),
                all_xyz[21:42].flatten(),
                pose.flatten()
            ])

        return landmarks

    def process_video_for_inference(self, video_path):
        """
        Reads a video, samples frames, and extracts landmarks.
        This function now mirrors the `preprocess_video` logic from training.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file '{video_path}'.")
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- Frame Sampling Logic (Identical to Training) ---
        if total_frames > self.config.SEQUENCE_LENGTH:
            seg_len = self.config.SEQUENCE_LENGTH // 3
            rem = self.config.SEQUENCE_LENGTH % 3
            begin_indices = np.linspace(0, total_frames // 3 - 1, seg_len, dtype=int)
            middle_indices = np.linspace(total_frames // 3, 2 * total_frames // 3 - 1, seg_len, dtype=int)
            end_indices = np.linspace(2 * total_frames // 3, total_frames - 1, seg_len + rem, dtype=int)
            frame_indices = np.unique(np.concatenate([begin_indices, middle_indices, end_indices]))
        else:
            frame_indices = np.arange(total_frames)

        frames, landmarks_sequence = [], []
        frame_idx, extracted_count = 0, 0
        while cap.isOpened() and extracted_count < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break
            # Process only the frames at the sampled indices
            if frame_idx == frame_indices[extracted_count]:
                # Resize frame to the model's expected input size
                resized_frame = cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
                frames.append(resized_frame)
                landmarks_sequence.append(self.extract_landmarks(frame))
                extracted_count += 1
            frame_idx += 1
        cap.release()

        # --- Padding Logic (Identical to Training) ---
        current_length = len(frames)
        target_length = self.config.SEQUENCE_LENGTH
        if 0 < current_length < target_length:
            pad_needed = target_length - current_length
            frames.extend([frames[-1]] * pad_needed)
            landmarks_sequence.extend([landmarks_sequence[-1]] * pad_needed)
        elif current_length == 0:
            print(f"⚠️ Warning: Could not read any frames from {video_path}.")
            return None, None
            
        # Convert to numpy arrays and normalize frames
        frames_np = np.array(frames[:target_length], dtype=np.float32) / 255.0
        landmarks_np = np.array(landmarks_sequence[:target_length], dtype=np.float32)
        
        return frames_np, landmarks_np

    # -------------------------------------------------------------------------
    # MAIN PREDICTION FUNCTION
    # -------------------------------------------------------------------------

    def predict(self, video_path):
        """
        Processes a video file and returns the model's prediction.
        """
        # 1. Process video to get frames and landmarks
        frames, landmarks = self.process_video_for_inference(video_path)
        if frames is None or landmarks is None:
            return None, None

        # 2. Normalize landmarks
        landmarks_normalized = self.normalize_landmarks(landmarks)

        # 3. Convert to PyTorch Tensors
        frames_tensor = torch.from_numpy(frames).float().to(self.device)
        landmarks_tensor = torch.from_numpy(landmarks_normalized).float().to(self.device)

        # Reshape frames tensor for the model: (T, H, W, C) -> (T, C, H, W)
        if frames_tensor.dim() == 4 and frames_tensor.shape[-1] == 3:
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)

        # 4. Add batch dimension: (T, ...) -> (1, T, ...)
        frames_tensor = frames_tensor.unsqueeze(0)
        landmarks_tensor = landmarks_tensor.unsqueeze(0)

        # 5. Perform Inference
        with torch.no_grad():
            logits = self.model(frames_tensor, landmarks_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)

        # 6. Decode the prediction
        pred_label = self.label_encoder.inverse_transform([pred_idx.item()])[0]
        
        return pred_label, confidence.item()


# =============================================================================
# 2. MAIN EXECUTION BLOCK
# =============================================================================

def main(args):
    """Main function to predict a gesture from a video file."""
    # Determine the device to run the model on
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🚀 Using device: {device}")

    # Initialize the configuration and the inference engine
    config = ISLConfig()
    engine = InferenceEngine(config, args.checkpoint, args.label_encoder, device)

    # Check if the video path is valid
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found at '{args.video_path}'")
        return

    # Get the prediction
    print(f"\n▶️  Processing video: {args.video_path}")
    pred_label, confidence = engine.predict(args.video_path)

    # Print the result
    if pred_label:
        print(f"\n✅ Final Prediction: '{pred_label}' (Confidence: {confidence:.2f})\n")
    else:
        print("\n❌ Could not make a prediction from the given video.")

# =============================================================================
# 3. ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict a single gesture from a video using ISLViTModel.")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to the trained model checkpoint (.pth file).'
    )
    parser.add_-argument(
        '--label_encoder',
        type=str,
        default='processed_data/chunked/label_encoder.pkl',
        help='Path to the saved LabelEncoder (.pkl file).'
    )
    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='Path to the gesture video file for inference.'
    )

    args = parser.parse_args()
    main(args)
