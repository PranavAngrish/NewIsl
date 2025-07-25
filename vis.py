import os
import cv2
import numpy as np
import mediapipe as mp
import json
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from config import ISLConfig
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from scipy import ndimage
from scipy.spatial.distance import cdist
import math
from scipy.ndimage import gaussian_filter1d

class EnhancedVideoDataset(Dataset):
    """Enhanced PyTorch Dataset with robust augmentation for video data - Fixed for multiprocessing"""
    
    def __init__(self, manifest, config, mode='train', transform=None):
        self.manifest = manifest
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # DON'T initialize MediaPipe here - do it lazily in __getitem__
        self._mp_selfie_segmentation = None
        self._selfie_segmentation = None

    
    @property
    def selfie_segmentation(self):
        """Lazy initialization of MediaPipe selfie segmentation"""
        if self._selfie_segmentation is None:
            self._mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self._selfie_segmentation = self._mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        return self._selfie_segmentation
    
    def __len__(self):
        return len(self.manifest)

    def generate_random_background(self, height, width):
        """Generates a random background image."""
        if random.random() < 0.5:
            return np.full((height, width, 3), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), dtype=np.uint8)
        else:
            return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    def replace_background(self, frame):
        """Replaces the background of a frame using selfie segmentation."""
        H, W, _ = frame.shape
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(rgb_frame)
        frame.flags.writeable = True
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = self.generate_random_background(H, W)
        output_image = np.where(condition, frame, bg_image)
        return output_image
    
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        
        try:
            # Load frames and landmarks from disk
            frames = np.load(entry["frame_path"])         # Expected shape: (T, H, W, 3)
            landmarks = np.load(entry["landmarks_path"])  # Expected shape: (T, 154)
            label = entry["encoded_label"]
        
            # Convert (T, H, W, 3) to (T, 3, H, W)
            if frames.ndim == 4 and frames.shape[-1] == 3:
                frames = frames.transpose(0, 3, 1, 2)  # NumPy: permute dims
            
            # Convert to tensors
            frames = torch.from_numpy(frames).float()
            landmarks = torch.from_numpy(landmarks).float()
            label = torch.tensor(label, dtype=torch.long)
            
            return (frames, landmarks), label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            T, H, W, C = self.config.SEQUENCE_LENGTH, 224, 224, 3
            return (torch.zeros(T, C, H, W), torch.zeros(T, 154)), torch.tensor(0, dtype=torch.long)


class EnhancedDataPreprocessor:
    """Enhanced data preprocessing module with better generalization techniques"""
    
    def __init__(self, config: ISLConfig, mode='train', motion_threshold=0.02, min_gesture_length=3):
        self.config = config
        self.config.create_directories()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=config.MAX_NUM_HANDS, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.label_encoder = LabelEncoder()
        self.mode = mode
        self.motion_threshold = motion_threshold
        self.min_gesture_length = min_gesture_length
        
        # Augmentation parameters for creating new video samples
        self.augmentation_factor = 2  # Create 2 additional videos per original video
        self.augmentation_params = {
            'rotation_range': (-10, 10),      # Reduced for better stability
            'shear_range': (-0.08, 0.08),     # Reduced for natural motion
            'noise_scale': 0.008,             # Reduced noise
            'scale_range': (0.97, 1.03),      # Subtle scaling
            'brightness_range': (0.9, 1.1),   # Frame brightness variation
            'contrast_range': (0.9, 1.1),     # Frame contrast variation
        }
        
    def create_augmented_landmarks_and_frames(self, landmarks, frames, aug_id=0):
        """
        Create augmented landmarks and corresponding frames from original data.
        This ensures temporal consistency across the entire video sequence.
        
        Args:
            landmarks: Original landmarks array (T, 154)
            frames: Original frames array (T, H, W, 3)
            aug_id: Augmentation ID for reproducible results
            
        Returns:
            augmented_landmarks, augmented_frames
        """
        # Set seed for consistent augmentation across frames in the same video
        np.random.seed(aug_id * 42)
        random.seed(aug_id * 42)
        
        augmented_landmarks = landmarks.copy()
        augmented_frames = frames.copy()
        
        T = landmarks.shape[0]
        
        # Deconstruct landmarks for geometric transformations
        hand1_landmarks = augmented_landmarks[:, :63].reshape(T, 21, 3)
        hand2_landmarks = augmented_landmarks[:, 63:126].reshape(T, 21, 3)
        pose_landmarks = augmented_landmarks[:, 126:].reshape(T, 7, 4)
        
        # Generate consistent augmentation parameters for the entire video
        # 1. Rotation (applied consistently across all frames)
        angle = random.uniform(*self.augmentation_params['rotation_range'])
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        # 2. Shearing (applied consistently)
        shear_factor = random.uniform(*self.augmentation_params['shear_range'])
        shear_matrix = np.array([[1, shear_factor], [0, 1]])
        
        # 3. Scaling (applied consistently)
        scale_factor = random.uniform(*self.augmentation_params['scale_range'])
        
        # 4. Frame-level augmentation parameters
        brightness_factor = random.uniform(*self.augmentation_params['brightness_range'])
        contrast_factor = random.uniform(*self.augmentation_params['contrast_range'])
        
        # Apply transformations to landmarks
        for t in range(T):
            # Apply rotation to x,y coordinates
            hand1_landmarks[t, :, :2] = np.dot(hand1_landmarks[t, :, :2], rotation_matrix.T)
            hand2_landmarks[t, :, :2] = np.dot(hand2_landmarks[t, :, :2], rotation_matrix.T)
            pose_landmarks[t, :, :2] = np.dot(pose_landmarks[t, :, :2], rotation_matrix.T)
            
            # Apply shearing to x,y coordinates
            hand1_landmarks[t, :, :2] = np.dot(hand1_landmarks[t, :, :2], shear_matrix.T)
            hand2_landmarks[t, :, :2] = np.dot(hand2_landmarks[t, :, :2], shear_matrix.T)
            pose_landmarks[t, :, :2] = np.dot(pose_landmarks[t, :, :2], shear_matrix.T)
            
            # Apply scaling
            hand1_landmarks[t] *= scale_factor
            hand2_landmarks[t] *= scale_factor
            pose_landmarks[t, :, :3] *= scale_factor  # Don't scale visibility
            
            # Add frame-specific noise (small variation between frames)
            frame_noise_scale = self.augmentation_params['noise_scale'] * (0.8 + 0.4 * random.random())
            hand1_landmarks[t] += np.random.normal(0, frame_noise_scale, hand1_landmarks[t].shape)
            hand2_landmarks[t] += np.random.normal(0, frame_noise_scale, hand2_landmarks[t].shape)
            pose_landmarks[t, :, :3] += np.random.normal(0, frame_noise_scale, pose_landmarks[t, :, :3].shape)
            
            # Apply frame-level augmentations
            frame = augmented_frames[t].copy()
            
            # Brightness adjustment
            frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=(brightness_factor - 1) * 50)
            
            # Add subtle noise to frame
            if random.random() < 0.3:  # Apply to 30% of frames
                noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, noise)
            
            # Slight blur variation (simulate slight camera shake)
            if random.random() < 0.2:  # Apply to 20% of frames
                kernel_size = random.choice([3, 5])
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            
            augmented_frames[t] = frame
        
        # Reconstruct augmented landmarks
        augmented_landmarks = np.concatenate([
            hand1_landmarks.reshape(T, 63),
            hand2_landmarks.reshape(T, 63),
            pose_landmarks.reshape(T, 28)
        ], axis=1)
        
        return augmented_landmarks, augmented_frames
        
        
    def augment_landmarks(self, landmarks):
        """
        Applies augmentations to landmarks.
        This version correctly handles the mixed structure of hand and pose landmarks.
        """
        if self.mode != 'train':
            return landmarks
            
        landmarks = landmarks.copy() # Shape (T, 154)
        print("[DEBUG] in landmarks")

        # Deconstruct the flat array into structured landmarks for geometric augmentations
        hand1_landmarks = landmarks[:, :63].reshape(landmarks.shape[0], 21, 3)
        hand2_landmarks = landmarks[:, 63:126].reshape(landmarks.shape[0], 21, 3)
        pose_landmarks = landmarks[:, 126:].reshape(landmarks.shape[0], 7, 4)

        # 1. Random 2D Rotation on x,y coordinates
        if random.random() < 0.5:
            angle = random.uniform(-15, 15) # degrees
            rad = math.radians(angle)
            c, s = math.cos(rad), math.sin(rad)
            rotation_matrix = np.array([[c, -s], [s, c]])
            
            hand1_landmarks[:, :, :2] = np.dot(hand1_landmarks[:, :, :2], rotation_matrix.T)
            hand2_landmarks[:, :, :2] = np.dot(hand2_landmarks[:, :, :2], rotation_matrix.T)
            pose_landmarks[:, :, :2] = np.dot(pose_landmarks[:, :, :2], rotation_matrix.T)

        # 2. Random 2D Shearing on x,y coordinates
        if random.random() < 0.5:
            shear_factor = random.uniform(-0.1, 0.1)
            shear_matrix = np.array([[1, shear_factor], [0, 1]])

            hand1_landmarks[:, :, :2] = np.dot(hand1_landmarks[:, :, :2], shear_matrix.T)
            hand2_landmarks[:, :, :2] = np.dot(hand2_landmarks[:, :, :2], shear_matrix.T)
            pose_landmarks[:, :, :2] = np.dot(pose_landmarks[:, :, :2], shear_matrix.T)

        # Reconstruct the flat array before applying other augmentations
        augmented_landmarks = np.concatenate([
            hand1_landmarks.reshape(landmarks.shape[0], 63),
            hand2_landmarks.reshape(landmarks.shape[0], 63),
            pose_landmarks.reshape(landmarks.shape[0], 28)
        ], axis=1)

        # 3. Add small random noise
        noise_scale = 0.01
        augmented_landmarks += np.random.normal(0, noise_scale, augmented_landmarks.shape)
        
        # 5. Random scaling
        if random.random() < 0.4:
            scale_factor = random.uniform(0.95, 1.05)
            augmented_landmarks *= scale_factor
        
        return augmented_landmarks
        
        
        
    def normalize_landmarks(self, landmarks):
        """
        Normalizes the full video landmarks to be translation and scale invariant,
        preserving temporal consistency across frames.
        Input:
            landmarks: np.array of shape (T, 154)
        Output:
            normalized landmarks: np.array of shape (T, 154)
        """
        landmarks = landmarks.copy()  # Shape (T, 154)

        # Deconstruct the flat array into structured landmarks for the entire video
        T = landmarks.shape[0]
        hand1_all = landmarks[:, :63].reshape(T, 21, 3)        # (T, 21, 3)
        hand2_all = landmarks[:, 63:126].reshape(T, 21, 3)     # (T, 21, 3)
        pose_all = landmarks[:, 126:].reshape(T, 7, 4)         # (T, 7, 4)

        # Combine all (x,y,z) coordinates across time
        all_xyz = np.concatenate([
            hand1_all.reshape(T * 21, 3),
            hand2_all.reshape(T * 21, 3),
            pose_all[:, :, :3].reshape(T * 7, 3)
        ], axis=0)  # Shape: (T * 49 landmarks, 3)

        # Filter out zero landmarks
        valid_mask = np.any(all_xyz != 0, axis=1)
        if np.any(valid_mask):
            valid_coords = all_xyz[valid_mask]

            # === Translation normalization (X, Y) ===
            mean_pos = np.mean(valid_coords[:, :2], axis=0)
            all_xyz[:, :2] -= mean_pos

            # === Scale normalization (X, Y) ===
            distances = np.linalg.norm(valid_coords[:, :2], axis=1)
            max_dist = np.max(distances)
            if max_dist > 0:
                all_xyz[:, :2] /= max_dist

            # === Z normalization (standardization) ===
            z_coords = valid_coords[:, 2]
            if len(z_coords) > 1:
                z_mean = np.mean(z_coords)
                z_std = np.std(z_coords)
                if z_std > 0:
                    all_xyz[:, 2] = (all_xyz[:, 2] - z_mean) / z_std

        # Put back the normalized coordinates into original shape
        hand1_all = all_xyz[:T * 21].reshape(T, 21, 3)
        hand2_all = all_xyz[T * 21:T * 42].reshape(T, 21, 3)
        pose_xyz_all = all_xyz[T * 42:].reshape(T, 7, 3)

        # Reconstruct the landmarks array
        normalized_landmarks = np.zeros_like(landmarks)
        for t in range(T):
            pose_all[t, :, :3] = pose_xyz_all[t]

            normalized_landmarks[t] = np.concatenate([
                hand1_all[t].flatten(),
                hand2_all[t].flatten(),
                pose_all[t].flatten()
            ])

        return normalized_landmarks
    
    
        
    def preprocess_frame(self, frame, apply_blur=True):
        if frame is None or len(frame.shape) != 3:
            raise ValueError("Invalid input frame.")

        # Convert to LAB color space for better brightness-contrast handling
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the L channel (luminance)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # Merge enhanced L with original A and B channels
        merged_lab = cv2.merge((cl, a_channel, b_channel))

        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # Optionally apply Gaussian blur to smooth noise
        if apply_blur:
            enhanced_frame = cv2.GaussianBlur(enhanced_frame, (3, 3), 0)

        return enhanced_frame

    
    def extract_landmarks(self, frame):
        landmarks = []
        rgb_frame = cv2.cvtColor(self.preprocess_frame(frame), cv2.COLOR_BGR2RGB)
        
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]  # 21 landmarks (x, y, z) for each hand -> 21 * 3 = 63 for one hand
        hand_results = self.hands.process(rgb_frame)
        if not hand_results.multi_hand_landmarks:
            hand_results = self.hands.process(cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=20))
        
        if hand_results.multi_hand_landmarks:
            for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                if i >= 2: break
                hand_landmarks_list[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]).flatten()

        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])
        
        pose_lm_list = np.zeros(28)
        pose_results = self.pose.process(rgb_frame)
        if not pose_results.pose_landmarks:
            pose_results = self.pose.process(cv2.convertScaleAbs(rgb_frame, alpha=1.1, beta=15))

        if pose_results.pose_landmarks:
            pose_lm_list = np.array([
                [
                    pose_results.pose_landmarks.landmark[i].x,
                    pose_results.pose_landmarks.landmark[i].y,
                    pose_results.pose_landmarks.landmark[i].z,
                    pose_results.pose_landmarks.landmark[i].visibility
                ]
                for i in [0, 11, 12, 13, 14, 15, 16]
            ]).flatten() #(7 landmarks, each with (x, y, z, visibility) -> 7 * 4 = 28)

        landmarks.extend(pose_lm_list) #landmarks -> [hnd1, hand2, pose] -> [63 + 63 + 28 = 154]
        return np.array(landmarks)
    
    
    def detect_gesture_boundaries_with_debug(self, landmarks_sequence):
        """
        Enhanced version of detect_gesture_boundaries that returns debug information
        """
        if len(landmarks_sequence) < 2:
            return 0, len(landmarks_sequence) - 1, {
                'motion_scores': np.array([0]),
                'dynamic_threshold': self.motion_threshold,
                'raw_motion': np.array([0])
            }
            
        # Calculate frame-to-frame motion for hands and upper body
        motion_scores = []
        raw_motion_scores = []
        
        for i in range(1, len(landmarks_sequence)):
            # Focus on hand landmarks (0:126) and key pose points for motion detection
            prev_frame = landmarks_sequence[i-1][:126]  # Hand landmarks only
            curr_frame = landmarks_sequence[i][:126]
            
            # Calculate motion as average displacement of non-zero landmarks
            motion = 0
            valid_points = 0
            
            # Reshape to get individual landmark points
            prev_hands = prev_frame.reshape(-1, 3)  # (42, 3) for both hands
            curr_hands = curr_frame.reshape(-1, 3)
            
            for j in range(len(prev_hands)):
                # Only consider landmarks that are detected (non-zero)
                if np.any(prev_hands[j]) and np.any(curr_hands[j]):
                    point_motion = np.linalg.norm(curr_hands[j] - prev_hands[j])
                    motion += point_motion
                    valid_points += 1
                    
            if valid_points > 0:
                motion_scores.append(motion / valid_points)
                raw_motion_scores.append(motion / valid_points)
            else:
                motion_scores.append(0)
                raw_motion_scores.append(0)
        
        motion_scores = np.array(motion_scores)
        raw_motion_scores = np.array(raw_motion_scores)
        
        # Smooth the motion curve to reduce noise
        if len(motion_scores) > 1:
            motion_scores = gaussian_filter1d(motion_scores, sigma=1.0)
        
        # Find gesture boundaries using adaptive threshold
        mean_motion = np.mean(motion_scores)
        std_motion = np.std(motion_scores)
        dynamic_threshold = max(self.motion_threshold, mean_motion + 0.5 * std_motion)
        
        # Find start of gesture (first sustained motion)
        start_frame = 0
        for i in range(len(motion_scores)):
            if motion_scores[i] > dynamic_threshold:
                # Check if motion is sustained for at least 2 frames
                sustained = True
                for j in range(i, min(i + 2, len(motion_scores))):
                    if motion_scores[j] <= dynamic_threshold * 0.7:
                        sustained = False
                        break
                if sustained:
                    start_frame = i
                    break
        
        # Find end of gesture (last sustained motion)
        end_frame = len(motion_scores)
        for i in range(len(motion_scores) - 1, -1, -1):
            if motion_scores[i] > dynamic_threshold:
                # Check if motion was sustained before this point
                sustained = True
                for j in range(max(0, i - 1), i + 1):
                    if motion_scores[j] <= dynamic_threshold * 0.7:
                        sustained = False
                        break
                if sustained:
                    end_frame = i + 1  # +1 because motion_scores is 1 element shorter
                    break
        
        # Ensure minimum gesture length
        if end_frame - start_frame < self.min_gesture_length:
            # If detected gesture is too short, expand it
            center = (start_frame + end_frame) // 2
            half_min = self.min_gesture_length // 2
            start_frame = max(0, center - half_min)
            end_frame = min(len(landmarks_sequence), center + half_min + 1)
        
        debug_info = {
            'motion_scores': motion_scores,
            'raw_motion_scores': raw_motion_scores,
            'dynamic_threshold': dynamic_threshold,
            'mean_motion': mean_motion,
            'std_motion': std_motion,
            'static_threshold': self.motion_threshold
        }
        print("The start frame is", start_frame, "end frame is", end_frame)
        return start_frame, min(end_frame, len(landmarks_sequence) - 1), debug_info
    
    
    def extract_frames_from_timestamp_range(self, video_path, start_timestamp, end_timestamp, target_count=16):
        """
        Extract specific number of frames from a given timestamp range in the video
        Args:
            video_path: path to video file
            start_timestamp: starting timestamp in seconds
            end_timestamp: ending timestamp in seconds  
            target_count: number of frames to extract (default 16)
        Returns:
            frames, landmarks_sequence
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        print(f"[DEBUG] Video FPS: {fps}, Total duration: {total_duration:.2f}s")
        print(f"[DEBUG] Extracting from {start_timestamp:.2f}s to {end_timestamp:.2f}s")
        
        # Ensure timestamps are within bounds
        start_timestamp = max(0.0, start_timestamp)
        end_timestamp = min(total_duration, end_timestamp)
        
        # Calculate duration of the gesture segment
        gesture_duration = end_timestamp - start_timestamp
        
        if gesture_duration <= 0:
            cap.release()
            return self._create_empty_sequence(target_count)
        
        # Set video position to start timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, start_timestamp * 1000)
        
        # Calculate time interval between frames to extract
        if gesture_duration > 0:
            time_interval = gesture_duration / (target_count - 1) if target_count > 1 else 0
        else:
            time_interval = 0
        print("Time_interval i", time_interval)
        frames = []
        landmarks_sequence = []
        
        for i in range(target_count):
            # Calculate timestamp for this frame
            current_timestamp = start_timestamp + (i * time_interval)
            
            # Ensure we don't exceed end timestamp
            if current_timestamp > end_timestamp:
                current_timestamp = end_timestamp
            
            # Set video position to current timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, current_timestamp * 1000)
            
            ret, frame = cap.read()
            if ret:
                # Resize frame and extract landmarks
                resized_frame = cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
                frames.append(resized_frame)
                landmarks_sequence.append(self.extract_landmarks(frame))
                
                print(f"[DEBUG] Extracted frame {i+1}/{target_count} at timestamp {current_timestamp:.2f}s")
            else:
                print(f"[WARNING] Could not read frame at timestamp {current_timestamp:.2f}s")
                # If we can't read the frame, use the last valid frame or create empty
                if len(frames) > 0:
                    frames.append(frames[-1])
                    landmarks_sequence.append(landmarks_sequence[-1])
                else:
                    # Create empty frame
                    empty_frame = np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)
                    frames.append(empty_frame)
                    landmarks_sequence.append(np.zeros(154))  # Adjust based on your landmark size
        
        cap.release()
        
        # Ensure we have exactly target_count frames
        while len(frames) < target_count:
            if len(frames) > 0:
                frames.append(frames[-1])
                landmarks_sequence.append(landmarks_sequence[-1])
            else:
                empty_frame = np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)
                frames.append(empty_frame)
                landmarks_sequence.append(np.zeros(154))
        
        print(f"[DEBUG] Final extraction: {len(frames)} frames from timestamp range")
        return np.array(frames[:target_count]), np.array(landmarks_sequence[:target_count])
    
    

    def _create_empty_sequence(self, target_count):
        """Create empty frames and landmarks when no valid data is available"""
        frames = [np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)] * target_count
        landmarks_sequence = [np.zeros(154)] * target_count  # Adjust based on your landmark size
        return frames, landmarks_sequence

    
    
    def preprocess(self, video_path):
        """
        Enhanced preprocessing that first detects motion boundaries, then extracts frames from motion region using timestamps
        """

        # STEP 1: Initial sampling to detect motion boundaries
        initial_frames, initial_landmarks, frame_indices, frame_timestamps = self.preprocess_video(video_path)

        # STEP 2: Detect gesture boundaries from initial sampling with timestamps
        print(f"[DEBUG] Step 2: Detecting gesture boundaries from {len(initial_landmarks)} frames")
        motion_start_idx, motion_end_idx, debug_info = self.detect_gesture_boundaries_with_debug(
            initial_landmarks
        )
  
        start_timestamp = frame_timestamps[motion_start_idx]
        end_timestamp = frame_timestamps[motion_end_idx]
        
        
        # STEP 3: Extract exactly 16 frames from the timestamp range
        print(f"[DEBUG] Step 3: Extracting 16 frames from timestamp range")
        final_frames, final_landmarks = self.extract_frames_from_timestamp_range(
            video_path, 
            start_timestamp, 
            end_timestamp, 
            target_count=16
        )

        
        return final_frames, final_landmarks, debug_info

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames > self.config.SEQUENCE_LENGTH:
            frame_indices = np.linspace(0, total_frames - 1, self.config.SEQUENCE_LENGTH, dtype=int)

        else:
            frame_indices = list(range(total_frames))
            
        frame_timestamps = frame_indices / fps
        actual_timestamps = []
        frames, landmarks_sequence = [], []
        frame_idx, extracted_count = 0, 0
        while cap.isOpened() and extracted_count < len(frame_indices):
            ret, frame = cap.read()
            if not ret: break
            if frame_idx == frame_indices[extracted_count]:
                frames.append(cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)) # 224 * 224 * 3
                landmarks_sequence.append(self.extract_landmarks(frame))
                # Get actual timestamp of extracted frame
                actual_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                actual_timestamps.append(actual_timestamp)
                
                extracted_count += 1
            frame_idx += 1
        cap.release()
        
        # Convert to numpy array for easier handling
        actual_timestamps = np.array(actual_timestamps)
        current_length = len(frames)
        target_length = self.config.SEQUENCE_LENGTH
        if 0 < current_length < target_length:
            pad_needed = target_length - current_length
            frames.extend([frames[-1]] * pad_needed)
            landmarks_sequence.extend([landmarks_sequence[-1]] * pad_needed)
            frame_indices = np.pad(frame_indices, (0, pad_needed), mode='edge')
            # Pad timestamps with the last timestamp
            if len(actual_timestamps) > 0:
                actual_timestamps = np.pad(actual_timestamps, (0, pad_needed), mode='edge')
            else:
                actual_timestamps = np.zeros(target_length)
        elif current_length == 0:
            frames = [np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)] * target_length
            landmarks_sequence = [np.zeros(154)] * target_length
            frame_indices = np.zeros(target_length, dtype=int)
            actual_timestamps = np.zeros(target_length)
        
        return (np.array(frames[:target_length]), 
                np.array(landmarks_sequence[:target_length]), 
                frame_indices[:target_length],
                actual_timestamps[:target_length]) # (16, 224, 224, 3), (16, 154)

    def load_and_preprocess_data(self):
        print("Starting enhanced data preprocessing with augmentation...")
        manifest, label_set = [], set()
        categories = [d for d in os.listdir(self.config.DATA_PATH) if os.path.isdir(os.path.join(self.config.DATA_PATH, d))]
        sample_index = 0
        
        for category in tqdm(categories, desc="Processing categories"):
            category_path = os.path.join(self.config.DATA_PATH, category)
            for class_name in tqdm(os.listdir(category_path), desc=f"Processing {category}", leave=False):
                class_path = os.path.join(category_path, class_name)
                if not os.path.isdir(class_path): continue
                
                video_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                print(f"\n[INFO] Processing class {category}_{class_name} with {len(video_files)} original videos")
                
                for video_file in video_files:
                    video_path = os.path.join(class_path, video_file)
                    try:
                        # Process original video
                        frames, landmarks, _ = self.preprocess(video_path)
                        landmarks = self.normalize_landmarks(landmarks)
                        
                        video_name = os.path.splitext(video_file)[0]
                        label = f"{category}_{class_name}"
                        label_set.add(label)
                        
                        # Save original video
                        self._save_processed_video(frames, landmarks, category, class_name, 
                                                 video_name, "original", sample_index, 
                                                 video_path, label, manifest)
                        sample_index += 1
                        
                        # Create augmented versions
                        for aug_id in range(1, self.augmentation_factor + 1):
                            print(f"[DEBUG] Creating augmentation {aug_id}/{self.augmentation_factor} for {video_name}")
                            
                            # Create augmented landmarks and frames from original data
                            aug_landmarks, aug_frames = self.create_augmented_landmarks_and_frames(
                                landmarks, frames, aug_id
                            )
                            
                            # Normalize augmented landmarks
                            aug_landmarks = self.normalize_landmarks(aug_landmarks)
                            
                            # Save augmented video
                            aug_video_name = f"{video_name}_aug{aug_id}"
                            self._save_processed_video(aug_frames, aug_landmarks, category, class_name,
                                                     aug_video_name, f"augmented_{aug_id}", sample_index,
                                                     video_path, label, manifest)
                            sample_index += 1
                            
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Print statistics for this class
                class_samples = sum(1 for entry in manifest if entry["label"] == label)
                print(f"[INFO] Class {label}: {class_samples} total samples "
                      f"({len(video_files)} original + {len(video_files) * self.augmentation_factor} augmented)")
        
        # Encode labels
        labels = [entry["label"] for entry in manifest]
        y_encoded = self.label_encoder.fit_transform(labels)
        for i, entry in enumerate(manifest):
            entry["encoded_label"] = int(y_encoded[i])
        
        # Save manifest and label encoder
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Print final statistics
        total_classes = len(self.label_encoder.classes_)
        avg_samples_per_class = len(manifest) / total_classes if total_classes > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"✅ DATA AUGMENTATION COMPLETED!")
        print(f"📊 Total samples: {sample_index}")
        print(f"📊 Total classes: {total_classes}")
        print(f"📊 Average samples per class: {avg_samples_per_class:.1f}")
        print(f"📊 Augmentation factor: {self.augmentation_factor}x")
        print(f"{'='*60}")
        
        return manifest, list(self.label_encoder.classes_)
    
    
    def _save_processed_video(self, frames, landmarks, category, class_name, video_name, 
                            video_type, sample_index, original_video_path, label, manifest):
        """
        Helper method to save processed frames and landmarks
        """
        target_dir = os.path.join(self.config.CHUNKED_DATA_DIR, category, class_name, video_name)
        os.makedirs(target_dir, exist_ok=True)
        
        frame_path = os.path.join(target_dir, "frames.npy")
        landmarks_path = os.path.join(target_dir, "landmarks.npy")
        
        # Save frames (normalized to 0-1 range)
        np.save(frame_path, frames.astype('float32') / 255.0)
        # Save landmarks
        np.save(landmarks_path, landmarks)
        
        # Add to manifest
        manifest.append({
            "index": sample_index,
            "label": label,
            "original_video": original_video_path,
            "frame_path": frame_path,
            "landmarks_path": landmarks_path,
            "video_type": video_type,  # 'original', 'augmented_1', 'augmented_2', etc.
            "augmentation_id": video_type.split('_')[-1] if 'augmented' in video_type else 'original'
        })
        
    
    def load_data_for_training(self, test_size=0.2, batch_size=32, random_state=42, num_workers=2, shuffle=True, verbose=True):
        manifest_path = os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        with open(manifest_path, 'r') as f: 
            manifest = json.load(f)
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl"), 'rb') as f: 
            self.label_encoder = pickle.load(f)
        
        # Ensure stratified split considers both original and augmented samples
        labels = [entry["encoded_label"] for entry in manifest]
        
        # For better generalization, we can choose to:
        # Option 1: Include all augmented samples in training
        # Option 2: Keep original videos separate for testing when possible
        
        # Option 1 implementation (recommended for small datasets):
        manifest_train, manifest_test = train_test_split(
            manifest, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Option 2 implementation (if you want to ensure test set has only original videos):
        # Uncomment the following if you prefer this approach:
        """
        # Separate original and augmented samples
        original_samples = [entry for entry in manifest if entry["video_type"] == "original"]
        augmented_samples = [entry for entry in manifest if entry["video_type"] != "original"]
        
        # Split original samples for test set
        if len(original_samples) > 0:
            original_labels = [entry["encoded_label"] for entry in original_samples]
            orig_train, orig_test = train_test_split(
                original_samples, test_size=test_size, random_state=random_state, 
                stratify=original_labels if len(set(original_labels)) > 1 else None
            )
            
            # All augmented samples go to training
            manifest_train = orig_train + augmented_samples
            manifest_test = orig_test
        else:
            # Fallback to regular split if no original samples
            manifest_train, manifest_test = train_test_split(
                manifest, test_size=test_size, random_state=random_state, stratify=labels
            )
        """
        
        train_dataset = EnhancedVideoDataset(manifest_train, self.config, mode='train')
        test_dataset = EnhancedVideoDataset(manifest_test, self.config, mode='test')
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
            pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
            pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0
        )
        
        if verbose: 
            print("="*60 + "\nENHANCED DATA LOADING WITH AUGMENTATION COMPLETED!")
            print(f"Training samples: {len(manifest_train)}")
            print(f"Test samples: {len(manifest_test)}")
            
            # Show augmentation distribution in training set
            train_types = {}
            for entry in manifest_train:
                video_type = entry.get("video_type", "unknown")
                train_types[video_type] = train_types.get(video_type, 0) + 1
            
            print(f"Training set composition: {train_types}")
            print("="*60)
        
        return train_loader, test_loader
