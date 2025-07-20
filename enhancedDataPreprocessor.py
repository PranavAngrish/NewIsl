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

class EnhancedVideoDataset(Dataset):
    """Enhanced PyTorch Dataset with robust augmentation for video data"""
    
    def __init__(self, manifest, config, mode='train', transform=None):
        self.manifest = manifest
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # Create augmentation pipeline based on mode
        if mode == 'train':
            self.frame_transform = A.Compose([
                # Color augmentations to handle different lighting
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                
                # Geometric augmentations
                A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.8),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
                A.HorizontalFlip(p=0.5),
                
                # Noise and blur to improve robustness
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                ], p=0.3),
                
                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.2),
                
                # Background augmentations
                A.RandomShadow(p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),
                
                # Cutout/Erasing
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            ])
        else:
            # Minimal augmentation for validation/test
            self.frame_transform = A.Compose([
                A.Resize(height=224, width=224),
            ])
    
    def __len__(self):
        return len(self.manifest)
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to make them translation and scale invariant"""
        landmarks = landmarks.copy()
        
        for t in range(landmarks.shape[0]):  # For each frame
            frame_landmarks = landmarks[t].reshape(-1, 3)  # Reshape to (N, 3)
            
            # Find valid landmarks (non-zero)
            valid_mask = np.any(frame_landmarks != 0, axis=1)
            
            if np.any(valid_mask):
                valid_landmarks = frame_landmarks[valid_mask]
                
                # Center normalization: subtract mean position
                mean_pos = np.mean(valid_landmarks[:, :2], axis=0)
                frame_landmarks[valid_mask, :2] -= mean_pos
                
                # Scale normalization: normalize by max distance from center
                distances = np.linalg.norm(frame_landmarks[valid_mask, :2], axis=1)
                max_dist = np.max(distances)
                if max_dist > 0:
                    frame_landmarks[valid_mask, :2] /= max_dist
                
                # Z-coordinate normalization
                z_coords = frame_landmarks[valid_mask, 2]
                if len(z_coords) > 1:
                    z_std = np.std(z_coords)
                    if z_std > 0:
                        frame_landmarks[valid_mask, 2] = (z_coords - np.mean(z_coords)) / z_std
            
            landmarks[t] = frame_landmarks.flatten()
        
        return landmarks
    
    def augment_landmarks(self, landmarks):
        """Apply augmentation to landmarks during training"""
        if self.mode != 'train':
            return landmarks
            
        landmarks = landmarks.copy()
        
        # Add small random noise to landmarks
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, landmarks.shape)
        landmarks += noise
        
        # Random temporal shift (small)
        if random.random() < 0.3:
            shift = random.randint(-1, 1)
            if shift != 0:
                landmarks = np.roll(landmarks, shift, axis=0)
        
        # Random scaling
        if random.random() < 0.4:
            scale_factor = random.uniform(0.95, 1.05)
            landmarks = landmarks * scale_factor
        
        return landmarks
    
    def temporal_augmentation(self, frames, landmarks):
        """Apply temporal augmentation techniques"""
        if self.mode != 'train':
            return frames, landmarks
        
        # Random temporal sampling
        if random.random() < 0.3:
            seq_len = frames.shape[0]
            if seq_len > 8:  # Only if we have enough frames
                start_idx = random.randint(0, max(1, seq_len - self.config.SEQUENCE_LENGTH))
                end_idx = min(start_idx + self.config.SEQUENCE_LENGTH, seq_len)
                
                indices = np.linspace(start_idx, end_idx-1, self.config.SEQUENCE_LENGTH, dtype=int)
                frames = frames[indices]
                landmarks = landmarks[indices]
        
        # Random frame dropout (simulate missing frames)
        if random.random() < 0.2:
            dropout_rate = 0.1
            seq_len = frames.shape[0]
            keep_mask = np.random.random(seq_len) > dropout_rate
            keep_indices = np.where(keep_mask)[0]
            
            if len(keep_indices) > seq_len // 2:  # Keep at least half the frames
                # Interpolate to maintain sequence length
                old_indices = np.arange(seq_len)
                new_frames = np.zeros_like(frames)
                new_landmarks = np.zeros_like(landmarks)
                
                for i in range(seq_len):
                    closest_idx = keep_indices[np.argmin(np.abs(keep_indices - i))]
                    new_frames[i] = frames[closest_idx]
                    new_landmarks[i] = landmarks[closest_idx]
                
                frames = new_frames
                landmarks = new_landmarks
        
        return frames, landmarks
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        
        try:
            # Load frames and landmarks
            frames = np.load(entry["frame_path"])  # Shape: (T, H, W, C)
            landmarks = np.load(entry["landmarks_path"])  # Shape: (T, 170)
            label = entry["encoded_label"]
            
            # Normalize landmarks for better generalization
            landmarks = self.normalize_landmarks(landmarks)
            
            # Apply temporal augmentation
            frames, landmarks = self.temporal_augmentation(frames, landmarks)
            
            # Apply frame-level augmentation
            augmented_frames = []
            for i in range(frames.shape[0]):
                frame = frames[i]
                if frame.max() <= 1.0:  # If already normalized
                    frame = (frame * 255).astype(np.uint8)
                
                augmented = self.frame_transform(image=frame)['image']
                
                # Convert back to float and normalize
                if isinstance(augmented, np.ndarray):
                    augmented = augmented.astype(np.float32) / 255.0
                
                augmented_frames.append(augmented)
            
            frames = np.stack(augmented_frames)
            
            # Apply landmark augmentation
            landmarks = self.augment_landmarks(landmarks)
            
            # Convert to torch tensors
            frames = torch.from_numpy(frames).float()
            landmarks = torch.from_numpy(landmarks).float()
            label = torch.tensor(label, dtype=torch.long)
            
            # Ensure frames are in (T, C, H, W) format
            if frames.dim() == 4 and frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
            
            if self.transform:
                frames, landmarks = self.transform(frames, landmarks)
            
            return (frames, landmarks), label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample in case of error
            T, H, W, C = self.config.SEQUENCE_LENGTH, 224, 224, 3
            frames = torch.zeros(T, C, H, W)
            landmarks = torch.zeros(T, 170)
            label = torch.tensor(0, dtype=torch.long)
            return (frames, landmarks), label

class EnhancedDataPreprocessor:
    """Enhanced data preprocessing module with better generalization techniques"""
    
    def __init__(self, config: ISLConfig):
        self.config = config
        self.config.create_directories()
        
        # Initialize MediaPipe with more robust settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.5,
            model_complexity=1  # Use more complex model
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use more complex model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.label_encoder = LabelEncoder()
        
        # Background subtractor for better focus on hands
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50
        )
        
    def preprocess_frame(self, frame):
        """Enhanced frame preprocessing for better landmark detection"""
        original_frame = frame.copy()
        
        # Apply histogram equalization for better contrast
        if len(frame.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        return frame
    
    def extract_landmarks(self, frame):
        """Enhanced landmark extraction with better error handling"""
        landmarks = []
        
        # Preprocess frame for better detection
        processed_frame = self.preprocess_frame(frame)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Hand landmarks with multiple attempts
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]  # Prepare 2 hand slots
        
        # Try with original frame first
        hand_results = self.hands.process(rgb_frame)
        
        # If no hands detected, try with enhanced contrast
        if not hand_results.multi_hand_landmarks:
            enhanced_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=20)
            hand_results = self.hands.process(enhanced_frame)
        
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if i >= 2:  # We only support MAX_NUM_HANDS = 2
                    break
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                hand_landmarks_list[i] = np.array(coords)

        # Add both hands' landmarks
        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])
        
        # Pose landmarks (upper body only) with multiple attempts
        pose_results = self.pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            # Extract only upper body landmarks (0-10)
            for i in range(11):
                landmark = pose_results.pose_landmarks.landmark[i]
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            # Try with enhanced frame if pose detection failed
            enhanced_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.1, beta=15)
            pose_results = self.pose.process(enhanced_frame)
            
            if pose_results.pose_landmarks:
                for i in range(11):
                    landmark = pose_results.pose_landmarks.landmark[i]
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                landmarks.extend([0.0] * 44)  # 11 landmarks * 4 coordinates
        
        return np.array(landmarks)
    
    def preprocess_video(self, video_path):
        """Enhanced video preprocessing with better frame sampling"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        landmarks_sequence = []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract with better temporal coverage
        if total_frames > self.config.SEQUENCE_LENGTH:
            # Use more sophisticated frame sampling
            # Take frames from beginning, middle, and end
            segment_size = total_frames // 3
            
            begin_indices = np.linspace(0, segment_size-1, self.config.SEQUENCE_LENGTH//3, dtype=int)
            middle_indices = np.linspace(segment_size, 2*segment_size-1, self.config.SEQUENCE_LENGTH//3, dtype=int)
            end_indices = np.linspace(2*segment_size, total_frames-1, self.config.SEQUENCE_LENGTH - 2*(self.config.SEQUENCE_LENGTH//3), dtype=int)
            
            frame_indices = np.concatenate([begin_indices, middle_indices, end_indices])
            frame_indices = np.unique(frame_indices)  # Remove duplicates
            frame_indices.sort()
            
            # If we still have too many, subsample evenly
            if len(frame_indices) > self.config.SEQUENCE_LENGTH:
                step = len(frame_indices) // self.config.SEQUENCE_LENGTH
                frame_indices = frame_indices[::step][:self.config.SEQUENCE_LENGTH]
        else:
            frame_indices = list(range(total_frames))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_indices:
                # Resize frame with better interpolation
                frame_resized = cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame_resized)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                landmarks_sequence.append(landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        # Pad sequences if needed with better interpolation
        target_length = self.config.SEQUENCE_LENGTH
        current_length = len(frames)
        
        if current_length < target_length:
            if current_length > 0:
                # Interpolate missing frames instead of just repeating
                pad_needed = target_length - current_length
                
                # Add interpolated frames
                for i in range(pad_needed):
                    # Use weighted average of existing frames
                    if current_length == 1:
                        frames.append(frames[0])
                        landmarks_sequence.append(landmarks_sequence[0])
                    else:
                        # Linear interpolation between random frames
                        idx1 = random.randint(0, current_length-1)
                        idx2 = random.randint(0, current_length-1)
                        alpha = random.random()
                        
                        interpolated_frame = (alpha * frames[idx1] + (1-alpha) * frames[idx2]).astype(np.uint8)
                        interpolated_landmarks = alpha * landmarks_sequence[idx1] + (1-alpha) * landmarks_sequence[idx2]
                        
                        frames.append(interpolated_frame)
                        landmarks_sequence.append(interpolated_landmarks)
            else:
                # Create blank frames if no frames extracted
                for i in range(target_length):
                    blank_frame = np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)
                    frames.append(blank_frame)
                    landmarks_sequence.append(np.zeros(170))
        
        # Ensure we have exactly the target length
        frames = frames[:target_length]
        landmarks_sequence = landmarks_sequence[:target_length]
        
        return np.array(frames), np.array(landmarks_sequence)

    def load_and_preprocess_data(self):
        """Process and save video data with enhanced preprocessing"""
        print("Starting enhanced data preprocessing...")

        manifest = []
        label_set = set()

        categories = [d for d in os.listdir(self.config.DATA_PATH)
                      if os.path.isdir(os.path.join(self.config.DATA_PATH, d))]

        sample_index = 0
        for category in tqdm(categories, desc="Processing categories"):
            category_path = os.path.join(self.config.DATA_PATH, category)

            classes = [d for d in os.listdir(category_path)
                       if os.path.isdir(os.path.join(category_path, d))]

            for class_name in tqdm(classes, desc=f"Processing {category}", leave=False):
                class_path = os.path.join(category_path, class_name)

                video_files = [f for f in os.listdir(class_path)
                               if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

                for video_file in video_files:
                    video_path = os.path.join(class_path, video_file)

                    try:
                        frames, landmarks = self.preprocess_video(video_path)

                        # Create output directory matching the input structure
                        video_name = os.path.splitext(video_file)[0]
                        target_dir = os.path.join(self.config.CHUNKED_DATA_DIR, category, class_name, video_name)
                        os.makedirs(target_dir, exist_ok=True)

                        # Save frames and landmarks (normalized frames)
                        frame_path = os.path.join(target_dir, "frames.npy")
                        landmarks_path = os.path.join(target_dir, "landmarks.npy")
                        np.save(frame_path, frames.astype('float32') / 255.0)
                        np.save(landmarks_path, landmarks)

                        # Metadata
                        label = f"{category}_{class_name}"
                        label_set.add(label)

                        sample_meta = {
                            "index": sample_index,
                            "label": label,
                            "original_video": video_path,
                            "frame_path": frame_path,
                            "landmarks_path": landmarks_path,
                            "sequence_length": self.config.SEQUENCE_LENGTH,
                            "frame_shape": list(frames.shape),
                            "landmarks_shape": list(landmarks.shape)
                        }

                        with open(os.path.join(target_dir, "metadata.json"), 'w') as f:
                            json.dump(sample_meta, f, indent=2)

                        manifest.append(sample_meta)
                        sample_index += 1
    
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")
                        continue

        # Encode labels
        labels = [entry["label"] for entry in manifest]
        y_encoded = self.label_encoder.fit_transform(labels)
        class_names = list(self.label_encoder.classes_)

        # Add encoded labels to each entry + update each metadata.json
        for i, entry in enumerate(manifest):
            entry["encoded_label"] = int(y_encoded[i])

            metadata_path = os.path.join(self.config.CHUNKED_DATA_DIR,
                                         os.path.relpath(entry["frame_path"], start=self.config.CHUNKED_DATA_DIR))
            metadata_path = os.path.join(os.path.dirname(metadata_path), "metadata.json")

            with open(metadata_path, 'w') as f:
                json.dump(entry, f, indent=2)

        # Save manifest and label encoder
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)

        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"✅ Saved {sample_index} samples with enhanced preprocessing.")
        print(f"🔖 Found {len(class_names)} classes.")
        return manifest, class_names

    def load_data_for_training(self, test_size=0.2, batch_size=32, random_state=42, num_workers=2, shuffle=True, verbose=True):
        """
        Load data for training with enhanced dataset and augmentation.
        """
        if verbose:
            print("=" * 60)
            print("LOADING DATA WITH ENHANCED AUGMENTATION")
            print("=" * 60)

        # Load manifest
        manifest_path = os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Load label encoder
        label_encoder_path = os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")

        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        if verbose:
            print(f"📋 Found {len(manifest)} samples")
            print(f"🔖 Found {len(self.label_encoder.classes_)} classes")

        # Stratified split
        labels = [entry["encoded_label"] for entry in manifest]
        manifest_train, manifest_test = train_test_split(
            manifest,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        if verbose:
            print(f"✅ Train/Test split completed:")
            print(f"   Train samples: {len(manifest_train)}")
            print(f"   Test samples: {len(manifest_test)}")

        # Create enhanced datasets with augmentation
        train_dataset = EnhancedVideoDataset(manifest_train, self.config, mode='train')
        test_dataset = EnhancedVideoDataset(manifest_test, self.config, mode='test')

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )

        if verbose:
            print("=" * 60)
            print("ENHANCED DATA LOADING COMPLETED!")
            print("✓ Applied robust data augmentation")
            print("✓ Improved landmark normalization") 
            print("✓ Enhanced frame preprocessing")
            print("✓ Temporal augmentation enabled")
            print("=" * 60)

        return train_loader, test_loader