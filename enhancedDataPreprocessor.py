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

class EnhancedVideoDataset(Dataset):
    """Enhanced PyTorch Dataset with robust augmentation for video data"""
    
    def __init__(self, manifest, config, mode='train', transform=None):
        self.manifest = manifest
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # Initialize selfie segmentation for background replacement
        if self.mode == 'train':
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # Create augmentation pipeline based on mode
        if mode == 'train':
            self.frame_transform = A.Compose([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.8),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5), # Increased rotation limit
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                ], p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.2),
                A.RandomShadow(p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            ])
        else:
            self.frame_transform = A.Compose([
                A.Resize(height=224, width=224),
            ])
    
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

    def normalize_landmarks(self, landmarks):
        """
        Normalizes landmarks to be translation and scale invariant.
        This version correctly handles the mixed structure of hand and pose landmarks.
        """
        landmarks = landmarks.copy() # Shape (T, 170)
        for t in range(landmarks.shape[0]): # For each frame
            frame_landmarks_flat = landmarks[t]

            # Deconstruct the flat array into structured landmarks
            hand1 = frame_landmarks_flat[:63].reshape(21, 3)
            hand2 = frame_landmarks_flat[63:126].reshape(21, 3)
            pose = frame_landmarks_flat[126:].reshape(11, 4)

            # Combine all (x,y,z) coordinates for normalization calculation
            all_xyz = np.concatenate([hand1, hand2, pose[:, :3]], axis=0) # Shape (53, 3)

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
                if max_dist > 0:
                    all_xyz[:, :2] /= max_dist
                
                # Z-coordinate normalization (standardization)
                z_coords = all_xyz[valid_mask, 2]
                if len(z_coords) > 1:
                    z_mean = np.mean(z_coords)
                    z_std = np.std(z_coords)
                    if z_std > 0:
                        all_xyz[:, 2] = (all_xyz[:, 2] - z_mean) / z_std
            
            # Put the normalized coordinates back into their original structures
            hand1_norm = all_xyz[:21]
            hand2_norm = all_xyz[21:42]
            pose_xyz_norm = all_xyz[42:]
            
            pose[:, :3] = pose_xyz_norm

            # Reconstruct the flat 170-dimensional array
            landmarks[t] = np.concatenate([
                hand1_norm.flatten(),
                hand2_norm.flatten(),
                pose.flatten()
            ])

        return landmarks
    
    def augment_landmarks(self, landmarks):
        """
        Applies augmentations to landmarks.
        This version correctly handles the mixed structure of hand and pose landmarks.
        """
        if self.mode != 'train':
            return landmarks
            
        landmarks = landmarks.copy() # Shape (T, 170)

        # Deconstruct the flat array into structured landmarks for geometric augmentations
        hand1_landmarks = landmarks[:, :63].reshape(landmarks.shape[0], 21, 3)
        hand2_landmarks = landmarks[:, 63:126].reshape(landmarks.shape[0], 21, 3)
        pose_landmarks = landmarks[:, 126:].reshape(landmarks.shape[0], 11, 4)

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
            pose_landmarks.reshape(landmarks.shape[0], 44)
        ], axis=1)

        # 3. Add small random noise
        noise_scale = 0.01
        augmented_landmarks += np.random.normal(0, noise_scale, augmented_landmarks.shape)
        
        # 4. Random temporal shift
        if random.random() < 0.3:
            shift = random.randint(-1, 1)
            if shift != 0:
                augmented_landmarks = np.roll(augmented_landmarks, shift, axis=0)
        
        # 5. Random scaling
        if random.random() < 0.4:
            scale_factor = random.uniform(0.95, 1.05)
            augmented_landmarks *= scale_factor
        
        return augmented_landmarks
    
    def temporal_augmentation(self, frames, landmarks):
        if self.mode != 'train':
            return frames, landmarks
        
        if random.random() < 0.3:
            seq_len = frames.shape[0]
            if seq_len > 8:
                start_idx = random.randint(0, max(1, seq_len - self.config.SEQUENCE_LENGTH))
                end_idx = min(start_idx + self.config.SEQUENCE_LENGTH, seq_len)
                indices = np.linspace(start_idx, end_idx-1, self.config.SEQUENCE_LENGTH, dtype=int)
                frames = frames[indices]
                landmarks = landmarks[indices]
        
        if random.random() < 0.2:
            dropout_rate = 0.1
            seq_len = frames.shape[0]
            keep_mask = np.random.random(seq_len) > dropout_rate
            keep_indices = np.where(keep_mask)[0]
            if len(keep_indices) > seq_len // 2:
                new_frames = np.zeros_like(frames)
                new_landmarks = np.zeros_like(landmarks)
                for i in range(seq_len):
                    closest_idx = keep_indices[np.argmin(np.abs(keep_indices - i))]
                    new_frames[i] = frames[closest_idx]
                    new_landmarks[i] = landmarks[closest_idx]
                frames, landmarks = new_frames, new_landmarks
        
        return frames, landmarks
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        
        try:
            frames = np.load(entry["frame_path"])
            landmarks = np.load(entry["landmarks_path"])
            label = entry["encoded_label"]
            
            landmarks = self.normalize_landmarks(landmarks)
            frames, landmarks = self.temporal_augmentation(frames, landmarks)
            
            augmented_frames = []
            for i in range(frames.shape[0]):
                frame = frames[i]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)

                if self.mode == 'train' and random.random() < 0.3:
                    frame = self.replace_background(frame)

                augmented = self.frame_transform(image=frame)['image']
                
                if isinstance(augmented, np.ndarray):
                    augmented = augmented.astype(np.float32) / 255.0
                
                augmented_frames.append(augmented)
            
            frames = np.stack(augmented_frames)
            landmarks = self.augment_landmarks(landmarks)
            
            frames = torch.from_numpy(frames).float()
            landmarks = torch.from_numpy(landmarks).float()
            label = torch.tensor(label, dtype=torch.long)
            
            if frames.dim() == 4 and frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
            
            if self.transform:
                frames, landmarks = self.transform(frames, landmarks)
            
            return (frames, landmarks), label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            T, H, W, C = self.config.SEQUENCE_LENGTH, 224, 224, 3
            return (torch.zeros(T, C, H, W), torch.zeros(T, 170)), torch.tensor(0, dtype=torch.long)

class EnhancedDataPreprocessor:
    """Enhanced data preprocessing module with better generalization techniques"""
    
    def __init__(self, config: ISLConfig):
        self.config = config
        self.config.create_directories()
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=config.MAX_NUM_HANDS, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.label_encoder = LabelEncoder()
        
    def preprocess_frame(self, frame):
        if len(frame.shape) == 3:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return cv2.GaussianBlur(frame, (3, 3), 0)
    
    def extract_landmarks(self, frame):
        landmarks = []
        rgb_frame = cv2.cvtColor(self.preprocess_frame(frame), cv2.COLOR_BGR2RGB)
        
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]
        hand_results = self.hands.process(rgb_frame)
        if not hand_results.multi_hand_landmarks:
            hand_results = self.hands.process(cv2.convertScaleAbs(rgb_frame, alpha=1.2, beta=20))
        
        if hand_results.multi_hand_landmarks:
            for i, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                if i >= 2: break
                hand_landmarks_list[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]).flatten()

        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])
        
        pose_lm_list = np.zeros(44)
        pose_results = self.pose.process(rgb_frame)
        if not pose_results.pose_landmarks:
            pose_results = self.pose.process(cv2.convertScaleAbs(rgb_frame, alpha=1.1, beta=15))

        if pose_results.pose_landmarks:
            pose_lm_list = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_results.pose_landmarks.landmark[:11]]).flatten()

        landmarks.extend(pose_lm_list)
        return np.array(landmarks)

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > self.config.SEQUENCE_LENGTH:
            seg_len = self.config.SEQUENCE_LENGTH // 3
            rem = self.config.SEQUENCE_LENGTH % 3
            begin_indices = np.linspace(0, total_frames//3 -1, seg_len, dtype=int)
            middle_indices = np.linspace(total_frames//3, 2*total_frames//3 - 1, seg_len, dtype=int)
            end_indices = np.linspace(2*total_frames//3, total_frames - 1, seg_len + rem, dtype=int)
            frame_indices = np.unique(np.concatenate([begin_indices, middle_indices, end_indices]))
        else:
            frame_indices = list(range(total_frames))
        
        frames, landmarks_sequence = [], []
        frame_idx, extracted_count = 0, 0
        while cap.isOpened() and extracted_count < len(frame_indices):
            ret, frame = cap.read()
            if not ret: break
            if frame_idx == frame_indices[extracted_count]:
                frames.append(cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4))
                landmarks_sequence.append(self.extract_landmarks(frame))
                extracted_count += 1
            frame_idx += 1
        cap.release()
        
        current_length = len(frames)
        target_length = self.config.SEQUENCE_LENGTH
        if 0 < current_length < target_length:
            pad_needed = target_length - current_length
            frames.extend([frames[-1]] * pad_needed)
            landmarks_sequence.extend([landmarks_sequence[-1]] * pad_needed)
        elif current_length == 0:
            frames = [np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)] * target_length
            landmarks_sequence = [np.zeros(170)] * target_length
        
        return np.array(frames[:target_length]), np.array(landmarks_sequence[:target_length])

    def load_and_preprocess_data(self):
        print("Starting enhanced data preprocessing...")
        manifest, label_set = [], set()
        categories = [d for d in os.listdir(self.config.DATA_PATH) if os.path.isdir(os.path.join(self.config.DATA_PATH, d))]
        sample_index = 0
        for category in tqdm(categories, desc="Processing categories"):
            category_path = os.path.join(self.config.DATA_PATH, category)
            for class_name in tqdm(os.listdir(category_path), desc=f"Processing {category}", leave=False):
                class_path = os.path.join(category_path, class_name)
                if not os.path.isdir(class_path): continue
                for video_file in os.listdir(class_path):
                    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')): continue
                    video_path = os.path.join(class_path, video_file)
                    try:
                        frames, landmarks = self.preprocess_video(video_path)
                        video_name = os.path.splitext(video_file)[0]
                        target_dir = os.path.join(self.config.CHUNKED_DATA_DIR, category, class_name, video_name)
                        os.makedirs(target_dir, exist_ok=True)
                        frame_path = os.path.join(target_dir, "frames.npy")
                        landmarks_path = os.path.join(target_dir, "landmarks.npy")
                        np.save(frame_path, frames.astype('float32') / 255.0)
                        np.save(landmarks_path, landmarks)
                        label = f"{category}_{class_name}"
                        label_set.add(label)
                        manifest.append({
                            "index": sample_index, "label": label, "original_video": video_path,
                            "frame_path": frame_path, "landmarks_path": landmarks_path
                        })
                        sample_index += 1
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")
        
        labels = [entry["label"] for entry in manifest]
        y_encoded = self.label_encoder.fit_transform(labels)
        for i, entry in enumerate(manifest):
            entry["encoded_label"] = int(y_encoded[i])
        
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ Saved {sample_index} samples. Found {len(self.label_encoder.classes_)} classes.")
        return manifest, list(self.label_encoder.classes_)

    def load_data_for_training(self, test_size=0.2, batch_size=32, random_state=42, num_workers=2, shuffle=True, verbose=True):
        manifest_path = os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        with open(manifest_path, 'r') as f: manifest = json.load(f)
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl"), 'rb') as f: self.label_encoder = pickle.load(f)
        
        labels = [entry["encoded_label"] for entry in manifest]
        manifest_train, manifest_test = train_test_split(manifest, test_size=test_size, random_state=random_state, stratify=labels)
        
        train_dataset = EnhancedVideoDataset(manifest_train, self.config, mode='train')
        test_dataset = EnhancedVideoDataset(manifest_test, self.config, mode='test')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
        
        if verbose: print("="*60 + "\nENHANCED DATA LOADING COMPLETED!\n" + "="*60)
        return train_loader, test_loader
