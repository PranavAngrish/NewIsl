import torch
import torch.nn.functional as F
import numpy as np
import random

class ISLAugmentation:
    """Advanced augmentation for ISL video data"""
    
    def __init__(self, config):
        self.config = config
        
    def temporal_augmentation(self, frames, landmarks, p=0.5):
        """Apply temporal augmentations"""
        if random.random() > p:
            return frames, landmarks
            
        # Random temporal cropping
        if frames.shape[0] > self.config.SEQUENCE_LENGTH:
            start_idx = random.randint(0, frames.shape[0] - self.config.SEQUENCE_LENGTH)
            frames = frames[start_idx:start_idx + self.config.SEQUENCE_LENGTH]
            landmarks = landmarks[start_idx:start_idx + self.config.SEQUENCE_LENGTH]
        
        # Random speed variation (frame skipping/duplication)
        if random.random() < 0.3:
            speed_factor = random.uniform(0.8, 1.2)
            if speed_factor < 1.0:  # Slow down (duplicate frames)
                indices = torch.linspace(0, frames.shape[0]-1, int(frames.shape[0]/speed_factor)).long()
            else:  # Speed up (skip frames)
                indices = torch.linspace(0, frames.shape[0]-1, int(frames.shape[0]*speed_factor)).long()
                indices = indices[:self.config.SEQUENCE_LENGTH]
            
            frames = frames[indices]
            landmarks = landmarks[indices]
            
            # Pad if necessary
            while frames.shape[0] < self.config.SEQUENCE_LENGTH:
                frames = torch.cat([frames, frames[-1:]], dim=0)
                landmarks = torch.cat([landmarks, landmarks[-1:]], dim=0)
                
        return frames[:self.config.SEQUENCE_LENGTH], landmarks[:self.config.SEQUENCE_LENGTH]
    
    def spatial_augmentation(self, frames, p=0.7):
        """Apply spatial augmentations to frames"""
        if random.random() > p:
            return frames
            
        # Random rotation (small angles)
        if random.random() < 0.4:
            angle = random.uniform(-10, 10)  # degrees
            angle_rad = torch.tensor(angle * np.pi / 180.0)
            cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
            
            # Create rotation matrix
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ]).unsqueeze(0).repeat(frames.shape[0], 1, 1)
            
            grid = F.affine_grid(rotation_matrix, frames.shape, align_corners=False)
            frames = F.grid_sample(frames, grid, align_corners=False)
        
        # Random scaling
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            scale_matrix = torch.tensor([
                [scale, 0, 0],
                [0, scale, 0]
            ]).unsqueeze(0).repeat(frames.shape[0], 1, 1)
            
            grid = F.affine_grid(scale_matrix, frames.shape, align_corners=False)
            frames = F.grid_sample(frames, grid, align_corners=False)
        
        # Color jittering
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            frames = frames * contrast + (brightness - 1.0)
            frames = torch.clamp(frames, 0, 1)
        
        return frames
    
    def landmark_augmentation(self, landmarks, p=0.6):
        """Apply augmentations to landmarks"""
        if random.random() > p:
            return landmarks
            
        # Add small Gaussian noise
        if random.random() < 0.5:
            noise_std = 0.02  # Small noise
            noise = torch.randn_like(landmarks) * noise_std
            landmarks = landmarks + noise
        
        # Random landmark dropout (set some to zero)
        if random.random() < 0.3:
            dropout_prob = 0.1
            mask = torch.rand_like(landmarks) > dropout_prob
            landmarks = landmarks * mask
        
        # Mirror horizontally (flip x coordinates)
        if random.random() < 0.2:
            # For hand landmarks (first 126 features: 2 hands * 21 points * 3 coords)
            for i in range(0, 126, 3):  # Every 3rd element is x coordinate
                landmarks[:, i] = 1.0 - landmarks[:, i]
        
        return landmarks
    
    def normalize_landmarks_relative(self, landmarks):
        """Normalize landmarks relative to pose keypoints for better generalization"""
        # Extract pose landmarks (last 44 features: 11 points * 4 coords)
        pose_landmarks = landmarks[:, -44:].reshape(landmarks.shape[0], 11, 4)
        
        # Use shoulder midpoint as reference (assuming shoulders are at indices 5,6)
        if pose_landmarks.shape[1] > 6:
            left_shoulder = pose_landmarks[:, 5, :2]  # x, y only
            right_shoulder = pose_landmarks[:, 6, :2]
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
            
            # Normalize hand landmarks relative to shoulder midpoint
            hand_landmarks = landmarks[:, :126].reshape(landmarks.shape[0], 42, 3)
            hand_landmarks[:, :, :2] = hand_landmarks[:, :, :2] - shoulder_midpoint.unsqueeze(1)
            
            # Reconstruct landmarks
            landmarks[:, :126] = hand_landmarks.reshape(landmarks.shape[0], 126)
        
        return landmarks
    
    def __call__(self, frames, landmarks):
        """Apply all augmentations"""
        # Temporal augmentation
        frames, landmarks = self.temporal_augmentation(frames, landmarks)
        
        # Spatial augmentation on frames
        frames = self.spatial_augmentation(frames)
        
        # Landmark augmentation
        landmarks = self.landmark_augmentation(landmarks)
        
        # Normalize landmarks
        landmarks = self.normalize_landmarks_relative(landmarks)
        
        return frames, landmarks

# Updated VideoDataset class with augmentation
class AugmentedVideoDataset:
    """Enhanced VideoDataset with augmentation"""
    
    def __init__(self, manifest, config, is_training=True, transform=None):
        self.manifest = manifest
        self.config = config
        self.is_training = is_training
        self.transform = transform
        self.augmentation = ISLAugmentation(config) if is_training else None
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        
        try:
            # Load frames and landmarks
            frames = np.load(entry["frame_path"])  # Shape: (T, H, W, C)
            landmarks = np.load(entry["landmarks_path"])  # Shape: (T, 170)
            label = entry["encoded_label"]
            
            # Convert to torch tensors
            frames = torch.from_numpy(frames).float()
            landmarks = torch.from_numpy(landmarks).float()
            label = torch.tensor(label, dtype=torch.long)
            
            # Permute frames to (T, C, H, W) for PyTorch
            frames = frames.permute(0, 3, 1, 2)
            
            # Apply augmentation during training
            if self.is_training and self.augmentation:
                frames, landmarks = self.augmentation(frames, landmarks)
            
            if self.transform:
                frames, landmarks = self.transform(frames, landmarks)
            
            return (frames, landmarks), label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample in case of error
            T, H, W, C = 16, 224, 224, 3
            frames = torch.zeros(T, C, H, W)
            landmarks = torch.zeros(T, 170)
            label = torch.tensor(0, dtype=torch.long)
            return (frames, landmarks), label