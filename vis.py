import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.ndimage import gaussian_filter1d
import cv2

class GestureSimilarityCalculator:
    def __init__(self, motion_threshold=0.02, min_gesture_length=3):
        self.scaler = StandardScaler()
        self.motion_threshold = motion_threshold  # Minimum motion to consider as gesture
        self.min_gesture_length = min_gesture_length  # Minimum frames for valid gesture
        
    def detect_gesture_boundaries(self, landmarks_sequence):
        """
        Detect the start and end of actual gesture movement
        Args:
            landmarks_sequence: (16, 154) array of landmarks
        Returns:
            (start_frame, end_frame) tuple indicating gesture boundaries
        """
        if len(landmarks_sequence) < 2:
            return 0, len(landmarks_sequence) - 1
            
        # Calculate frame-to-frame motion for hands and upper body
        motion_scores = []
        
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
            else:
                motion_scores.append(0)
        
        motion_scores = np.array(motion_scores)
        
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
        
        return start_frame, min(end_frame, len(landmarks_sequence) - 1)
    
    def extract_gesture_segment(self, landmarks_sequence):
        """
        Extract only the gesture portion from the sequence
        Args:
            landmarks_sequence: (16, 154) array of landmarks
        Returns:
            gesture_landmarks: extracted gesture segment
            boundaries: (start, end) frame indices
        """
        start_frame, end_frame = self.detect_gesture_boundaries(landmarks_sequence)
        gesture_segment = landmarks_sequence[start_frame:end_frame + 1]
        
        return gesture_segment, (start_frame, end_frame)
    
    def align_gesture_sequences(self, landmarks1, landmarks2):
        """
        Extract and align gesture segments from both sequences
        Args:
            landmarks1: (16, 154) landmarks for video 1
            landmarks2: (16, 154) landmarks for video 2
        Returns:
            aligned_landmarks1, aligned_landmarks2, alignment_info
        """
        # Extract gesture segments
        gesture1, bounds1 = self.extract_gesture_segment(landmarks1)
        gesture2, bounds2 = self.extract_gesture_segment(landmarks2)
        
        # If one gesture is significantly longer, we may want to resample
        len1, len2 = len(gesture1), len(gesture2)
        
        alignment_info = {
            'original_bounds1': bounds1,
            'original_bounds2': bounds2,
            'gesture_lengths': (len1, len2),
            'length_ratio': max(len1, len2) / max(min(len1, len2), 1)
        }
        
        # If length difference is significant, resample to match
        if alignment_info['length_ratio'] > 1.5:
            target_length = max(len1, len2)
            
            if len1 < target_length:
                gesture1 = self._resample_sequence(gesture1, target_length)
            if len2 < target_length:
                gesture2 = self._resample_sequence(gesture2, target_length)
                
        return gesture1, gesture2, alignment_info
    
    def _resample_sequence(self, sequence, target_length):
        """
        Resample sequence to target length using interpolation
        Args:
            sequence: (N, 154) sequence to resample
            target_length: desired length
        Returns:
            resampled sequence
        """
        if len(sequence) == target_length:
            return sequence
            
        # Create interpolation indices
        original_indices = np.linspace(0, len(sequence) - 1, len(sequence))
        target_indices = np.linspace(0, len(sequence) - 1, target_length)
        
        # Interpolate each landmark dimension
        resampled = np.zeros((target_length, sequence.shape[1]))
        
        for dim in range(sequence.shape[1]):
            resampled[:, dim] = np.interp(target_indices, original_indices, sequence[:, dim])
            
        return resampled
        
    def normalize_landmarks(self, landmarks_sequence):
        """
        Normalize full landmark sequences to make them scale and translation invariant
        Args:
            landmarks_sequence: (N, 154) array of full landmarks
        Returns:
            normalized landmarks
        """
        # Reshape to (N, 154) if needed
        if len(landmarks_sequence.shape) == 3:
            landmarks_sequence = landmarks_sequence.reshape(landmarks_sequence.shape[0], -1)
            
        # Check if this is full landmarks (154) or partial
        feature_count = landmarks_sequence.shape[1]
        
        if feature_count == 154:
            return self._normalize_full_landmarks(landmarks_sequence)
        elif feature_count == 126:
            return self._normalize_hand_landmarks(landmarks_sequence)
        elif feature_count == 28:
            return self._normalize_pose_landmarks(landmarks_sequence)
        else:
            raise ValueError(f"Unsupported landmark feature count: {feature_count}")
    
    def _normalize_full_landmarks(self, landmarks_sequence):
        """Normalize full 154-feature landmarks"""
        normalized_sequence = []
        
        for frame_landmarks in landmarks_sequence:
            # Split into hand1 (63), hand2 (63), pose (28)
            hand1_landmarks = frame_landmarks[:63].reshape(-1, 3)  # (21, 3)
            hand2_landmarks = frame_landmarks[63:126].reshape(-1, 3)  # (21, 3)
            pose_landmarks = frame_landmarks[126:154].reshape(-1, 4)  # (7, 4)
            
            # Normalize each hand relative to wrist (first landmark)
            if np.any(hand1_landmarks):
                wrist1 = hand1_landmarks[0]
                hand1_normalized = hand1_landmarks - wrist1
                hand1_normalized = hand1_normalized.flatten()
            else:
                hand1_normalized = hand1_landmarks.flatten()
                
            if np.any(hand2_landmarks):
                wrist2 = hand2_landmarks[0]
                hand2_normalized = hand2_landmarks - wrist2
                hand2_normalized = hand2_normalized.flatten()
            else:
                hand2_normalized = hand2_landmarks.flatten()
            
            # Normalize pose landmarks relative to nose (first landmark)
            if np.any(pose_landmarks):
                nose = pose_landmarks[0, :3]  # x, y, z only
                pose_normalized = pose_landmarks.copy()
                pose_normalized[:, :3] = pose_landmarks[:, :3] - nose
                pose_normalized = pose_normalized.flatten()
            else:
                pose_normalized = pose_landmarks.flatten()
            
            # Combine normalized landmarks
            frame_normalized = np.concatenate([hand1_normalized, hand2_normalized, pose_normalized])
            normalized_sequence.append(frame_normalized)
            
        return np.array(normalized_sequence)
    
    def _normalize_hand_landmarks(self, landmarks_sequence):
        """Normalize 126-feature hand landmarks"""
        normalized_sequence = []
        
        for frame_landmarks in landmarks_sequence:
            # Split into hand1 (63), hand2 (63)
            hand1_landmarks = frame_landmarks[:63].reshape(-1, 3)  # (21, 3)
            hand2_landmarks = frame_landmarks[63:126].reshape(-1, 3)  # (21, 3)
            
            # Normalize each hand relative to wrist (first landmark)
            if np.any(hand1_landmarks):
                wrist1 = hand1_landmarks[0]
                hand1_normalized = hand1_landmarks - wrist1
                hand1_normalized = hand1_normalized.flatten()
            else:
                hand1_normalized = hand1_landmarks.flatten()
                
            if np.any(hand2_landmarks):
                wrist2 = hand2_landmarks[0]
                hand2_normalized = hand2_landmarks - wrist2
                hand2_normalized = hand2_normalized.flatten()
            else:
                hand2_normalized = hand2_landmarks.flatten()
            
            # Combine normalized hand landmarks
            frame_normalized = np.concatenate([hand1_normalized, hand2_normalized])
            normalized_sequence.append(frame_normalized)
            
        return np.array(normalized_sequence)
    
    def _normalize_pose_landmarks(self, landmarks_sequence):
        """Normalize 28-feature pose landmarks"""
        normalized_sequence = []
        
        for frame_landmarks in landmarks_sequence:
            # Reshape to (7, 4) for pose landmarks
            pose_landmarks = frame_landmarks.reshape(-1, 4)  # (7, 4)
            
            # Normalize pose landmarks relative to nose (first landmark)
            if np.any(pose_landmarks):
                nose = pose_landmarks[0, :3]  # x, y, z only (exclude visibility)
                pose_normalized = pose_landmarks.copy()
                pose_normalized[:, :3] = pose_landmarks[:, :3] - nose
                pose_normalized = pose_normalized.flatten()
            else:
                pose_normalized = pose_landmarks.flatten()
            
            normalized_sequence.append(pose_normalized)
            
        return np.array(normalized_sequence)
    
    def calculate_frame_similarity(self, frame1_landmarks, frame2_landmarks, method='cosine'):
        """
        Calculate similarity between two frames
        Args:
            frame1_landmarks: (154,) landmarks for frame 1
            frame2_landmarks: (154,) landmarks for frame 2
            method: 'cosine', 'euclidean', 'correlation'
        Returns:
            similarity score (higher = more similar)
        """
        if method == 'cosine':
            # Cosine similarity (0 to 1, higher is more similar)
            similarity = cosine_similarity([frame1_landmarks], [frame2_landmarks])[0][0]
            return max(0, similarity)  # Ensure non-negative
            
        elif method == 'euclidean':
            # Euclidean distance (convert to similarity)
            distance = euclidean(frame1_landmarks, frame2_landmarks)
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            return similarity
            
        elif method == 'correlation':
            # Pearson correlation coefficient
            if np.std(frame1_landmarks) == 0 or np.std(frame2_landmarks) == 0:
                return 0.0
            correlation, _ = pearsonr(frame1_landmarks, frame2_landmarks)
            return max(0, correlation)  # Only positive correlations
            
        else:
            raise ValueError("Method must be 'cosine', 'euclidean', or 'correlation'")
    
    def calculate_sequence_similarity(self, landmarks1, landmarks2, method='dtw_aligned'):
        """
        Calculate similarity between two gesture sequences with temporal alignment
        Args:
            landmarks1: (16, 154) landmarks for video 1
            landmarks2: (16, 154) landmarks for video 2
            method: 'dtw_aligned', 'average_frames', 'dtw', 'weighted_temporal'
        Returns:
            similarity score (0 to 1, higher is more similar)
        """
        if method == 'dtw_aligned':
            # First align gesture segments, then apply DTW
            aligned_landmarks1, aligned_landmarks2, alignment_info = self.align_gesture_sequences(landmarks1, landmarks2)
            
            # Normalize the aligned sequences
            norm_landmarks1 = self.normalize_landmarks(aligned_landmarks1)
            norm_landmarks2 = self.normalize_landmarks(aligned_landmarks2)
            
            # Apply DTW on the aligned sequences
            distance, _ = fastdtw(norm_landmarks1, norm_landmarks2, dist=euclidean)
            
            # Convert distance to similarity
            max_possible_distance = np.sqrt(len(norm_landmarks1) * len(norm_landmarks1[0]))
            similarity = 1 - (distance / (max_possible_distance * max(len(norm_landmarks1), len(norm_landmarks2))))
            
            return max(0, similarity)
            
        else:
            # For other methods, use the original implementation but with alignment
            aligned_landmarks1, aligned_landmarks2, _ = self.align_gesture_sequences(landmarks1, landmarks2)
            
            # Normalize landmarks first
            norm_landmarks1 = self.normalize_landmarks(aligned_landmarks1)
            norm_landmarks2 = self.normalize_landmarks(aligned_landmarks2)
            
            if method == 'average_frames':
                # Simple average of frame-by-frame similarities
                min_length = min(len(norm_landmarks1), len(norm_landmarks2))
                similarities = []
                for i in range(min_length):
                    sim = self.calculate_frame_similarity(norm_landmarks1[i], norm_landmarks2[i], 'cosine')
                    similarities.append(sim)
                return np.mean(similarities)
                
            elif method == 'dtw':
                # Pure DTW without pre-alignment
                distance, _ = fastdtw(norm_landmarks1, norm_landmarks2, dist=euclidean)
                max_possible_distance = np.sqrt(len(norm_landmarks1) * len(norm_landmarks1[0]))
                similarity = 1 - (distance / (max_possible_distance * len(norm_landmarks1)))
                return max(0, similarity)
                
            elif method == 'weighted_temporal':
                # Weight middle frames more heavily (peak gesture moments)
                min_length = min(len(norm_landmarks1), len(norm_landmarks2))
                similarities = []
                weights = self._get_temporal_weights(min_length)
                
                for i in range(min_length):
                    sim = self.calculate_frame_similarity(norm_landmarks1[i], norm_landmarks2[i], 'cosine')
                    similarities.append(sim * weights[i])
                
                return np.sum(similarities) / np.sum(weights)
                
            else:
                raise ValueError("Method must be 'dtw_aligned', 'average_frames', 'dtw', or 'weighted_temporal'")
    
    def _get_temporal_weights(self, sequence_length):
        """Generate weights that emphasize middle frames"""
        weights = np.ones(sequence_length)
        middle = sequence_length // 2
        
        # Create a bell curve centered at the middle
        for i in range(sequence_length):
            distance_from_middle = abs(i - middle)
            weights[i] = np.exp(-0.3 * distance_from_middle)
            
        return weights
    
    def calculate_hand_similarity(self, landmarks1, landmarks2):
        """
        Calculate similarity focusing only on hand landmarks with alignment
        Args:
            landmarks1: (16, 154) landmarks for video 1
            landmarks2: (16, 154) landmarks for video 2
        Returns:
            hand similarity score
        """
        # First align the full sequences to get proper gesture boundaries
        aligned_landmarks1, aligned_landmarks2, _ = self.align_gesture_sequences(landmarks1, landmarks2)
        
        # Extract hand landmarks only (first 126 features)
        hand_landmarks1 = aligned_landmarks1[:, :126]
        hand_landmarks2 = aligned_landmarks2[:, :126]
        
        # Use DTW for hand comparison as hands have more complex temporal patterns
        norm_landmarks1 = self.normalize_landmarks(hand_landmarks1)
        norm_landmarks2 = self.normalize_landmarks(hand_landmarks2)
        
        distance, _ = fastdtw(norm_landmarks1, norm_landmarks2, dist=euclidean)
        max_possible_distance = np.sqrt(len(norm_landmarks1) * len(norm_landmarks1[0]))
        similarity = 1 - (distance / (max_possible_distance * max(len(norm_landmarks1), len(norm_landmarks2))))
        
        return max(0, similarity)
    
    def calculate_pose_similarity(self, landmarks1, landmarks2):
        """
        Calculate similarity focusing only on pose landmarks with alignment
        Args:
            landmarks1: (16, 154) landmarks for video 1
            landmarks2: (16, 154) landmarks for video 2
        Returns:
            pose similarity score
        """
        # First align the full sequences to get proper gesture boundaries
        aligned_landmarks1, aligned_landmarks2, _ = self.align_gesture_sequences(landmarks1, landmarks2)
        
        # Extract pose landmarks only (last 28 features)
        pose_landmarks1 = aligned_landmarks1[:, 126:]
        pose_landmarks2 = aligned_landmarks2[:, 126:]
        
        # Use simpler comparison for pose as it's usually more stable
        norm_landmarks1 = self.normalize_landmarks(pose_landmarks1)
        norm_landmarks2 = self.normalize_landmarks(pose_landmarks2)
        
        # Use average frame similarity for pose
        min_length = min(len(norm_landmarks1), len(norm_landmarks2))
        similarities = []
        for i in range(min_length):
            sim = self.calculate_frame_similarity(norm_landmarks1[i], norm_landmarks2[i], 'cosine')
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_comprehensive_similarity(self, landmarks1, landmarks2):
        """
        Calculate comprehensive similarity using multiple methods with temporal alignment
        Args:
            landmarks1: (16, 154) landmarks for video 1
            landmarks2: (16, 154) landmarks for video 2
        Returns:
            dictionary with different similarity scores
        """
        results = {}
        
        # Get alignment info for debugging
        _, _, alignment_info = self.align_gesture_sequences(landmarks1, landmarks2)
        results['alignment_info'] = alignment_info
        
        # Overall similarities with alignment-aware methods
        results['overall_dtw_aligned'] = self.calculate_sequence_similarity(
            landmarks1, landmarks2, method='dtw_aligned'
        )
        results['overall_average_aligned'] = self.calculate_sequence_similarity(
            landmarks1, landmarks2, method='average_frames'
        )
        results['overall_weighted_aligned'] = self.calculate_sequence_similarity(
            landmarks1, landmarks2, method='weighted_temporal'
        )
        
        # Component-specific similarities (already use alignment internally)
        results['hand_similarity'] = self.calculate_hand_similarity(landmarks1, landmarks2)
        results['pose_similarity'] = self.calculate_pose_similarity(landmarks1, landmarks2)
        
        # Combined weighted score emphasizing DTW alignment
        results['combined_score'] = (
            0.4 * results['overall_dtw_aligned'] +
            0.3 * results['hand_similarity'] +
            0.2 * results['overall_weighted_aligned'] +
            0.1 * results['pose_similarity']
        )
        
        return results
    
    def compare_gestures(self, video_path1, video_path2, preprocessor):
        """
        Complete pipeline to compare gestures from two videos
        Args:
            video_path1: path to first video
            video_path2: path to second video  
            preprocessor: your preprocessing class instance
        Returns:
            similarity results dictionary
        """
        # Preprocess both videos
        frames1, landmarks1 = preprocessor.preprocess_video(video_path1)
        frames2, landmarks2 = preprocessor.preprocess_video(video_path2)
        
        # Calculate comprehensive similarity
        similarity_results = self.calculate_comprehensive_similarity(landmarks1, landmarks2)
        
        # Add interpretation
        combined_score = similarity_results['combined_score']
        if combined_score >= 0.8:
            interpretation = "Very similar gestures"
        elif combined_score >= 0.6:
            interpretation = "Moderately similar gestures"
        elif combined_score >= 0.4:
            interpretation = "Somewhat similar gestures"
        else:
            interpretation = "Different gestures"
            
        similarity_results['interpretation'] = interpretation
        
        return similarity_results

# Example usage with temporal alignment:
"""
# Initialize the calculator with motion detection parameters
similarity_calculator = GestureSimilarityCalculator(
    motion_threshold=0.02,  # Adjust based on your data
    min_gesture_length=3    # Minimum frames for valid gesture
)

# Method 1: If you already have preprocessed landmarks
landmarks_video1 = ...  # (16, 154) array from your preprocessing
landmarks_video2 = ...  # (16, 154) array from your preprocessing

# Calculate comprehensive similarity with temporal alignment
results = similarity_calculator.calculate_comprehensive_similarity(landmarks_video1, landmarks_video2)

print("Similarity Results:")
for key, value in results.items():
    if key == 'alignment_info':
        print(f"Alignment Info:")
        print(f"  Video 1 gesture frames: {value['original_bounds1']}")
        print(f"  Video 2 gesture frames: {value['original_bounds2']}")
        print(f"  Gesture lengths: {value['gesture_lengths']}")
        print(f"  Length ratio: {value['length_ratio']:.2f}")
    elif key != 'interpretation':
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Method 2: Debug gesture detection
gesture1, bounds1 = similarity_calculator.extract_gesture_segment(landmarks_video1)
gesture2, bounds2 = similarity_calculator.extract_gesture_segment(landmarks_video2)

print(f"\\nGesture Detection:")
print(f"Video 1: gesture from frame {bounds1[0]} to {bounds1[1]} (length: {len(gesture1)})")
print(f"Video 2: gesture from frame {bounds2[0]} to {bounds2[1]} (length: {len(gesture2)})")

# Method 3: Complete pipeline with video paths
# results = similarity_calculator.compare_gestures(
#     "path/to/video1.mp4", 
#     "path/to/video2.mp4", 
#     your_preprocessor_instance
# )
"""



import os
import sys
sys.path.append("/Users/I528933/Desktop/NewIsl-main")
import torch
import numpy as np
import cv2
import argparse
from config import ISLConfig

from enhancedDataPreprocessor import EnhancedDataPreprocessor, EnhancedVideoDataset  # your files
from albumentations.pytorch import ToTensorV2

def augment_video_from_file(video_path, output_dir, config_path=None):
    # Load config
    config = ISLConfig() if config_path is None else ISLConfig(config_path)
    preprocessor = EnhancedDataPreprocessor(config)

    print(f"[INFO] Preprocessing {video_path}")
    frames, landmarks = preprocessor.preprocess_video(video_path)
    print("[DEBUG] Landmarks shape before saving:", landmarks.shape)


    # Dummy manifest for a single sample
    dummy_manifest = [{
        "frame_path": "temp_frames.npy",
        "landmarks_path": "temp_landmarks.npy",
        "encoded_label": 0  # Dummy label
    }]

    # Save temp files for compatibility with Dataset class
    np.save("temp_frames.npy", frames.astype('float32') / 255.0)
    np.save("temp_landmarks.npy", landmarks)

    # Initialize Dataset for applying augmentations
    dataset = EnhancedVideoDataset(dummy_manifest, config, mode='train')
    (augmented_frames, augmented_landmarks), _ = dataset[0]  # Apply __getitem__

    # Save augmented frames to output folder
    os.makedirs(output_dir, exist_ok=True)
    for i in range(augmented_frames.shape[0]):
        frame = augmented_frames[i].permute(1, 2, 0).numpy() * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"aug_frame_{i:03d}.jpg"), frame)

    print(f"[✅] Augmented frames saved to: {output_dir}")

    # Clean up temp files
    os.remove("temp_frames.npy")
    os.remove("temp_landmarks.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="augmented_output", help="Directory to save augmented frames")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config file")
    args = parser.parse_args()

    augment_video_from_file(args.video, args.output, args.config)





write me a code to extract the landmark indices that are being accepted via the detect_gesture_boundaries and use the other visualisation code to help me visualise those images so that I can see which frames are being selected by the code
