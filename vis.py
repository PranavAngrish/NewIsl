import numpy as np
import cv2
import os
import json
import pickle
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

class EnhancedDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.label_encoder = None  # Initialize your label encoder here
        
    def detect_gesture_boundaries(self, landmarks_sequence, motion_threshold=0.02, min_gesture_length=3):
        """
        Detect the start and end of actual gesture movement
        Args:
            landmarks_sequence: (N, 154) array of landmarks
        Returns:
            (start_frame, end_frame) tuple indicating gesture boundaries
        """
        if len(landmarks_sequence) < 2:
            return 0, len(landmarks_sequence) - 1
            
        # Calculate frame-to-frame motion for hands and upper body
        motion_scores = []
        
        for i in range(1, len(landmarks_sequence)):
            # Focus on hand landmarks (0:126) for motion detection
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
        dynamic_threshold = max(motion_threshold, mean_motion + 0.5 * std_motion)
        
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
        if end_frame - start_frame < min_gesture_length:
            # If detected gesture is too short, expand it
            center = (start_frame + end_frame) // 2
            half_min = min_gesture_length // 2
            start_frame = max(0, center - half_min)
            end_frame = min(len(landmarks_sequence), center + half_min + 1)
        
        return start_frame, min(end_frame, len(landmarks_sequence) - 1)

    def extract_frames_from_range(self, video_path, start_frame_idx, end_frame_idx, target_count):
        """
        Extract specific number of frames from a given range in the video
        Args:
            video_path: path to video file
            start_frame_idx: starting frame index
            end_frame_idx: ending frame index  
            target_count: number of frames to extract (config.SEQUENCE_LENGTH)
        Returns:
            frames, landmarks_sequence
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure frame indices are within bounds
        start_frame_idx = max(0, start_frame_idx)
        end_frame_idx = min(total_frames - 1, end_frame_idx)
        
        # Calculate available frames in the range
        available_frames = end_frame_idx - start_frame_idx + 1
        
        if available_frames <= 0:
            cap.release()
            return self._create_empty_sequence(target_count)
        
        # Determine frame indices to extract from the motion range
        if available_frames >= target_count:
            # If we have enough frames, sample evenly from the range
            frame_indices = np.linspace(start_frame_idx, end_frame_idx, target_count, dtype=int)
        else:
            # If we don't have enough frames, take all available frames and pad later
            frame_indices = np.arange(start_frame_idx, end_frame_idx + 1, dtype=int)
        
        frames, landmarks_sequence = [], []
        frame_idx, extracted_count = 0, 0
        
        # Extract frames
        while cap.isOpened() and extracted_count < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx == frame_indices[extracted_count]:
                frames.append(cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4))
                landmarks_sequence.append(self.extract_landmarks(frame))
                extracted_count += 1
                
            frame_idx += 1
        
        cap.release()
        
        # Pad if necessary
        current_length = len(frames)
        if current_length < target_count:
            pad_needed = target_count - current_length
            if current_length > 0:
                # Pad with last frame
                frames.extend([frames[-1]] * pad_needed)
                landmarks_sequence.extend([landmarks_sequence[-1]] * pad_needed)
            else:
                # Create empty sequence if no frames were extracted
                frames, landmarks_sequence = self._create_empty_sequence(target_count)
        
        return np.array(frames[:target_count]), np.array(landmarks_sequence[:target_count])

    def _create_empty_sequence(self, target_count):
        """Create empty frames and landmarks when no valid data is available"""
        frames = [np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)] * target_count
        landmarks_sequence = [np.zeros(170)] * target_count  # Adjust based on your landmark size
        return frames, landmarks_sequence

    def preprocess_video(self, video_path):
        """
        Enhanced preprocessing that first detects motion boundaries, then extracts frames from motion region
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames == 0:
            print(f"Warning: No frames found in {video_path}")
            return self._create_empty_sequence(self.config.SEQUENCE_LENGTH)
        
        # STEP 1: Initial sampling to detect motion boundaries
        print(f"[DEBUG] Step 1: Initial sampling from {total_frames} total frames")
        initial_frames, initial_landmarks = self._initial_frame_sampling(video_path, total_frames)
        
        if len(initial_frames) == 0:
            print(f"Warning: No frames extracted from {video_path}")
            return self._create_empty_sequence(self.config.SEQUENCE_LENGTH)
        
        # STEP 2: Detect gesture boundaries from initial sampling  
        print(f"[DEBUG] Step 2: Detecting gesture boundaries from {len(initial_landmarks)} frames")
        motion_start_idx, motion_end_idx = self.detect_gesture_boundaries(initial_landmarks)
        
        print(f"[DEBUG] Motion detected from frame {motion_start_idx} to {motion_end_idx} (in sampled sequence)")
        
        # STEP 3: Map back to original video frame indices
        # We need to map from our sampled indices back to original video frame indices
        if total_frames > self.config.SEQUENCE_LENGTH:
            # Recreate the original sampling pattern to map indices
            seg_len = self.config.SEQUENCE_LENGTH // 3
            rem = self.config.SEQUENCE_LENGTH % 3
            begin_indices = np.linspace(0, total_frames//3 - 1, seg_len, dtype=int)
            middle_indices = np.linspace(total_frames//3, 2*total_frames//3 - 1, seg_len, dtype=int)
            end_indices = np.linspace(2*total_frames//3, total_frames - 1, seg_len + rem, dtype=int)
            original_frame_indices = np.unique(np.concatenate([begin_indices, middle_indices, end_indices]))
            
            # Map motion boundaries to original video frame indices
            original_motion_start = original_frame_indices[motion_start_idx]
            original_motion_end = original_frame_indices[min(motion_end_idx, len(original_frame_indices) - 1)]
        else:
            # For short videos, direct mapping
            original_motion_start = motion_start_idx
            original_motion_end = min(motion_end_idx, total_frames - 1)
        
        print(f"[DEBUG] Mapped to original video frames: {original_motion_start} to {original_motion_end}")
        
        # STEP 4: Extract frames specifically from the motion region
        print(f"[DEBUG] Step 4: Extracting {self.config.SEQUENCE_LENGTH} frames from motion region")
        final_frames, final_landmarks = self.extract_frames_from_range(
            video_path, 
            original_motion_start, 
            original_motion_end, 
            self.config.SEQUENCE_LENGTH
        )
        
        print(f"[DEBUG] Final extraction: {len(final_frames)} frames, {len(final_landmarks)} landmark sets")
        
        return final_frames, final_landmarks

    def _initial_frame_sampling(self, video_path, total_frames):
        """
        Perform initial frame sampling using the original method (for motion detection)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Use original sampling logic
        if total_frames > self.config.SEQUENCE_LENGTH:
            seg_len = self.config.SEQUENCE_LENGTH // 3
            rem = self.config.SEQUENCE_LENGTH % 3
            begin_indices = np.linspace(0, total_frames//3 - 1, seg_len, dtype=int)
            middle_indices = np.linspace(total_frames//3, 2*total_frames//3 - 1, seg_len, dtype=int)
            end_indices = np.linspace(2*total_frames//3, total_frames - 1, seg_len + rem, dtype=int)
            frame_indices = np.unique(np.concatenate([begin_indices, middle_indices, end_indices]))
        else:
            frame_indices = list(range(total_frames))
        
        frames, landmarks_sequence = [], []
        frame_idx, extracted_count = 0, 0
        
        while cap.isOpened() and extracted_count < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx == frame_indices[extracted_count]:
                frames.append(cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4))
                landmarks_sequence.append(self.extract_landmarks(frame))
                extracted_count += 1
                
            frame_idx += 1
        
        cap.release()
        
        # Pad to target length if necessary
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
        """
        Enhanced data preprocessing with motion-based frame selection
        """
        print("Starting enhanced data preprocessing with motion detection...")
        manifest, label_set = [], set()
        categories = [d for d in os.listdir(self.config.DATA_PATH) if os.path.isdir(os.path.join(self.config.DATA_PATH, d))]
        sample_index = 0
        
        # Statistics tracking
        motion_stats = {
            'videos_processed': 0,
            'motion_regions_detected': 0,
            'average_motion_percentage': 0,
            'motion_percentages': []
        }
        
        for category in tqdm(categories, desc="Processing categories"):
            category_path = os.path.join(self.config.DATA_PATH, category)
            
            for class_name in tqdm(os.listdir(category_path), desc=f"Processing {category}", leave=False):
                class_path = os.path.join(category_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                for video_file in os.listdir(class_path):
                    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        continue
                        
                    video_path = os.path.join(class_path, video_file)
                    
                    try:
                        print(f"\n[INFO] Processing: {video_path}")
                        frames, landmarks = self.preprocess_video(video_path)
                        
                        # Save processed data
                        video_name = os.path.splitext(video_file)[0]
                        target_dir = os.path.join(self.config.CHUNKED_DATA_DIR, category, class_name, video_name)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        frame_path = os.path.join(target_dir, "frames.npy")
                        landmarks_path = os.path.join(target_dir, "landmarks.npy")
                        
                        np.save(frame_path, frames.astype('float32') / 255.0)
                        np.save(landmarks_path, landmarks)
                        
                        # Update manifest
                        label = f"{category}_{class_name}"
                        label_set.add(label)
                        
                        manifest.append({
                            "index": sample_index,
                            "label": label,
                            "original_video": video_path,
                            "frame_path": frame_path,
                            "landmarks_path": landmarks_path
                        })
                        
                        sample_index += 1
                        motion_stats['videos_processed'] += 1
                        
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")
                        import traceback
                        traceback.print_exc()
        
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
        
        # Save processing statistics
        stats_path = os.path.join(self.config.CHUNKED_DATA_DIR, "motion_detection_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(motion_stats, f, indent=2)
        
        print(f"\n✅ Enhanced preprocessing complete!")
        print(f"   - Processed {motion_stats['videos_processed']} videos")
        print(f"   - Saved {sample_index} samples")
        print(f"   - Found {len(self.label_encoder.classes_)} classes")
        print(f"   - Motion detection applied to all videos")
        print(f"   - Statistics saved to: {stats_path}")
        
        return manifest, list(self.label_encoder.classes_)

    def extract_landmarks(self, frame):
        """
        Placeholder for your landmark extraction method
        Replace this with your actual landmark extraction implementation
        """
        # This should return your landmark features (e.g., 154 or 170 dimensional vector)
        # For now, returning zeros as placeholder
        return np.zeros(154)  # Adjust size based on your landmark format
