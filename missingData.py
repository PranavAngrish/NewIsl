def extract_frames_from_timestamp_range(self, video_path, start_timestamp, end_timestamp, target_count=16):
    """
    Enhanced frame extraction with smart frame selection and interpolation
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
    print(f"[DEBUG] Smart extraction from {start_timestamp:.2f}s to {end_timestamp:.2f}s")
    
    # Ensure timestamps are within bounds
    start_timestamp = max(0.0, start_timestamp)
    end_timestamp = min(total_duration, end_timestamp)
    
    gesture_duration = end_timestamp - start_timestamp
    
    if gesture_duration <= 0:
        cap.release()
        return self._create_empty_sequence(target_count)
    
    # STEP 1: Sample more frames than needed for smart selection
    sample_factor = 2.5  # Sample 2.5x more frames for selection
    candidate_count = min(int(target_count * sample_factor), int(gesture_duration * fps))
    candidate_count = max(candidate_count, target_count)  # Ensure we have at least target_count
    
    print(f"[DEBUG] Sampling {candidate_count} candidate frames for smart selection")
    
    # Extract candidate frames with their quality metrics
    candidates = []
    
    # Calculate time interval for candidate sampling
    if candidate_count > 1:
        time_interval = gesture_duration / (candidate_count - 1)
    else:
        time_interval = 0
    
    for i in range(candidate_count):
        current_timestamp = start_timestamp + (i * time_interval)
        if current_timestamp > end_timestamp:
            current_timestamp = end_timestamp
        
        cap.set(cv2.CAP_PROP_POS_MSEC, current_timestamp * 1000)
        ret, frame = cap.read()
        
        if ret:
            # Resize frame and extract landmarks
            resized_frame = cv2.resize(frame, self.config.IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
            landmarks = self.extract_landmarks(frame)
            
            # Calculate frame quality metrics
            quality_score = self._calculate_frame_quality(resized_frame, landmarks)
            
            candidates.append({
                'frame': resized_frame,
                'landmarks': landmarks,
                'timestamp': current_timestamp,
                'quality_score': quality_score,
                'frame_index': i
            })
            
            print(f"[DEBUG] Candidate {i+1}/{candidate_count}: timestamp={current_timestamp:.2f}s, quality={quality_score:.3f}")
    
    cap.release()
    
    if len(candidates) == 0:
        return self._create_empty_sequence(target_count)
    
    # STEP 2: Smart frame selection
    selected_frames = self._smart_frame_selection(candidates, target_count)
    
    # STEP 3: Sort selected frames by timestamp to maintain temporal order
    selected_frames.sort(key=lambda x: x['timestamp'])
    
    # STEP 4: Extract frames and landmarks, apply interpolation if needed
    final_frames = []
    final_landmarks = []
    
    for i, frame_data in enumerate(selected_frames):
        final_frames.append(frame_data['frame'])
        final_landmarks.append(frame_data['landmarks'])
    
    # STEP 5: Apply interpolation for missing/poor quality landmarks
    final_landmarks = self._interpolate_missing_landmarks(final_landmarks)
    
    print(f"[DEBUG] Smart selection completed: {len(final_frames)} frames selected")
    
    return np.array(final_frames), np.array(final_landmarks)


def _calculate_frame_quality(self, frame, landmarks):
    """
    Calculate a quality score for a frame based on multiple factors
    Args:
        frame: input frame (H, W, 3)
        landmarks: extracted landmarks (154,)
    Returns:
        quality_score: float between 0 and 1 (higher is better)
    """
    quality_factors = []
    
    # Factor 1: Landmark detection quality (30% weight)
    landmark_quality = self._assess_landmark_quality(landmarks)
    quality_factors.append(('landmarks', landmark_quality, 0.30))
    
    # Factor 2: Motion/activity level (25% weight)
    motion_score = self._assess_motion_content(landmarks)
    quality_factors.append(('motion', motion_score, 0.25))
    
    # Factor 3: Image quality (20% weight)
    image_quality = self._assess_image_quality(frame)
    quality_factors.append(('image', image_quality, 0.20))
    
    # Factor 4: Hand visibility and pose (15% weight)
    hand_visibility = self._assess_hand_visibility(landmarks)
    quality_factors.append(('hands', hand_visibility, 0.15))
    
    # Factor 5: Temporal diversity (10% weight) - will be calculated later
    temporal_diversity = 0.5  # Placeholder, will be updated in smart selection
    quality_factors.append(('temporal', temporal_diversity, 0.10))
    
    # Calculate weighted quality score
    total_score = sum(score * weight for _, score, weight in quality_factors)
    
    return min(max(total_score, 0.0), 1.0)  # Clamp between 0 and 1


def _assess_landmark_quality(self, landmarks):
    """Assess the quality of landmark detection"""
    if landmarks is None or len(landmarks) == 0:
        return 0.0
    
    # Check what percentage of landmarks are detected (non-zero)
    hand1_landmarks = landmarks[:63].reshape(21, 3)
    hand2_landmarks = landmarks[63:126].reshape(21, 3)
    pose_landmarks = landmarks[126:].reshape(7, 4)
    
    # Count detected landmarks
    hand1_detected = np.sum(np.any(hand1_landmarks != 0, axis=1))
    hand2_detected = np.sum(np.any(hand2_landmarks != 0, axis=1))
    pose_detected = np.sum(np.any(pose_landmarks[:, :3] != 0, axis=1))
    
    # Calculate detection rates
    hand1_rate = hand1_detected / 21
    hand2_rate = hand2_detected / 21
    pose_rate = pose_detected / 7
    
    # Weighted combination (hands are more important for sign language)
    landmark_score = (hand1_rate * 0.4 + hand2_rate * 0.4 + pose_rate * 0.2)
    
    return landmark_score


def _assess_motion_content(self, landmarks):
    """Assess the motion/activity content in landmarks"""
    if landmarks is None or len(landmarks) == 0:
        return 0.0
    
    # For a single frame, we assess the "expressiveness" of the pose
    hand1_landmarks = landmarks[:63].reshape(21, 3)
    hand2_landmarks = landmarks[63:126].reshape(21, 3)
    
    motion_indicators = []
    
    # Check hand spread (fingers extended vs closed)
    for hand_lm in [hand1_landmarks, hand2_landmarks]:
        if np.any(hand_lm):
            # Calculate distances between fingertips and palm center
            if np.any(hand_lm[0]):  # Wrist detected
                distances = []
                fingertips = [4, 8, 12, 16, 20]  # Fingertip indices
                for tip_idx in fingertips:
                    if np.any(hand_lm[tip_idx]):
                        dist = np.linalg.norm(hand_lm[tip_idx] - hand_lm[0])
                        distances.append(dist)
                
                if distances:
                    # Higher variance in distances indicates more expressive hand pose
                    motion_indicators.append(np.std(distances))
    
    if motion_indicators:
        return min(np.mean(motion_indicators) * 10, 1.0)  # Scale and clamp
    else:
        return 0.0


def _assess_image_quality(self, frame):
    """Assess the technical quality of the image"""
    if frame is None or frame.size == 0:
        return 0.0
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Factor 1: Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize
    
    # Factor 2: Contrast (standard deviation of pixel intensities)
    contrast_score = min(np.std(gray) / 50, 1.0)  # Normalize
    
    # Factor 3: Brightness appropriateness (not too dark or too bright)
    mean_brightness = np.mean(gray)
    brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
    
    # Combine factors
    image_quality = (sharpness_score * 0.4 + contrast_score * 0.4 + brightness_score * 0.2)
    
    return image_quality


def _assess_hand_visibility(self, landmarks):
    """Assess how well hands are visible and positioned"""
    if landmarks is None or len(landmarks) == 0:
        return 0.0
    
    hand1_landmarks = landmarks[:63].reshape(21, 3)
    hand2_landmarks = landmarks[63:126].reshape(21, 3)
    
    visibility_scores = []
    
    for hand_lm in [hand1_landmarks, hand2_landmarks]:
        if np.any(hand_lm):
            # Check if key landmarks are detected
            key_points = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
            detected_key_points = sum(1 for idx in key_points if np.any(hand_lm[idx]))
            visibility_score = detected_key_points / len(key_points)
            
            # Bonus for hand being in good position (not at edges, reasonable size)
            if np.any(hand_lm[0]):  # Wrist detected
                wrist_pos = hand_lm[0][:2]
                # Check if hand is reasonably centered (not at extreme edges)
                center_bonus = 1.0 - max(abs(wrist_pos[0] - 0.5), abs(wrist_pos[1] - 0.5))
                visibility_score = (visibility_score + center_bonus * 0.3) / 1.3
            
            visibility_scores.append(visibility_score)
    
    if visibility_scores:
        return np.mean(visibility_scores)
    else:
        return 0.0


def _smart_frame_selection(self, candidates, target_count):
    """
    Select the best frames using a combination of quality and temporal diversity
    """
    if len(candidates) <= target_count:
        return candidates
    
    # Update temporal diversity scores
    self._update_temporal_diversity_scores(candidates)
    
    # STRATEGY: Combination of quality-based and uniform temporal selection
    
    # Method 1: Select top quality frames (60% of target)
    quality_count = int(target_count * 0.6)
    
    # Sort by quality score
    quality_sorted = sorted(candidates, key=lambda x: x['quality_score'], reverse=True)
    quality_selected = quality_sorted[:quality_count]
    
    # Method 2: Ensure temporal coverage (40% of target)
    temporal_count = target_count - quality_count
    temporal_selected = self._select_temporally_diverse_frames(
        candidates, temporal_count, exclude=quality_selected
    )
    
    # Combine selections
    final_selection = quality_selected + temporal_selected
    
    # If we have duplicates, fill with next best quality frames
    seen_indices = set(frame['frame_index'] for frame in final_selection)
    remaining_candidates = [c for c in quality_sorted if c['frame_index'] not in seen_indices]
    
    while len(final_selection) < target_count and remaining_candidates:
        final_selection.append(remaining_candidates.pop(0))
    
    print(f"[DEBUG] Frame selection: {quality_count} quality-based, {temporal_count} temporal-based")
    
    return final_selection[:target_count]


def _update_temporal_diversity_scores(self, candidates):
    """Update temporal diversity scores for all candidates"""
    if len(candidates) <= 1:
        return
    
    timestamps = [c['timestamp'] for c in candidates]
    
    for i, candidate in enumerate(candidates):
        # Calculate how evenly this frame is spaced from others
        current_time = candidate['timestamp']
        
        # Find distances to other frames
        distances = [abs(current_time - t) for j, t in enumerate(timestamps) if j != i]
        
        if distances:
            # Reward frames that are well-spaced from others
            min_distance = min(distances)
            avg_distance = np.mean(distances)
            
            # Normalize by total duration
            total_duration = max(timestamps) - min(timestamps)
            if total_duration > 0:
                temporal_score = min(min_distance / (total_duration / len(candidates)), 1.0)
            else:
                temporal_score = 0.5
        else:
            temporal_score = 0.5
        
        # Update the candidate's quality score to include temporal diversity
        original_quality = candidate['quality_score']
        candidate['quality_score'] = original_quality * 0.9 + temporal_score * 0.1


def _select_temporally_diverse_frames(self, candidates, count, exclude=None):
    """Select frames that are temporally well-distributed"""
    if exclude is None:
        exclude = []
    
    exclude_indices = set(frame['frame_index'] for frame in exclude)
    available_candidates = [c for c in candidates if c['frame_index'] not in exclude_indices]
    
    if len(available_candidates) <= count:
        return available_candidates
    
    # Sort by timestamp
    available_candidates.sort(key=lambda x: x['timestamp'])
    
    if count == 1:
        # Return the middle frame
        mid_idx = len(available_candidates) // 2
        return [available_candidates[mid_idx]]
    
    # Select evenly spaced frames
    selected = []
    step = len(available_candidates) / count
    
    for i in range(count):
        idx = int(i * step)
        idx = min(idx, len(available_candidates) - 1)
        selected.append(available_candidates[idx])
    
    return selected


def _interpolate_missing_landmarks(self, landmarks_sequence):
    """
    Interpolate missing or poor quality landmarks using temporal information
    Args:
        landmarks_sequence: List of landmark arrays, each of shape (154,)
    Returns:
        interpolated_landmarks: numpy array of shape (T, 154)
    """
    landmarks_array = np.array(landmarks_sequence)  # Shape: (T, 154)
    T, num_landmarks = landmarks_array.shape
    
    print(f"[DEBUG] Applying landmark interpolation for {T} frames")
    
    # Process each landmark coordinate separately
    for landmark_idx in range(num_landmarks):
        landmark_values = landmarks_array[:, landmark_idx]
        
        # Find missing values (zeros or very small values that indicate poor detection)
        missing_mask = np.abs(landmark_values) < 1e-6
        
        if np.all(missing_mask):
            # If all values are missing, keep as zeros
            continue
        elif np.any(missing_mask) and not np.all(missing_mask):
            # Interpolate missing values
            valid_indices = np.where(~missing_mask)[0]
            valid_values = landmark_values[valid_indices]
            
            if len(valid_indices) >= 2:
                # Use linear interpolation for missing values
                interpolated_values = np.interp(
                    range(T), 
                    valid_indices, 
                    valid_values
                )
                landmarks_array[:, landmark_idx] = interpolated_values
            elif len(valid_indices) == 1:
                # If only one valid value, propagate it
                landmarks_array[:, landmark_idx] = valid_values[0]
    
    # Additional smoothing for hand landmarks to reduce jitter
    landmarks_array = self._smooth_landmarks_temporal(landmarks_array)
    
    print(f"[DEBUG] Landmark interpolation completed")
    
    return landmarks_array


def _smooth_landmarks_temporal(self, landmarks_array):
    """
    Apply temporal smoothing to reduce jitter in landmarks
    """
    smoothed = landmarks_array.copy()
    
    # Apply gentle smoothing only to hand landmarks (they tend to be more jittery)
    # Hand 1: landmarks 0:63, Hand 2: landmarks 63:126
    for start_idx in [0, 63]:
        end_idx = start_idx + 63
        hand_landmarks = smoothed[:, start_idx:end_idx]
        
        # Apply Gaussian filter along time dimension for each landmark coordinate
        for coord_idx in range(hand_landmarks.shape[1]):
            if np.any(hand_landmarks[:, coord_idx]):  # Only smooth non-zero sequences
                smoothed[:, start_idx + coord_idx] = gaussian_filter1d(
                    hand_landmarks[:, coord_idx], 
                    sigma=0.8  # Gentle smoothing
                )
    
    return smoothed
