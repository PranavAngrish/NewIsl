import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.append("/Users/I528933/Desktop/NewIsl-main")

# Import your existing classes
from config import ISLConfig
from enhancedDataPreprocessor import EnhancedDataPreprocessor

class GestureBoundaryVisualizer:
    def __init__(self, motion_threshold=0.02, min_gesture_length=3):
        self.motion_threshold = motion_threshold
        self.min_gesture_length = min_gesture_length
        
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
        
        return start_frame, min(end_frame, len(landmarks_sequence) - 1), debug_info
    
    def visualize_gesture_detection(self, video_path, output_dir="gesture_analysis", save_frames=True):
        """
        Visualize the gesture detection process and save selected frames
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load config and preprocessor
        config = ISLConfig()
        preprocessor = EnhancedDataPreprocessor(config)
        
        print(f"[INFO] Processing video: {video_path}")
        
        # Preprocess video to get frames and landmarks
        frames, landmarks = preprocessor.preprocess_video(video_path)
        
        print(f"[INFO] Video shape: {frames.shape}")
        print(f"[INFO] Landmarks shape: {landmarks.shape}")
        
        # Detect gesture boundaries with debug info
        start_frame, end_frame, debug_info = self.detect_gesture_boundaries_with_debug(landmarks)
        
        print(f"\n[RESULTS] Gesture Detection Results:")
        print(f"  - Total frames: {len(frames)}")
        print(f"  - Gesture start frame: {start_frame}")
        print(f"  - Gesture end frame: {end_frame}")
        print(f"  - Gesture length: {end_frame - start_frame + 1}")
        print(f"  - Dynamic threshold: {debug_info['dynamic_threshold']:.4f}")
        print(f"  - Mean motion: {debug_info['mean_motion']:.4f}")
        print(f"  - Motion std: {debug_info['std_motion']:.4f}")
        
        # Create motion analysis plot
        self.plot_motion_analysis(debug_info, start_frame, end_frame, 
                                 len(frames), output_dir)
        
        if save_frames:
            # Save all frames with annotations
            self.save_annotated_frames(frames, start_frame, end_frame, output_dir)
            
            # Save only gesture frames
            self.save_gesture_frames(frames, start_frame, end_frame, output_dir)
        
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'total_frames': len(frames),
            'gesture_length': end_frame - start_frame + 1,
            'debug_info': debug_info
        }
    
    def plot_motion_analysis(self, debug_info, start_frame, end_frame, total_frames, output_dir):
        """
        Create a comprehensive motion analysis plot
        """
        motion_scores = debug_info['motion_scores']
        raw_motion = debug_info['raw_motion_scores']
        dynamic_threshold = debug_info['dynamic_threshold']
        static_threshold = debug_info['static_threshold']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Frame indices (motion scores are 1 shorter than total frames)
        frame_indices = np.arange(1, len(motion_scores) + 1)
        
        # Plot 1: Raw vs Smoothed Motion Scores
        ax1.plot(frame_indices, raw_motion, 'lightblue', alpha=0.7, label='Raw Motion', linewidth=1)
        ax1.plot(frame_indices, motion_scores, 'darkblue', linewidth=2, label='Smoothed Motion')
        ax1.axhline(y=dynamic_threshold, color='red', linestyle='--', linewidth=2, label=f'Dynamic Threshold ({dynamic_threshold:.4f})')
        ax1.axhline(y=static_threshold, color='orange', linestyle=':', linewidth=2, label=f'Static Threshold ({static_threshold:.4f})')
        
        # Highlight gesture region
        ax1.axvspan(start_frame, end_frame, alpha=0.3, color='green', label='Detected Gesture')
        
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Motion Score')
        ax1.set_title('Motion Score Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Frame-by-frame analysis with gesture boundaries
        frame_labels = ['Non-Gesture'] * total_frames
        for i in range(start_frame, end_frame + 1):
            if i < len(frame_labels):
                frame_labels[i] = 'Gesture'
        
        colors = ['red' if label == 'Non-Gesture' else 'green' for label in frame_labels]
        ax2.bar(range(total_frames), [1] * total_frames, color=colors, alpha=0.6)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Frame Type')
        ax2.set_title('Frame Classification (Red: Non-Gesture, Green: Gesture)')
        ax2.set_ylim(0, 1.2)
        
        # Add frame numbers on x-axis
        step = max(1, total_frames // 20)  # Show at most 20 labels
        ax2.set_xticks(range(0, total_frames, step))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'motion_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[✅] Motion analysis plot saved to: {os.path.join(output_dir, 'motion_analysis.png')}")
    
    def save_annotated_frames(self, frames, start_frame, end_frame, output_dir):
        """
        Save all frames with annotations showing which are gesture/non-gesture
        """
        annotated_dir = os.path.join(output_dir, "all_frames_annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            # Convert frame to uint8 if needed
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame_img = (frame * 255).astype(np.uint8)
                else:
                    frame_img = frame.astype(np.uint8)
            else:
                frame_img = frame.copy()
            
            # Add annotation
            is_gesture = start_frame <= i <= end_frame
            color = (0, 255, 0) if is_gesture else (0, 0, 255)  # Green for gesture, Red for non-gesture
            text = f"Frame {i}: {'GESTURE' if is_gesture else 'NON-GESTURE'}"
            
            # Add text overlay
            cv2.putText(frame_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2, cv2.LINE_AA)
            
            # Add colored border
            cv2.rectangle(frame_img, (0, 0), (frame_img.shape[1]-1, frame_img.shape[0]-1), 
                         color, 3)
            
            # Save frame
            cv2.imwrite(os.path.join(annotated_dir, f"frame_{i:03d}.jpg"), frame_img)
        
        print(f"[✅] Annotated frames saved to: {annotated_dir}")
    
    def save_gesture_frames(self, frames, start_frame, end_frame, output_dir):
        """
        Save only the frames identified as gesture frames
        """
        gesture_dir = os.path.join(output_dir, "gesture_frames_only")
        os.makedirs(gesture_dir, exist_ok=True)
        
        gesture_count = 0
        for i in range(start_frame, end_frame + 1):
            if i < len(frames):
                frame = frames[i]
                
                # Convert frame to uint8 if needed
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame_img = (frame * 255).astype(np.uint8)
                    else:
                        frame_img = frame.astype(np.uint8)
                else:
                    frame_img = frame.copy()
                
                # Add frame number annotation
                cv2.putText(frame_img, f"Original Frame {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Save frame
                cv2.imwrite(os.path.join(gesture_dir, f"gesture_frame_{gesture_count:03d}_orig_{i:03d}.jpg"), 
                           frame_img)
                gesture_count += 1
        
        print(f"[✅] {gesture_count} gesture frames saved to: {gesture_dir}")
    
    def compare_multiple_videos(self, video_paths, output_dir="multi_video_analysis"):
        """
        Compare gesture detection across multiple videos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, video_path in enumerate(video_paths):
            print(f"\n[INFO] Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            video_output_dir = os.path.join(output_dir, f"video_{i+1}")
            result = self.visualize_gesture_detection(video_path, video_output_dir, save_frames=True)
            result['video_path'] = video_path
            results.append(result)
        
        # Create comparison summary
        self.create_comparison_summary(results, output_dir)
        
        return results
    
    def create_comparison_summary(self, results, output_dir):
        """
        Create a summary comparison of multiple videos
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        video_names = [f"Video {i+1}" for i in range(len(results))]
        
        # Plot 1: Gesture lengths
        gesture_lengths = [r['gesture_length'] for r in results]
        axes[0, 0].bar(video_names, gesture_lengths, color='skyblue')
        axes[0, 0].set_title('Gesture Lengths (frames)')
        axes[0, 0].set_ylabel('Number of Frames')
        
        # Plot 2: Gesture start positions
        start_frames = [r['start_frame'] for r in results]
        axes[0, 1].bar(video_names, start_frames, color='lightcoral')
        axes[0, 1].set_title('Gesture Start Positions')
        axes[0, 1].set_ylabel('Frame Number')
        
        # Plot 3: Total video lengths
        total_frames = [r['total_frames'] for r in results]
        axes[1, 0].bar(video_names, total_frames, color='lightgreen')
        axes[1, 0].set_title('Total Video Lengths')
        axes[1, 0].set_ylabel('Number of Frames')
        
        # Plot 4: Gesture percentage
        gesture_percentages = [(r['gesture_length'] / r['total_frames']) * 100 for r in results]
        axes[1, 1].bar(video_names, gesture_percentages, color='gold')
        axes[1, 1].set_title('Gesture Percentage of Video')
        axes[1, 1].set_ylabel('Percentage (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results to text file
        with open(os.path.join(output_dir, 'detailed_results.txt'), 'w') as f:
            f.write("GESTURE DETECTION COMPARISON RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Video {i+1}: {result['video_path']}\n")
                f.write(f"  Total frames: {result['total_frames']}\n")
                f.write(f"  Gesture start: {result['start_frame']}\n")
                f.write(f"  Gesture end: {result['end_frame']}\n")
                f.write(f"  Gesture length: {result['gesture_length']}\n")
                f.write(f"  Gesture percentage: {(result['gesture_length']/result['total_frames'])*100:.1f}%\n")
                f.write(f"  Dynamic threshold: {result['debug_info']['dynamic_threshold']:.4f}\n")
                f.write(f"  Mean motion: {result['debug_info']['mean_motion']:.4f}\n")
                f.write("-" * 40 + "\n")
        
        print(f"[✅] Comparison summary saved to: {output_dir}")


def main():
    """
    Example usage of the visualization tool
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize gesture boundary detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="gesture_analysis", 
                       help="Output directory for analysis results")
    parser.add_argument("--motion_threshold", type=float, default=0.02, 
                       help="Motion threshold for gesture detection")
    parser.add_argument("--min_gesture_length", type=int, default=3, 
                       help="Minimum gesture length in frames")
    parser.add_argument("--no_frames", action="store_true", 
                       help="Skip saving individual frames (only create plots)")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = GestureBoundaryVisualizer(
        motion_threshold=args.motion_threshold,
        min_gesture_length=args.min_gesture_length
    )
    
    # Run visualization
    results = visualizer.visualize_gesture_detection(
        args.video, 
        args.output, 
        save_frames=not args.no_frames
    )
    
    print(f"\n[SUMMARY] Gesture Detection Complete!")
    print(f"  Results saved to: {args.output}")
    print(f"  Gesture frames: {results['start_frame']} to {results['end_frame']}")
    print(f"  Total gesture length: {results['gesture_length']} frames")


if __name__ == "__main__":
    main()
