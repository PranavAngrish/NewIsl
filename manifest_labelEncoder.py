import os
import json
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from config import ISLConfig

def regenerate_manifest_from_processed_data(config: ISLConfig):
    """
    Regenerate manifest.json from existing processed data in chunked_data directory
    """
    print("Regenerating manifest from existing processed data...")
    
    manifest = []
    label_set = set()
    sample_index = 0
    
    # Walk through the chunked_data directory
    for root, dirs, files in os.walk(config.CHUNKED_DATA_DIR):
        if "frames.npy" in files and "landmarks.npy" in files:
            # Extract category and class from path
            rel_path = os.path.relpath(root, config.CHUNKED_DATA_DIR)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 3:  # category/class/video_name
                category = path_parts[0]
                class_name = path_parts[1]
                video_name = path_parts[2]
                
                # Create label
                label = f"{category}_{class_name}"
                label_set.add(label)
                
                # Paths
                frame_path = os.path.join(root, "frames.npy")
                landmarks_path = os.path.join(root, "landmarks.npy")
                
                # Load data to get shapes
                try:
                    frames = np.load(frame_path)
                    landmarks = np.load(landmarks_path)
                    
                    # Create manifest entry
                    sample_meta = {
                        "index": sample_index,
                        "label": label,
                        "original_video": f"reconstructed_path_{video_name}",  # We don't have the original path
                        "frame_path": frame_path,
                        "landmarks_path": landmarks_path,
                        "sequence_length": config.SEQUENCE_LENGTH,
                        "frame_shape": list(frames.shape),
                        "landmarks_shape": list(landmarks.shape)
                    }
                    
                    manifest.append(sample_meta)
                    sample_index += 1
                    
                    print(f"Added: {label} - {video_name}")
                    
                except Exception as e:
                    print(f"Error loading {frame_path}: {e}")
                    continue
    
    if not manifest:
        print("❌ No valid processed data found!")
        return None, None
    
    # Create and fit label encoder
    label_encoder = LabelEncoder()
    labels = [entry["label"] for entry in manifest]
    y_encoded = label_encoder.fit_transform(labels)
    class_names = list(label_encoder.classes_)
    
    # Add encoded labels to manifest
    for i, entry in enumerate(manifest):
        entry["encoded_label"] = int(y_encoded[i])
        
        # Update metadata.json in each directory
        metadata_path = os.path.join(os.path.dirname(entry["frame_path"]), "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(entry, f, indent=2)
    
    # Save manifest
    manifest_path = os.path.join(config.CHUNKED_DATA_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save label encoder
    label_encoder_path = os.path.join(config.CHUNKED_DATA_DIR, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"✅ Regenerated manifest with {len(manifest)} samples")
    print(f"🔖 Found {len(class_names)} classes: {class_names}")
    print(f"📁 Saved to: {manifest_path}")
    
    return manifest, class_names

def debug_data_structure(config: ISLConfig):
    """
    Debug function to understand the current data structure
    """
    print("Debugging data structure...")
    print(f"Checking directory: {config.CHUNKED_DATA_DIR}")
    
    if not os.path.exists(config.CHUNKED_DATA_DIR):
        print(f"❌ Directory does not exist: {config.CHUNKED_DATA_DIR}")
        return
    
    # Walk through directories and show structure
    for root, dirs, files in os.walk(config.CHUNKED_DATA_DIR):
        level = root.replace(config.CHUNKED_DATA_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.npy') or file.endswith('.json'):
                file_path = os.path.join(root, file)
                if file.endswith('.npy'):
                    try:
                        data = np.load(file_path)
                        print(f"{subindent}{file} - Shape: {data.shape}")
                    except:
                        print(f"{subindent}{file} - Error loading")
                else:
                    print(f"{subindent}{file}")

def fix_preprocessing_issues(config: ISLConfig):
    """
    Fix common issues with the preprocessing pipeline
    """
    print("Checking for common preprocessing issues...")
    
    # Check if original data directory exists
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ Original data directory not found: {config.DATA_PATH}")
        return False
    
    # Check if chunked data directory exists
    if not os.path.exists(config.CHUNKED_DATA_DIR):
        print(f"❌ Chunked data directory not found: {config.CHUNKED_DATA_DIR}")
        os.makedirs(config.CHUNKED_DATA_DIR, exist_ok=True)
        print(f"✅ Created directory: {config.CHUNKED_DATA_DIR}")
    
    # Check manifest file
    manifest_path = os.path.join(config.CHUNKED_DATA_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            try:
                manifest = json.load(f)
                print(f"📋 Current manifest has {len(manifest)} entries")
                if len(manifest) == 0:
                    print("❌ Manifest is empty!")
                    return False
                else:
                    print("✅ Manifest exists and has data")
                    return True
            except json.JSONDecodeError:
                print("❌ Manifest file is corrupted!")
                return False
    else:
        print("❌ Manifest file does not exist!")
        return False

def main():
    """
    Main function to fix manifest issues
    """
    config = ISLConfig()
    
    print("=" * 60)
    print("MANIFEST RECOVERY TOOL")
    print("=" * 60)
    
    # Debug current structure
    debug_data_structure(config)
    
    # Check for issues
    if not fix_preprocessing_issues(config):
        print("\n" + "=" * 60)
        print("ATTEMPTING TO REGENERATE MANIFEST...")
        print("=" * 60)
        
        # Try to regenerate from existing processed data
        manifest, class_names = regenerate_manifest_from_processed_data(config)
        
        if manifest:
            print("✅ Successfully regenerated manifest!")
            print(f"📊 Statistics:")
            print(f"   - Total samples: {len(manifest)}")
            print(f"   - Total classes: {len(class_names)}")
            print(f"   - Classes: {class_names}")
        else:
            print("❌ Failed to regenerate manifest!")
            print("You may need to run the full preprocessing pipeline again.")
    else:
        print("✅ Manifest appears to be working correctly!")

if __name__ == "__main__":
    main()