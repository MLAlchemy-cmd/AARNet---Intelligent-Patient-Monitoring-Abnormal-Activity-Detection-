# save as encode_known_faces.py
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def encode_known_faces(folder="Lekha", output_prefix="known"):
    """
    Encode known faces from a folder and save embeddings
    
    Args:
        folder: Directory containing face images
        output_prefix: Prefix for output files
    """
    print("Initializing InsightFace...")
    try:
        app = FaceAnalysis(name='buffalo_l')  # Good quality model
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"Error initializing InsightFace: {e}")
        print("Please install insightface: pip install insightface")
        return False
    
    known_embeddings = []
    known_names = []
    
    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' not found!")
        return False
    
    # Get all image files
    image_files = [f for f in os.listdir(folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_files) == 0:
        print(f"No image files found in '{folder}'!")
        return False
    
    print(f"Found {len(image_files)} images in '{folder}'")
    
    successful_encodings = 0
    
    for img_name in sorted(image_files):
        img_path = os.path.join(folder, img_name)
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠ Could not load image: {img_name}")
                continue
            
            # Detect faces
            faces = app.get(img)
            
            if len(faces) == 0:
                print(f"⚠ No faces detected in: {img_name}")
                continue
            
            # Use the first face found
            face_embedding = faces[0].embedding
            
            # Extract name from filename (remove extension)
            name = os.path.splitext(img_name)[0]
            
            known_embeddings.append(face_embedding)
            known_names.append(name)
            
            successful_encodings += 1
            print(f"✓ Encoded face: {img_name} -> {name}")
            
        except Exception as e:
            print(f"⚠ Error processing {img_name}: {e}")
            continue
    
    if successful_encodings == 0:
        print("No faces were successfully encoded!")
        return False
    
    # Save to .npy files
    try:
        embeddings_file = f"{output_prefix}_embeddings.npy"
        names_file = f"{output_prefix}_names.npy"
        
        np.save(embeddings_file, known_embeddings)
        np.save(names_file, known_names)
        
        print(f"\n✓ Successfully encoded {successful_encodings} faces")
        print(f"✓ Saved embeddings to: {embeddings_file}")
        print(f"✓ Saved names to: {names_file}")
        
        # Print summary
        print(f"\nEncoded faces:")
        for i, name in enumerate(known_names):
            print(f"  {i+1}. {name}")
        
        return True
        
    except Exception as e:
        print(f"Error saving files: {e}")
        return False

def main():
    """Main function to run face encoding"""
    print("=== Face Encoding System ===")
    print("This script encodes faces from the 'Lekha' folder using InsightFace")
    print()
    
    # Ask user for folder path
    folder = input("Enter folder path (default: 'Lekha'): ").strip()
    if not folder:
        folder = "Lekha"
    
    # Ask user for output prefix
    output_prefix = input("Enter output prefix (default: 'known'): ").strip()
    if not output_prefix:
        output_prefix = "known"
    
    print(f"\nProcessing folder: {folder}")
    print(f"Output prefix: {output_prefix}")
    print()
    
    # Encode faces
    success = encode_known_faces(folder, output_prefix)
    
    if success:
        print("\n✓ Face encoding completed successfully!")
        print("You can now run subject_tracking.py")
    else:
        print("\n✗ Face encoding failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
