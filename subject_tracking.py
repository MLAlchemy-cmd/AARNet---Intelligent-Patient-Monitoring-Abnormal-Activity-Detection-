# subject_tracking.py
import cv2
import numpy as np
import os
import time
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis
from collections import defaultdict

class SubjectTracker:
    def __init__(self, embeddings_file="known_embeddings.npy", names_file="known_names.npy"):
        """
        Initialize the subject tracking system
        
        Args:
            embeddings_file: Path to saved face embeddings
            names_file: Path to saved face names
        """
        print("Initializing Subject Tracking System...")
        
        # Load known embeddings
        try:
            self.known_embeddings = np.load(embeddings_file, allow_pickle=True)
            self.known_names = np.load(names_file, allow_pickle=True)
            print(f"✓ Loaded {len(self.known_embeddings)} known faces")
        except FileNotFoundError:
            print(f"⚠ Embedding files not found: {embeddings_file}, {names_file}")
            print("Please run encode_known_faces.py first")
            raise
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            raise
        
        # Initialize FaceAnalysis (InsightFace)
        try:
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("✓ InsightFace initialized")
        except Exception as e:
            print(f"Error initializing InsightFace: {e}")
            print("Please install insightface: pip install insightface")
            raise
        
        # Initialize YOLO
        try:
            self.model = YOLO("yolov8n.pt")
            print("✓ YOLO model loaded")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
        
        # Initialize DeepSORT
        self.tracker = DeepSort(max_age=30)
        print("✓ DeepSORT tracker initialized")
        
        # Tracking state
        self.id_name_map = {}
        self.track_history = defaultdict(list)
        self.identity_history = defaultdict(list)
        
        # Performance metrics
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Check if display is available
        self.display_available = self._check_display_available()
        
        print("✓ Subject tracking system ready!")
    
    def _check_display_available(self):
        """Check if OpenCV display is available"""
        try:
            # Try to create a test window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("test", test_img)
            cv2.destroyAllWindows()
            return True
        except Exception:
            print("⚠ OpenCV display not available. Running in headless mode.")
            return False
    
    def match_face(self, face_embedding):
        """Match face embedding with known faces"""
        best_match = None
        best_similarity = 0.0
        
        for i, known_emb in enumerate(self.known_embeddings):
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, known_emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_emb))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (self.known_names[i], similarity)
        
        # Return match if similarity is above threshold
        if best_match and best_match[1] > 0.6:
            return best_match[0], best_match[1]
        
        return "Unknown", 0.0
    
    def process_frame(self, frame):
        """Process a single frame for tracking and identification"""
        self.frame_count += 1
        
        # Detect persons using YOLO
        detections = []
        results = self.model(frame)[0]
        
        for box in results.boxes:
            if int(box.cls[0]) != 0:  # Only person class
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Filter by confidence
            if conf > 0.5:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
        
        # Update DeepSORT tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process each track
        current_identities = {}
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            l, t, r, b = map(int, track.to_ltrb())
            track_id = track.track_id
            
            # Extract person crop for face detection
            crop = frame[t:b, l:r]
            
            # Update tracking history
            self.track_history[track_id].append([l, t, r, b])
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
            
            # Identify person if not already identified
            if track_id not in self.id_name_map:
                try:
                    faces = self.face_app.get(crop)
                    if faces:
                        emb = faces[0].embedding
                        name, confidence = self.match_face(emb)
                        self.id_name_map[track_id] = name
                        
                        # Update identity history
                        self.identity_history[track_id].append((name, confidence))
                    else:
                        self.id_name_map[track_id] = "Unknown"
                        self.identity_history[track_id].append(("Unknown", 0.0))
                except Exception as e:
                    print(f"Error in face recognition: {e}")
                    self.id_name_map[track_id] = "Unknown"
                    self.identity_history[track_id].append(("Unknown", 0.0))
            
            # Get current identity
            name = self.id_name_map.get(track_id, "Unknown")
            current_identities[track_id] = name
            
            # Draw bounding box and information
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            # Draw track ID
            cv2.putText(frame, f"ID: {track_id}", (l, t - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw identity
            cv2.putText(frame, f"{name}", (l, t - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw tracking path
            if len(self.track_history[track_id]) > 1:
                points = [(int((p[0] + p[2])/2), int((p[1] + p[3])/2)) for p in self.track_history[track_id]]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        # Calculate FPS
        if self.frame_count % 30 == 0:
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time
        
        return frame, current_identities
    
    def process_video(self, video_path=0, output_path='subject_tracking_output.avi'):
        """Process video file or webcam stream"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting video processing...")
        if self.display_available:
            print("Press 'q' to quit, 's' to save current frame")
        else:
            print("Running in headless mode (no display)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, current_identities = self.process_frame(frame)
            
            # Add FPS and statistics
            cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Tracks: {len(current_identities)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display frame if available
            if self.display_available:
                try:
                    cv2.imshow("Subject Tracking", processed_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"capture_{timestamp}.jpg", processed_frame)
                        print(f"Frame saved as capture_{timestamp}.jpg")
                except Exception as e:
                    print(f"Display error: {e}")
                    self.display_available = False
                    print("Switching to headless mode...")
            else:
                # In headless mode, just show progress
                if self.frame_count % 30 == 0:
                    print(f"Processed {self.frame_count} frames, FPS: {self.fps:.1f}, Tracks: {len(current_identities)}")
        
        # Cleanup
        cap.release()
        out.release()
        if self.display_available:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print(f"Video processing complete. Output saved to {output_path}")
        self._save_tracking_results()
    
    def _save_tracking_results(self):
        """Save tracking results to JSON file"""
        results = {
            'total_frames': self.frame_count,
            'average_fps': self.fps,
            'tracking_history': dict(self.track_history),
            'identity_history': dict(self.identity_history),
            'final_identities': dict(self.id_name_map)
        }
        
        with open('tracking_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Tracking results saved to tracking_results.json")
    
    def get_tracking_summary(self):
        """Get a summary of tracking results"""
        summary = {
            'total_frames_processed': self.frame_count,
            'average_fps': self.fps,
            'total_tracks': len(self.track_history),
            'current_tracks': len(self.id_name_map),
            'identified_subjects': list(set(self.id_name_map.values()))
        }
        return summary

def main():
    """Main function to run the subject tracking system"""
    print("=== Subject Identity and Tracking System ===")
    print("Using YOLO + DeepSORT + InsightFace")
    print()
    
    # Check if embedding files exist
    embeddings_file = "known_embeddings.npy"
    names_file = "known_names.npy"
    
    if not os.path.exists(embeddings_file) or not os.path.exists(names_file):
        print("⚠ Embedding files not found!")
        print("Please run encode_known_faces.py first to create face embeddings.")
        print()
        
        choice = input("Do you want to run face encoding now? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                from encode_known_faces import encode_known_faces
                success = encode_known_faces()
                if not success:
                    return
            except ImportError:
                print("Could not import encode_known_faces.py")
                return
        else:
            return
    
    try:
        # Initialize tracker
        tracker = SubjectTracker(embeddings_file, names_file)
        
        # Print system summary
        summary = tracker.get_tracking_summary()
        print("System Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print()
        
        # Ask user for input source
        print("Choose input source:")
        print("1. Webcam")
        print("2. Video file")
        print("3. Test with sample video")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            print("Starting webcam stream...")
            tracker.process_video(video_path=0)
        elif choice == "2":
            video_path = input("Enter video file path: ").strip()
            if os.path.exists(video_path):
                tracker.process_video(video_path=video_path)
            else:
                print("Error: Video file not found")
        elif choice == "3":
            # Use sample video if available
            sample_videos = [ "V_19.mp4"]
            for video in sample_videos:
                if os.path.exists(video):
                    print(f"Using sample video: {video}")
                    tracker.process_video(video_path=video)
                    break
            else:
                print("No sample video found, starting webcam...")
                tracker.process_video(video_path=0)
        else:
            print("Invalid choice, starting webcam...")
            tracker.process_video(video_path=0)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
