import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, LSTM, Dense, Dropout, TimeDistributed, Flatten, GlobalAveragePooling3D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Remove mixed precision since you don't have a GPU
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

class ViolenceDetectionTrainer:
    def __init__(self, data_dir, sequence_length=8, frame_size=(160, 160)):
        self.data_dir = os.path.normpath(data_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.batch_size = 16
        self.action_classes = None
        self.train_generator = None
        self.val_generator = None
        
    def load_action_classes(self):
        """Load action classes and their violence status from CSV"""
        csv_path = os.path.join(self.data_dir, 'action-class-occurrences.csv')
        df = pd.read_csv(csv_path, sep=';')
        self.action_classes = {
            row['ACTION CLASS']: row['VIOLENT'] == 'y'
            for _, row in df.iterrows()
        }
        print(f"Loaded {len(self.action_classes)} action classes")
        return self.action_classes
    
    def load_video_data(self):
        """Load video paths and labels from the dataset"""
        # Load action classes first
        if self.action_classes is None:
            self.load_action_classes()
        
        violent_dir = os.path.join(self.data_dir, 'violent')
        non_violent_dir = os.path.join(self.data_dir, 'non-violent')
        
        print(f"Looking for videos in:")
        print(f"Violent directory: {violent_dir}")
        print(f"Non-violent directory: {non_violent_dir}")
        
        # Initialize lists
        all_videos = []
        all_labels = []
        
        # Process violent videos
        if os.path.exists(violent_dir):
            # Process cam1 and cam2 folders
            for cam_folder in ['cam1', 'cam2']:
                cam_path = os.path.join(violent_dir, cam_folder)
                if os.path.exists(cam_path):
                    videos = [os.path.join(cam_path, f) for f in os.listdir(cam_path) 
                            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                    all_videos.extend(videos)
                    all_labels.extend([1] * len(videos))
                    print(f"Found {len(videos)} videos in violent/{cam_folder}")
        
        # Process non-violent videos
        if os.path.exists(non_violent_dir):
            # Process cam1 and cam2 folders
            for cam_folder in ['cam1', 'cam2']:
                cam_path = os.path.join(non_violent_dir, cam_folder)
                if os.path.exists(cam_path):
                    videos = [os.path.join(cam_path, f) for f in os.listdir(cam_path) 
                            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                    all_videos.extend(videos)
                    all_labels.extend([0] * len(videos))
                    print(f"Found {len(videos)} videos in non-violent/{cam_folder}")
        
        if not all_videos:
            raise ValueError("No videos found in the specified directories!")
        
        print(f"Total videos for training: {len(all_videos)}")
        print(f"Violent videos: {sum(all_labels)}")
        print(f"Non-violent videos: {len(all_labels) - sum(all_labels)}")
        
        return all_videos, all_labels
    
    def extract_frames(self, video_path):
        """Extract frames from video with efficient sampling"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return None
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate sampling interval
            if total_frames > self.sequence_length:
                interval = total_frames // self.sequence_length
            else:
                interval = 1
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % interval == 0:
                    # Resize and preprocess frame
                    frame = cv2.resize(frame, self.frame_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame / 255.0
                    frames.append(frame)
                    
                    if len(frames) == self.sequence_length:
                        break
                
                frame_count += 1
            
            cap.release()
            
            # If no frames were extracted, return None
            if not frames:
                print(f"Warning: No frames extracted from {video_path}")
                return None
            
            # Pad sequence if necessary
            while len(frames) < self.sequence_length:
                frames.append(np.zeros_like(frames[0]))
                
            return np.array(frames)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None
    
    def data_generator(self, video_paths, labels, batch_size):
        """Efficient data generator for training"""
        num_samples = len(video_paths)
        indices = np.arange(num_samples)
        
        while True:
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_videos = [video_paths[j] for j in batch_indices]
                batch_labels = [labels[j] for j in batch_indices]
                
                # Process videos in parallel
                batch_frames = []
                valid_labels = []
                
                for video_path, label in zip(batch_videos, batch_labels):
                    frames = self.extract_frames(video_path)
                    if frames is not None:  # Only add if frames were successfully extracted
                        batch_frames.append(frames)
                        valid_labels.append(label)
                
                if batch_frames:  # Only yield if we have valid frames
                    yield np.array(batch_frames), np.array(valid_labels)
    
    def create_model(self):
        """Create a lightweight hybrid CNN-LSTM model"""
        # Input layer
        input_layer = Input(shape=(self.sequence_length, *self.frame_size, 3))
        
        # Use MobileNetV2 as feature extractor
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        base_model.trainable = False  # Freeze the base model
        
        # Process each frame through MobileNetV2
        x = TimeDistributed(base_model)(input_layer)
        
        # Simplified LSTM layer
        x = LSTM(32)(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        output_layer = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    '''def train(self, epochs=5):
        """Train the model with efficient data loading"""
        # Load and split data
        video_paths, labels = self.load_video_data()
        train_videos, val_videos, train_labels, val_labels = train_test_split(
            video_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create data generators
        self.train_generator = self.data_generator(train_videos, train_labels, self.batch_size)
        self.val_generator = self.data_generator(val_videos, val_labels, self.batch_size)
        
        # Calculate steps per epoch
        train_steps = len(train_videos) // self.batch_size
        val_steps = len(val_videos) // self.batch_size
        
        # Create and train model
        model = self.create_model()
        
        # Add callbacks for efficient training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=1
            )
        ]
        
        # Train the model
        history = model.fit(
            self.train_generator,
            steps_per_epoch=train_steps,
            validation_data=self.val_generator,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Save the model
        model.save('violence_detection_model.h5')
        return history

def main():
    # Initialize trainer with proper path
    data_dir = os.path.join(os.getcwd(), "violence-detection-dataset")
    print(f"Using data directory: {data_dir}")
    
    trainer = ViolenceDetectionTrainer(
        data_dir=data_dir,
        sequence_length=8,
        frame_size=(160, 160)
    )
    
    # Train model
    history = trainer.train(epochs=5)
    
    # Save training history
    pd.DataFrame(history.history).to_csv('training_history.csv')

if __name__ == "__main__":
    main() '''