# src/data/preprocessor.py
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import requests
from PIL import Image
import io
import cv2
import mediapipe as mp
from sklearn.feature_extraction.text import TfidfVectorizer

class ExerciseDataPreprocessor:
    """
    Class to preprocess exercise data for machine learning
    """
    def __init__(self, 
                 raw_data_dir: str = "data/raw", 
                 processed_data_dir: str = "data/processed"):
        """
        Initialize preprocessor
        
        Args:
            raw_data_dir: Directory containing raw data
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.exercisedb_dir = f"{raw_data_dir}/exercisedb"
        self.wger_dir = f"{raw_data_dir}/wger"
        
        os.makedirs(processed_data_dir, exist_ok=True)
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from raw data directories
        
        Returns:
            Dictionary of DataFrames
        """
        exercisedb_path = f"{self.exercisedb_dir}/all_exercises.csv"
        wger_path = f"{self.wger_dir}/all_exercises.csv"
        
        data = {}
        
        if os.path.exists(exercisedb_path):
            data["exercisedb"] = pd.read_csv(exercisedb_path)
        
        if os.path.exists(wger_path):
            data["wger"] = pd.read_csv(wger_path)
            
        return data
    
    def clean_exercisedb_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess ExerciseDB data
        
        Args:
            df: ExerciseDB DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Make column names lowercase and replace spaces with underscores
        cleaned_df.columns = [col.lower().replace(' ', '_') for col in cleaned_df.columns]
        
        # Convert all string columns to lowercase
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].str.lower()
        
        # Extract movement patterns from exercise names and descriptions
        cleaned_df['movement_pattern'] = cleaned_df.apply(
            lambda row: self._extract_movement_pattern(row['name'], row['instructions']), 
            axis=1
        )
        
        # Extract joints used from descriptions
        cleaned_df['joints_used'] = cleaned_df.apply(
            lambda row: self._extract_joints_used(row['instructions']), 
            axis=1
        )
        
        # Add placeholders for the additional classifications we'll need
        cleaned_df['intensity_level'] = 'medium'  # Default value, to be refined
        cleaned_df['exercise_type'] = 'unknown'   # To be classified
        cleaned_df['movement_quality'] = 3        # Scale of 1-5, default middle value
        cleaned_df['risk_assessment'] = 'medium'  # Default value, to be refined
        
        # Extract compound vs isolation
        cleaned_df['is_compound'] = cleaned_df.apply(
            lambda row: self._is_compound_exercise(row['bodypart'], row['target'], row['instructions']),
            axis=1
        )
        
        return cleaned_df
    
    def _extract_movement_pattern(self, name: str, instructions: str) -> str:
        """
        Extract movement pattern from exercise name and instructions
        
        This is a simplified version and would need to be enhanced with a more 
        sophisticated NLP approach or expert labeling
        
        Args:
            name: Exercise name
            instructions: Exercise instructions
            
        Returns:
            Movement pattern category
        """
        name = name.lower()
        instructions = instructions.lower() if isinstance(instructions, str) else ""
        text = name + " " + instructions
        
        # Basic pattern matching
        if any(word in text for word in ['squat', 'lunge', 'step']):
            return 'squat'
        elif any(word in text for word in ['hinge', 'deadlift', 'hip thrust']):
            return 'hinge'
        elif any(word in text for word in ['push', 'press', 'bench']):
            return 'push'
        elif any(word in text for word in ['pull', 'row', 'chin', 'pull-up']):
            return 'pull'
        elif any(word in text for word in ['rotation', 'twist', 'turn']):
            return 'rotation'
        elif any(word in text for word in ['carry', 'walk', 'farmer']):
            return 'carry'
        elif any(word in text for word in ['plank', 'bridge', 'hold', 'isometric']):
            return 'isometric'
        else:
            return 'other'
    
    def _extract_joints_used(self, instructions: str) -> List[str]:
        """
        Extract joints used from instructions
        
        Args:
            instructions: Exercise instructions
            
        Returns:
            List of joints used
        """
        if not isinstance(instructions, str):
            return []
            
        instructions = instructions.lower()
        
        joints = []
        joint_keywords = {
            'shoulder': ['shoulder', 'deltoid', 'rotator cuff'],
            'elbow': ['elbow', 'bicep', 'tricep'],
            'wrist': ['wrist', 'forearm'],
            'hip': ['hip', 'glute', 'buttock'],
            'knee': ['knee', 'quad', 'hamstring'],
            'ankle': ['ankle', 'calf', 'shin'],
            'spine': ['spine', 'back', 'lumbar', 'thoracic', 'cervical', 'neck']
        }
        
        for joint, keywords in joint_keywords.items():
            if any(keyword in instructions for keyword in keywords):
                joints.append(joint)
                
        return joints
    
    def _is_compound_exercise(self, bodypart: str, target: str, instructions: str) -> bool:
        """
        Determine if an exercise is compound (multi-joint) or isolation (single-joint)
        
        Args:
            bodypart: Exercise bodypart
            target: Exercise target muscle
            instructions: Exercise instructions
            
        Returns:
            True if compound, False if isolation
        """
        # Count number of joints involved
        joints_used = self._extract_joints_used(instructions)
        
        # If more than one joint is used, it's likely a compound exercise
        return len(joints_used) > 1
    
    def clean_wger_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess Wger data
        
        Args:
            df: Wger DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Similar cleaning steps as for ExerciseDB data
        # This would need to be customized based on Wger's data structure
        
        return cleaned_df
    
    def download_and_process_images(self, df: pd.DataFrame, limit: int = None) -> pd.DataFrame:
        """
        Download and process exercise images using MediaPipe
        
        Args:
            df: DataFrame with exercise data (must contain 'gifUrl' column)
            limit: Optional limit on number of images to process
            
        Returns:
            DataFrame with added pose feature columns
        """
        if 'gifUrl' not in df.columns:
            raise ValueError("DataFrame must contain 'gifUrl' column")
        
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        
        # Process images and extract pose landmarks
        landmarks_list = []
        
        # Create a directory for saving processed images
        image_dir = f"{self.processed_data_dir}/images"
        os.makedirs(image_dir, exist_ok=True)
        
        # Limit the number of images to process if specified
        process_df = df.head(limit) if limit else df
        
        for i, row in tqdm(process_df.iterrows(), total=len(process_df), desc="Processing images"):
            try:
                # Download image
                response = requests.get(row['gifUrl'])
                response.raise_for_status()
                
                # For GIFs, extract the first frame
                image = Image.open(io.BytesIO(response.content))
                if hasattr(image, 'n_frames') and image.n_frames > 1:
                    image.seek(0)
                
                # Convert to numpy array for OpenCV
                image_np = np.array(image.convert('RGB'))
                
                # Process with MediaPipe
                results = pose.process(image_np)
                
                if results.pose_landmarks:
                    # Extract landmark positions
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                else:
                    # If no landmarks detected, fill with zeros
                    landmarks = [0] * (33 * 4)  # 33 landmarks with x,y,z,visibility
                
                landmarks_list.append(landmarks)
                
                # Save the first frame as an image
                image_path = f"{image_dir}/{row['id']}.jpg"
                cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing image {row['gifUrl']}: {e}")
                landmarks_list.append([0] * (33 * 4))
        
        # Create feature columns for landmarks
        landmark_columns = []
        for i in range(33):
            for dim in ['x', 'y', 'z', 'vis']:
                landmark_columns.append(f"landmark_{i}_{dim}")
        
        landmarks_df = pd.DataFrame(landmarks_list, columns=landmark_columns)
        
        # Add landmark columns to original DataFrame
        result_df = process_df.copy()
        for col in landmark_columns:
            result_df[col] = landmarks_df[col]
        
        return result_df
    
    def process_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process text features using TF-IDF
        
        Args:
            df: DataFrame with exercise data
            
        Returns:
            DataFrame with text features
        """
        # Combine name and instructions for better feature extraction
        if 'instructions' in df.columns and 'name' in df.columns:
            df['text_combined'] = df['name'] + ' ' + df['instructions'].fillna('')
        elif 'name' in df.columns:
            df['text_combined'] = df['name']
        else:
            return df
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(df['text_combined'])
        
        # Convert to DataFrame
        feature_names = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{f}" for f in feature_names])
        
        # Concatenate with original DataFrame
        result_df = pd.concat([df, tfidf_df], axis=1)
        
        return result_df
    
    def run_pipeline(self, process_images: bool = False, image_limit: Optional[int] = 100) -> Dict[str, pd.DataFrame]:
        """
        Run the full preprocessing pipeline
        
        Args:
            process_images: Whether to process images
            image_limit: Maximum number of images to process
            
        Returns:
            Dictionary of processed DataFrames
        """
        print("Loading data...")
        data = self.load_data()
        processed_data = {}
        
        if "exercisedb" in data:
            print("Processing ExerciseDB data...")
            exercisedb_df = self.clean_exercisedb_data(data["exercisedb"])
            
            # Process text features
            exercisedb_df = self.process_text_features(exercisedb_df)
            
            # Process images if requested
            if process_images:
                print("Processing ExerciseDB images...")
                exercisedb_df = self.download_and_process_images(exercisedb_df, limit=image_limit)
            
            processed_data["exercisedb"] = exercisedb_df
            
            # Save processed data
            exercisedb_df.to_csv(f"{self.processed_data_dir}/exercisedb_processed.csv", index=False)
            print("ExerciseDB data processed and saved.")
        
        if "wger" in data:
            print("Processing Wger data...")
            wger_df = self.clean_wger_data(data["wger"])
            
            # Process text features
            wger_df = self.process_text_features(wger_df)
            
            processed_data["wger"] = wger_df
            
            # Save processed data
            wger_df.to_csv(f"{self.processed_data_dir}/wger_processed.csv", index=False)
            print("Wger data processed and saved.")
        
        return processed_data


if __name__ == "__main__":
    preprocessor = ExerciseDataPreprocessor()
    processed_data = preprocessor.run_pipeline(process_images=True, image_limit=10)
    print("Preprocessing complete.")
