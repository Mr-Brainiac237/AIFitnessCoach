# src/features/pose_features.py
import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Union, Optional
import requests
from PIL import Image
import io
import json
from tqdm import tqdm
import math

class PoseFeatureExtractor:
    """
    Extract features from exercise images/videos using MediaPipe pose estimation
    """
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 image_dir: str = "data/processed/images",
                 use_face_landmarks: bool = False):
        """
        Initialize the pose feature extractor
        
        Args:
            confidence_threshold: Minimum confidence threshold for pose detection
            image_dir: Directory to save processed images
            use_face_landmarks: Whether to include face landmarks in features
        """
        self.confidence_threshold = confidence_threshold
        self.image_dir = image_dir
        self.use_face_landmarks = use_face_landmarks
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create directory for saving images
        os.makedirs(image_dir, exist_ok=True)
        
        # Define the landmark indices for different body parts
        self.landmark_groups = {
            'face': list(range(0, 11)),
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22], 
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32],
            'torso': [11, 12, 23, 24]
        }
        
        # Define key joint pairs for angle calculations
        self.joint_pairs = [
            # Left arm angles
            ([11, 13, 15], "left_elbow_angle"),       # Left shoulder, elbow, wrist
            ([13, 11, 23], "left_shoulder_angle"),    # Left elbow, shoulder, hip
            
            # Right arm angles
            ([12, 14, 16], "right_elbow_angle"),      # Right shoulder, elbow, wrist
            ([14, 12, 24], "right_shoulder_angle"),   # Right elbow, shoulder, hip
            
            # Left leg angles
            ([23, 25, 27], "left_knee_angle"),        # Left hip, knee, ankle
            ([25, 23, 11], "left_hip_angle"),         # Left knee, hip, shoulder
            
            # Right leg angles
            ([24, 26, 28], "right_knee_angle"),       # Right hip, knee, ankle
            ([26, 24, 12], "right_hip_angle"),        # Right knee, hip, shoulder
            
            # Torso angles
            ([12, 24, 26], "right_torso_bend"),       # Right shoulder, hip, knee
            ([11, 23, 25], "left_torso_bend"),        # Left shoulder, hip, knee
        ]
    
    def download_and_process_images(self, df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Download and process exercise images using MediaPipe
        
        Args:
            df: DataFrame with exercise data (must contain 'gifUrl' or 'imageUrl' column)
            limit: Optional limit on number of images to process
            
        Returns:
            DataFrame with added pose feature columns
        """
        url_column = None
        if 'gifUrl' in df.columns:
            url_column = 'gifUrl'
        elif 'imageUrl' in df.columns:
            url_column = 'imageUrl'
        else:
            raise ValueError("DataFrame must contain 'gifUrl' or 'imageUrl' column")
        
        # Initialize pose detector
        pose = self.mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=self.confidence_threshold
        )
        
        # Process images and extract pose landmarks
        landmarks_list = []
        angles_list = []
        heights_list = []
        widths_list = []
        distances_list = []
        
        # Limit the number of images to process if specified
        process_df = df.head(limit) if limit else df
        
        print(f"Processing {len(process_df)} images...")
        for i, row in tqdm(process_df.iterrows(), total=len(process_df), desc="Processing images"):
            try:
                # Check if image already exists
                image_path = f"{self.image_dir}/{row['id']}.jpg"
                if os.path.exists(image_path):
                    # Load existing image
                    image_np = cv2.imread(image_path)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                else:
                    # Download image
                    response = requests.get(row[url_column])
                    response.raise_for_status()
                    
                    # For GIFs, extract the first frame
                    image = Image.open(io.BytesIO(response.content))
                    if hasattr(image, 'n_frames') and image.n_frames > 1:
                        image.seek(0)
                    
                    # Convert to numpy array for OpenCV
                    image_np = np.array(image.convert('RGB'))
                    
                    # Save the first frame as an image
                    cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
                # Process with MediaPipe
                results = pose.process(image_np)
                
                if results.pose_landmarks:
                    # Extract landmark positions
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    
                    # Calculate joint angles
                    angles = self._calculate_joint_angles(results.pose_landmarks.landmark)
                    
                    # Calculate body proportions
                    heights, widths = self._calculate_body_proportions(results.pose_landmarks.landmark)
                    
                    # Calculate key point distances
                    distances = self._calculate_key_distances(results.pose_landmarks.landmark)
                    
                    # Draw pose landmarks on the image and save
                    annotated_image = image_np.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Save annotated image
                    cv2.imwrite(f"{self.image_dir}/{row['id']}_annotated.jpg", 
                                cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    
                else:
                    # If no landmarks detected, fill with zeros
                    landmarks = [0] * (33 * 4)  # 33 landmarks with x,y,z,visibility
                    angles = {joint_name: 0 for _, joint_name in self.joint_pairs}
                    heights = {group: 0 for group in self.landmark_groups}
                    widths = {group: 0 for group in self.landmark_groups}
                    distances = {}
                
                landmarks_list.append(landmarks)
                angles_list.append(angles)
                heights_list.append(heights)
                widths_list.append(widths)
                distances_list.append(distances)
                
            except Exception as e:
                print(f"Error processing image {row[url_column]}: {e}")
                landmarks_list.append([0] * (33 * 4))
                angles_list.append({joint_name: 0 for _, joint_name in self.joint_pairs})
                heights_list.append({group: 0 for group in self.landmark_groups})
                widths_list.append({group: 0 for group in self.landmark_groups})
                distances_list.append({})
        
        # Create feature columns for landmarks
        landmark_columns = []
        for i in range(33):
            for dim in ['x', 'y', 'z', 'vis']:
                landmark_columns.append(f"landmark_{i}_{dim}")
        
        landmarks_df = pd.DataFrame(landmarks_list, columns=landmark_columns)
        
        # Create angle feature columns
        angle_column_names = [joint_name for _, joint_name in self.joint_pairs]
        angles_data = []
        for angle_dict in angles_list:
            angles_data.append([angle_dict.get(name, 0) for name in angle_column_names])
        
        angles_df = pd.DataFrame(angles_data, columns=[f"angle_{name}" for name in angle_column_names])
        
        # Create height and width ratio features
        height_column_names = list(self.landmark_groups.keys())
        height_data = []
        width_data = []
        
        for height_dict, width_dict in zip(heights_list, widths_list):
            height_data.append([height_dict.get(name, 0) for name in height_column_names])
            width_data.append([width_dict.get(name, 0) for name in height_column_names])
        
        heights_df = pd.DataFrame(height_data, columns=[f"height_{name}" for name in height_column_names])
        widths_df = pd.DataFrame(width_data, columns=[f"width_{name}" for name in height_column_names])
        
        # Add all feature columns to original DataFrame
        result_df = process_df.copy()
        for col in landmark_columns:
            result_df[col] = landmarks_df[col]
        
        for col in angles_df.columns:
            result_df[col] = angles_df[col]
            
        for col in heights_df.columns:
            result_df[col] = heights_df[col]
            
        for col in widths_df.columns:
            result_df[col] = widths_df[col]
        
        return result_df
    
    def _calculate_angle(self, p1: list, p2: list, p3: list) -> float:
        """
        Calculate the angle between three points
        
        Args:
            p1: First point [x, y, z]
            p2: Second point [x, y, z] (the vertex)
            p3: Third point [x, y, z]
            
        Returns:
            Angle in degrees
        """
        # Extract x, y coordinates (ignore z for 2D angle)
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def _calculate_joint_angles(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, float]:
        """
        Calculate joint angles for all defined joint pairs
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Dictionary of joint angles
        """
        angles = {}
        for joint_indices, joint_name in self.joint_pairs:
            p1 = [landmarks[joint_indices[0]].x, landmarks[joint_indices[0]].y, landmarks[joint_indices[0]].z]
            p2 = [landmarks[joint_indices[1]].x, landmarks[joint_indices[1]].y, landmarks[joint_indices[1]].z]
            p3 = [landmarks[joint_indices[2]].x, landmarks[joint_indices[2]].y, landmarks[joint_indices[2]].z]
            
            angles[joint_name] = self._calculate_angle(p1, p2, p3)
        
        return angles
    
    def _calculate_body_proportions(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate height and width ratios for different body parts
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Tuple of (heights_dict, widths_dict)
        """
        heights = {}
        widths = {}
        
        for group_name, indices in self.landmark_groups.items():
            # Calculate height (y-coordinate difference)
            y_coords = [landmarks[i].y for i in indices]
            heights[group_name] = max(y_coords) - min(y_coords)
            
            # Calculate width (x-coordinate difference)
            x_coords = [landmarks[i].x for i in indices]
            widths[group_name] = max(x_coords) - min(x_coords)
        
        return heights, widths
    
    def _calculate_key_distances(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, float]:
        """
        Calculate key distances between body parts
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Dictionary of key distances
        """
        distances = {}
        
        # Calculate shoulder width
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
        distances['shoulder_width'] = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Calculate hip width
        left_hip = np.array([landmarks[23].x, landmarks[23].y])
        right_hip = np.array([landmarks[24].x, landmarks[24].y])
        distances['hip_width'] = np.linalg.norm(right_hip - left_hip)
        
        # Calculate arm lengths
        left_elbow = np.array([landmarks[13].x, landmarks[13].y])
        left_wrist = np.array([landmarks[15].x, landmarks[15].y])
        distances['left_arm_length'] = np.linalg.norm(left_wrist - left_elbow)
        
        right_elbow = np.array([landmarks[14].x, landmarks[14].y])
        right_wrist = np.array([landmarks[16].x, landmarks[16].y])
        distances['right_arm_length'] = np.linalg.norm(right_wrist - right_elbow)
        
        # Calculate leg lengths
        left_knee = np.array([landmarks[25].x, landmarks[25].y])
        left_ankle = np.array([landmarks[27].x, landmarks[27].y])
        distances['left_leg_length'] = np.linalg.norm(left_ankle - left_knee)
        
        right_knee = np.array([landmarks[26].x, landmarks[26].y])
        right_ankle = np.array([landmarks[28].x, landmarks[28].y])
        distances['right_leg_length'] = np.linalg.norm(right_ankle - right_knee)
        
        return distances 