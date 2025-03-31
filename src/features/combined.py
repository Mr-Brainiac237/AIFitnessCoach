# src/features/combined.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

from src.features.text_features import TextFeatureExtractor
from src.features.pose_features import PoseFeatureExtractor

class CombinedFeatureExtractor:
    """
    Combine text and pose features and create final feature set for modeling
    """
    def __init__(self, 
                 text_extractor: Optional[TextFeatureExtractor] = None,
                 pose_extractor: Optional[PoseFeatureExtractor] = None,
                 use_pca: bool = True,
                 pca_components: int = 50,
                 model_dir: str = "models"):
        """
        Initialize the combined feature extractor
        
        Args:
            text_extractor: Text feature extractor instance
            pose_extractor: Pose feature extractor instance
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components to use
            model_dir: Directory to save feature extractors
        """
        self.text_extractor = text_extractor or TextFeatureExtractor()
        self.pose_extractor = pose_extractor or PoseFeatureExtractor()
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.model_dir = model_dir
        
        self.text_scaler = StandardScaler()
        self.pose_scaler = StandardScaler()
        self.combined_scaler = StandardScaler()
        
        self.text_pca = PCA(n_components=pca_components) if use_pca else None
        self.pose_pca = PCA(n_components=pca_components) if use_pca else None
        
        os.makedirs(model_dir, exist_ok=True)
    
    def extract_all_features(self, 
                            df: pd.DataFrame, 
                            process_images: bool = True,
                            image_limit: Optional[int] = None,
                            save_intermediate: bool = True) -> pd.DataFrame:
        """
        Extract all features (text and pose) from a DataFrame
        
        Args:
            df: Input DataFrame with exercise data
            process_images: Whether to process images
            image_limit: Maximum number of images to process
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame with all extracted features
        """
        print("Starting feature extraction pipeline...")
        
        # Extract text features
        print("\n1. Extracting text features...")
        text_df = self.text_extractor.extract_features(df)
        
        # Extract keyword features
        print("2. Extracting keyword features...")
        text_kw_df = self.text_extractor.extract_keyword_features(text_df)
        
        # Extract exercise attributes from text
        print("3. Extracting exercise attributes from text...")
        text_attr_df = self.text_extractor.extract_exercise_attributes(text_kw_df)
        
        if save_intermediate:
            text_attr_df.to_csv(f"{self.model_dir}/text_features.csv", index=False)
            print(f"Text features saved to {self.model_dir}/text_features.csv")
        
        # Extract pose features
        if process_images:
            print("\n4. Processing images for pose features...")
            pose_df = self.pose_extractor.download_and_process_images(
                text_attr_df, 
                limit=image_limit
            )
            
            if save_intermediate:
                pose_df.to_csv(f"{self.model_dir}/pose_raw_features.csv", index=False)
                print(f"Raw pose features saved to {self.model_dir}/pose_raw_features.csv")
            
            print("5. Extracting derived pose features...")
            pose_derived_df = self.pose_extractor.extract_pose_features(pose_df)
            
            if save_intermediate:
                pose_derived_df.to_csv(f"{self.model_dir}/pose_derived_features.csv", index=False)
                print(f"Derived pose features saved to {self.model_dir}/pose_derived_features.csv")
            
            result_df = pose_derived_df
        else:
            result_df = text_attr_df
        
        # Combine features for final set
        print("\n6. Generating final feature set...")
        final_df = self.generate_final_features(result_df)
        
        if save_intermediate:
            final_df.to_csv(f"{self.model_dir}/all_features.csv", index=False)
            print(f"All features saved to {self.model_dir}/all_features.csv")
        
        print("Feature extraction complete!")
        return final_df
    
    def generate_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate final feature set for modeling
        
        Args:
            df: DataFrame with extracted features
            
        Returns:
            DataFrame with final features
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Split features into groups
        text_feature_cols = [col for col in df.columns if col.startswith(('tfidf_', 'embedding_', 'topic_'))]
        pose_feature_cols = [col for col in df.columns if col.startswith(('landmark_', 'angle_', 'height_', 'width_'))]
        
        # Extract text and pose feature matrices
        if text_feature_cols:
            text_features = df[text_feature_cols].fillna(0).values
            
            # Scale text features
            text_features_scaled = self.text_scaler.fit_transform(text_features)
            
            # Apply PCA if needed
            if self.use_pca and self.text_pca and text_features_scaled.shape[1] > self.pca_components:
                text_features_pca = self.text_pca.fit_transform(text_features_scaled)
                
                # Add PCA features to result DataFrame
                for i in range(text_features_pca.shape[1]):
                    result_df[f'text_pca_{i}'] = text_features_pca[:, i]
                
                print(f"Reduced text features from {text_features.shape[1]} to {text_features_pca.shape[1]} dimensions")
                print(f"Explained variance ratio: {sum(self.text_pca.explained_variance_ratio_):.2f}")
        
        # Process pose features if available
        if pose_feature_cols:
            pose_features = df[pose_feature_cols].fillna(0).values
            
            # Scale pose features
            pose_features_scaled = self.pose_scaler.fit_transform(pose_features)
            
            # Apply PCA if needed
            if self.use_pca and self.pose_pca and pose_features_scaled.shape[1] > self.pca_components:
                pose_features_pca = self.pose_pca.fit_transform(pose_features_scaled)
                
                # Add PCA features to result DataFrame
                for i in range(pose_features_pca.shape[1]):
                    result_df[f'pose_pca_{i}'] = pose_features_pca[:, i]
                
                print(f"Reduced pose features from {pose_features.shape[1]} to {pose_features_pca.shape[1]} dimensions")
                print(f"Explained variance ratio: {sum(self.pose_pca.explained_variance_ratio_):.2f}")
        
        # Create combined features
        # These are features that use both text and pose information
        self._create_combined_features(result_df)
        
        return result_df
    
    def _create_combined_features(self, df: pd.DataFrame) -> None:
        """
        Create combined features that leverage both text and pose data
        
        Args:
            df: DataFrame to update with combined features
        """
        # Example: Movement quality score
        # Combines pose symmetry with text intensity/difficulty
        if 'pose_arm_symmetry' in df.columns and 'pose_leg_symmetry' in df.columns:
            symmetry_score = (df['pose_arm_symmetry'] + df['pose_leg_symmetry']) / 2
            
            # Adjust score based on exercise type/intensity
            if 'intensity_level' in df.columns:
                intensity_multiplier = df['intensity_level'].map({
                    'beginner': 1.2,   # More forgiving for beginners
                    'intermediate': 1.0,
                    'advanced': 0.8    # Stricter for advanced exercises
                }).fillna(1.0)
                
                movement_quality = symmetry_score * intensity_multiplier
                # Clamp values between 1-5
                df['movement_quality'] = movement_quality.clip(0, 1) * 4 + 1
            else:
                df['movement_quality'] = symmetry_score.clip(0, 1) * 4 + 1
        
        # Example: Risk assessment combining pose position and text description
        if 'pose_position' in df.columns and 'risk_assessment' in df.columns:
            # Adjust risk based on pose position
            position_risk_multiplier = df['pose_position'].map({
                'standing': 1.0,     # Neutral risk
                'seated': 0.8,       # Lower risk
                'supine': 0.7,       # Lower risk (lying on back)
                'prone': 1.2,        # Higher risk (lying face down, requires neck extension)
                'unknown': 1.0,      # Neutral risk
                'other': 1.1         # Slightly higher risk for unusual positions
            }).fillna(1.0)
            
            # Get numerical risk value from text assessment
            text_risk_score = df['risk_assessment'].map({
                'low': 0.25,
                'medium': 0.5,
                'high': 0.75
            }).fillna(0.5)
            
            # Combine text and pose risk
            combined_risk = (text_risk_score * position_risk_multiplier).clip(0, 1)
            
            # Map back to categories, but with more granularity
            df['combined_risk_assessment'] = pd.cut(
                combined_risk, 
                bins=[0, 0.3, 0.6, 1], 
                labels=['low', 'medium', 'high']
            )
    
    def save_extractors(self) -> None:
        """
        Save feature extractors and transformers to disk
        """
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save text feature extractor
        joblib.dump(self.text_extractor, f"{self.model_dir}/text_extractor.pkl")
        
        # Save pose feature extractor
        joblib.dump(self.pose_extractor, f"{self.model_dir}/pose_extractor.pkl")
        
        # Save scalers
        joblib.dump(self.text_scaler, f"{self.model_dir}/text_scaler.pkl")
        joblib.dump(self.pose_scaler, f"{self.model_dir}/pose_scaler.pkl")
        joblib.dump(self.combined_scaler, f"{self.model_dir}/combined_scaler.pkl")
        
        # Save PCA transformers if used
        if self.use_pca:
            joblib.dump(self.text_pca, f"{self.model_dir}/text_pca.pkl")
            joblib.dump(self.pose_pca, f"{self.model_dir}/pose_pca.pkl")
    
    def load_extractors(self) -> None:
        """
        Load feature extractors and transformers from disk
        """
        # Load text feature extractor
        self.text_extractor = joblib.load(f"{self.model_dir}/text_extractor.pkl")
        
        # Load pose feature extractor
        self.pose_extractor = joblib.load(f"{self.model_dir}/pose_extractor.pkl")
        
        # Load scalers
        self.text_scaler = joblib.load(f"{self.model_dir}/text_scaler.pkl")
        self.pose_scaler = joblib.load(f"{self.model_dir}/pose_scaler.pkl")
        self.combined_scaler = joblib.load(f"{self.model_dir}/combined_scaler.pkl")
        
        # Load PCA transformers if used
        if self.use_pca:
            self.text_pca = joblib.load(f"{self.model_dir}/text_pca.pkl")
            self.pose_pca = joblib.load(f"{self.model_dir}/pose_pca.pkl") 