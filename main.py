# main.py
import os
import argparse
import pandas as pd
import torch
from src.data.api_fetcher import fetch_and_merge_data
from src.data.preprocessor import ExerciseDataPreprocessor
from src.models.classifier import ExerciseClassifier

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Exercise Classification System')
    parser.add_argument('--fetch', action='store_true', help='Fetch new data from APIs')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--process_images', action='store_true', help='Process exercise images')
    parser.add_argument('--image_limit', type=int, default=100, help='Limit number of images to process')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Fetch data
    if args.fetch:
        print("Fetching data from APIs...")
        data = fetch_and_merge_data()
        print("Data fetching complete.")
    
    # Preprocess data
    if args.preprocess:
        print("Preprocessing data...")
        preprocessor = ExerciseDataPreprocessor()
        processed_data = preprocessor.run_pipeline(
            process_images=args.process_images,
            image_limit=args.image_limit
        )
        print("Data preprocessing complete.")
    
    # Train models
    if args.train:
        print("Training models...")
        classifier = ExerciseClassifier()
        
        # Define target columns to train models for
        target_columns = [
            'movement_pattern',
            'is_compound',
            'intensity_level',
            'exercise_type',
            'risk_assessment'
        ]
        
        models = classifier.train_all_models(target_columns)
        print("Model training complete.")

    # Make predictions
    if args.predict:
        print("Making predictions...")
        # This would typically be used for new data
        # For demonstration purposes, we'll use the training data

        df = pd.read_csv("data/processed/exercisedb_processed.csv")

        # Load models
        for target_col in ['movement_pattern', 'is_compound', 'risk_assessment']:
            model_path = f"models/{target_col}_model.pth"

            if os.path.exists(model_path):
                print(f"Making predictions for {target_col}...")

                # Create a fresh classifier instance
                classifier = ExerciseClassifier()

                # Load the saved scalers, encoders, and metadata
                try:
                    # Load model metadata
                    metadata_path = f"models/{target_col}_metadata.pkl"
                    if os.path.exists(metadata_path):
                        metadata = joblib.load(metadata_path)
                        num_classes = metadata['num_classes']
                        print(f"  Loaded model metadata: {num_classes} classes")
                    else:
                        # Fall back to checking the model architecture
                        state_dict = torch.load(model_path)
                        num_classes = state_dict['combined_layers.3.bias'].shape[0]
                        print(f"  Detected {num_classes} classes from model architecture")

                    # Load scalers
                    text_scaler_path = f"models/{target_col}_text_scaler.pkl"
                    pose_scaler_path = f"models/{target_col}_pose_scaler.pkl"
                    encoder_path = f"models/{target_col}_encoder.pkl"

                    if os.path.exists(text_scaler_path) and os.path.exists(pose_scaler_path):
                        print("  Loading saved scalers")
                        classifier.text_scaler = joblib.load(text_scaler_path)
                        classifier.pose_scaler = joblib.load(pose_scaler_path)
                    else:
                        print("  Warning: Scalers not found, fitting on current data")
                        features = classifier._split_features(df)
                        classifier.text_scaler.fit(features['text'])
                        classifier.pose_scaler.fit(features['pose'])

                    # Load encoder
                    if os.path.exists(encoder_path):
                        print("  Loading saved encoder")
                        classifier.target_encoders[target_col] = joblib.load(encoder_path)

                except Exception as e:
                    print(f"  Error loading saved components: {e}")
                    print("  Falling back to defaults")
                    features = classifier._split_features(df)
                    num_classes = df[target_col].nunique()
                    classifier.text_scaler.fit(features['text'])
                    classifier.pose_scaler.fit(features['pose'])

                # Create and load model with the correct number of classes
                features = classifier._split_features(df)
                model = classifier.create_model(features, num_classes)
                model.load_state_dict(torch.load(model_path))

                # Make predictions
                predictions = classifier.predict(model, df, target_col)

                # Save predictions
                df[f"{target_col}_pred"] = predictions
                print(f"  Predictions complete for {target_col}")

        # Save results
        df.to_csv("data/predictions.csv", index=False)
        print("Predictions saved to data/predictions.csv")

    print("Done.")

if __name__ == '__main__':
    main()
