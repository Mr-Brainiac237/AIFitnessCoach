# main.py
import os
import argparse
import pandas as pd
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
        
        classifier = ExerciseClassifier()
        df = pd.read_csv("data/processed/exercisedb_processed.csv")
        
        # Load models
        for target_col in ['movement_pattern', 'is_compound', 'risk_assessment']:
            if os.path.exists(f"models/{target_col}_model.pth"):
                # Create model architecture
                features = classifier._split_features(df)
                
                # Determine number of classes
                if target_col == 'is_compound':
                    num_classes = 2  # binary
                else:
                    num_classes = df[target_col].nunique()
                
                # Create and load model
                model = classifier.create_model(features, num_classes)
                model.load_state_dict(pd.read_pickle(f"models/{target_col}_model.pth"))
                
                # Make predictions
                predictions = classifier.predict(model, df, target_col)
                
                # Save predictions
                df[f"{target_col}_pred"] = predictions
        
        # Save results
        df.to_csv("data/predictions.csv", index=False)
        print("Predictions saved to data/predictions.csv")
    
    print("Done.")

if __name__ == '__main__':
    main()
