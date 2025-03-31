# src/models/classifier.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import mlflow.pytorch

class ExerciseDataset(Dataset):
    """
    PyTorch Dataset for exercise data
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset
        
        Args:
            features: Feature matrix
            labels: Labels
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class MultimodalExerciseClassifier(nn.Module):
    """
    Neural network for classifying exercises using multimodal features
    """
    def __init__(self, 
                 text_input_dim: int, 
                 pose_input_dim: int,
                 num_classes: int,
                 hidden_dim: int = 128):
        """
        Initialize model
        
        Args:
            text_input_dim: Dimension of text features
            pose_input_dim: Dimension of pose features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        
        # Text processing branch
        self.text_layers = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Pose processing branch
        self.pose_layers = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined layers
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, text_features: torch.Tensor, pose_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            text_features: Text features
            pose_features: Pose features
            
        Returns:
            Output logits
        """
        text_output = self.text_layers(text_features)
        pose_output = self.pose_layers(pose_features)
        
        # Concatenate the outputs
        combined = torch.cat((text_output, pose_output), dim=1)
        
        # Pass through combined layers
        output = self.combined_layers(combined)
        
        return output


class ExerciseClassifier:
    """
    Main class for training and using exercise classification models
    """
    def __init__(self, 
                 data_path: str = "data/processed/exercisedb_processed.csv",
                 model_dir: str = "models"):
        """
        Initialize classifier
        
        Args:
            data_path: Path to processed data
            model_dir: Directory to save models
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_scaler = StandardScaler()
        self.pose_scaler = StandardScaler()
        self.target_encoders = {}
        
        os.makedirs(model_dir, exist_ok=True)
        
    def _split_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Split DataFrame into feature groups
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of feature arrays
        """
        # Text features (TF-IDF)
        text_cols = [col for col in df.columns if col.startswith('tfidf_')]
        
        # Pose features (landmarks)
        pose_cols = [col for col in df.columns if col.startswith('landmark_')]
        
        features = {}
        if text_cols:
            features['text'] = df[text_cols].values
        else:
            # If no text features, create dummy
            features['text'] = np.zeros((len(df), 1))
            
        if pose_cols:
            features['pose'] = df[pose_cols].values
        else:
            # If no pose features, create dummy
            features['pose'] = np.zeros((len(df), 1))
        
        return features
    
    def _prepare_data(self, 
                      df: pd.DataFrame, 
                      target_col: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of features and labels
        """
        # Encode target variable if it's not already numeric
        if target_col not in self.target_encoders:
            self.target_encoders[target_col] = LabelEncoder()
            y = self.target_encoders[target_col].fit_transform(df[target_col])
        else:
            y = self.target_encoders[target_col].transform(df[target_col])
        
        # Split features
        features = self._split_features(df)
        
        # Scale features
        features['text'] = self.text_scaler.fit_transform(features['text'])
        features['pose'] = self.pose_scaler.fit_transform(features['pose'])
        
        return features, y
    
    def create_model(self, features: Dict[str, np.ndarray], num_classes: int) -> MultimodalExerciseClassifier:
        """
        Create model
        
        Args:
            features: Feature dictionary
            num_classes: Number of output classes
            
        Returns:
            Initialized model
        """
        text_input_dim = features['text'].shape[1]
        pose_input_dim = features['pose'].shape[1]
        
        model = MultimodalExerciseClassifier(
            text_input_dim=text_input_dim,
            pose_input_dim=pose_input_dim,
            num_classes=num_classes
        )
        
        return model.to(self.device)
    
    def train_model(self, 
                    df: pd.DataFrame,
                    target_col: str,
                    batch_size: int = 32,
                    learning_rate: float = 0.001,
                    epochs: int = 50,
                    val_size: float = 0.2,
                    experiment_name: str = "exercise_classification") -> MultimodalExerciseClassifier:
        """
        Train a model for the specified target
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            batch_size: Batch size
            learning_rate: Learning rate
            epochs: Number of epochs
            val_size: Validation set size
            experiment_name: MLflow experiment name
            
        Returns:
            Trained model
        """
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Prepare data
        features, labels = self._prepare_data(df, target_col)
        
        # Split into train and validation sets
        X_train = {}
        X_val = {}
        
        # Split each feature set
        for key in features:
            X_train[key], X_val[key], y_train, y_val = train_test_split(
                features[key], labels, test_size=val_size, random_state=42, stratify=labels
            )
        
        # Create datasets
        train_text = torch.tensor(X_train['text'], dtype=torch.float32)
        train_pose = torch.tensor(X_train['pose'], dtype=torch.float32)
        train_labels = torch.tensor(y_train, dtype=torch.long)
        
        val_text = torch.tensor(X_val['text'], dtype=torch.float32)
        val_pose = torch.tensor(X_val['pose'], dtype=torch.float32)
        val_labels = torch.tensor(y_val, dtype=torch.long)
        
        # Create DataLoader
        train_data = [
            (train_text[i], train_pose[i], train_labels[i]) 
            for i in range(len(train_labels))
        ]
        val_data = [
            (val_text[i], val_pose[i], val_labels[i]) 
            for i in range(len(val_labels))
        ]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Create model
        num_classes = len(np.unique(labels))
        model = self.create_model(features, num_classes)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("target_column", target_col)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("text_features", features['text'].shape[1])
            mlflow.log_param("pose_features", features['pose'].shape[1])
            
            # Training loop
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                correct = 0
                total = 0
                
                for text, pose, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
                    text, pose, label = text.to(self.device), pose.to(self.device), label.to(self.device)
                    
                    # Forward pass
                    outputs = model(text, pose)
                    loss = criterion(outputs, label)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                
                train_loss = train_loss / len(train_loader)
                train_acc = 100 * correct / total
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                
                # Validation
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for text, pose, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                        text, pose, label = text.to(self.device), pose.to(self.device), label.to(self.device)
                        
                        outputs = model(text, pose)
                        loss = criterion(outputs, label)
                        
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Log metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = f"{self.model_dir}/{target_col}_model.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"Model saved to {model_path}")
            
            # Plot training curves
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'{target_col} - Loss Curves')
            
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(val_accs, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title(f'{target_col} - Accuracy Curves')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.model_dir}/{target_col}_training_curves.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            
            # Log model
            mlflow.pytorch.log_model(model, f"{target_col}_model")
        
        # Load best model
        model.load_state_dict(torch.load(f"{self.model_dir}/{target_col}_model.pth"))
        
        return model
    
    def train_all_models(self, target_columns: List[str]) -> Dict[str, MultimodalExerciseClassifier]:
        """
        Train models for all target columns
        
        Args:
            target_columns: List of target column names
            
        Returns:
            Dictionary of trained models
        """
        df = pd.read_csv(self.data_path)
        models = {}
        
        for target_col in target_columns:
            print(f"\nTraining model for {target_col}...")
            models[target_col] = self.train_model(df, target_col)
        
        return models
    
    def predict(self, 
                model: MultimodalExerciseClassifier, 
                df: pd.DataFrame, 
                target_col: str) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model: Trained model
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Predicted labels
        """
        # Split features
        features = self._split_features(df)
        
        # Scale features
        features['text'] = self.text_scaler.transform(features['text'])
        features['pose'] = self.pose_scaler.transform(features['pose'])
        
        # Convert to tensors
        text = torch.tensor(features['text'], dtype=torch.float32).to(self.device)
        pose = torch.tensor(features['pose'], dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = model(text, pose)
            _, predicted = torch.max(outputs, 1)
        
        # Convert to numpy
        predicted = predicted.cpu().numpy()
        
        # Decode predictions
        if target_col in self.target_encoders:
            predicted = self.target_encoders[target_col].inverse_transform(predicted)
        
        return predicted