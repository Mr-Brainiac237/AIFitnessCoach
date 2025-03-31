# src/models/trainer.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import joblib
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.models.classifier import MultimodalExerciseClassifier, ExerciseDataset
from src.utils.visualization import plot_confusion_matrix, plot_training_curves, plot_classification_report

class ModelTrainer:
    """
    Handles training and evaluation of exercise classification models
    """
    def __init__(self, 
                 model_dir: str = "models",
                 use_gpu: bool = True,
                 experiment_name: str = "exercise_classification",
                 log_mlflow: bool = True):
        """
        Initialize trainer
        
        Args:
            model_dir: Directory to save models
            use_gpu: Whether to use GPU for training
            experiment_name: MLflow experiment name
            log_mlflow: Whether to log metrics and artifacts to MLflow
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.experiment_name = experiment_name
        self.log_mlflow = log_mlflow
        
        self.target_encoders = {}
        self.scalers = {}
        self.models = {}
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        if self.log_mlflow:
            # Set up MLflow
            mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_col: str,
                    text_features: List[str], 
                    pose_features: List[str],
                    test_size: float = 0.2, 
                    val_size: float = 0.1,
                    random_state: int = 42) -> Dict[str, Any]:
        """
        Prepare data for training and evaluation
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            text_features: List of text feature columns
            pose_features: List of pose feature columns
            test_size: Test set size ratio
            val_size: Validation set size ratio
            random_state: Random seed
            
        Returns:
            Dictionary with prepared data and metadata
        """
        # Encode target variable
        if target_col not in self.target_encoders:
            encoder = LabelEncoder()
            y = encoder.fit_transform(df[target_col])
            self.target_encoders[target_col] = encoder
        else:
            encoder = self.target_encoders[target_col]
            y = encoder.transform(df[target_col])
        
        # Get number of classes
        num_classes = len(encoder.classes_)
        
        # Extract features
        if not text_features:
            # Use dummy feature if no text features provided
            X_text = np.zeros((len(df), 1))
        else:
            X_text = df[text_features].fillna(0).values
            
        if not pose_features:
            # Use dummy feature if no pose features provided
            X_pose = np.zeros((len(df), 1))
        else:
            X_pose = df[pose_features].fillna(0).values
        
        # Scale features
        if f"text_{target_col}" not in self.scalers:
            text_scaler = StandardScaler()
            X_text_scaled = text_scaler.fit_transform(X_text)
            self.scalers[f"text_{target_col}"] = text_scaler
        else:
            text_scaler = self.scalers[f"text_{target_col}"]
            X_text_scaled = text_scaler.transform(X_text)
            
        if f"pose_{target_col}" not in self.scalers:
            pose_scaler = StandardScaler()
            X_pose_scaled = pose_scaler.fit_transform(X_pose)
            self.scalers[f"pose_{target_col}"] = pose_scaler
        else:
            pose_scaler = self.scalers[f"pose_{target_col}"]
            X_pose_scaled = pose_scaler.transform(X_pose)
        
        # Initial train/test split
        X_text_train, X_text_test, X_pose_train, X_pose_test, y_train, y_test = train_test_split(
            X_text_scaled, X_pose_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Further split training set into train/validation
        if val_size > 0:
            # Calculate validation size relative to training set
            rel_val_size = val_size / (1 - test_size)
            
            X_text_train, X_text_val, X_pose_train, X_pose_val, y_train, y_val = train_test_split(
                X_text_train, X_pose_train, y_train, 
                test_size=rel_val_size, random_state=random_state, stratify=y_train
            )
        else:
            # No validation set
            X_text_val, X_pose_val, y_val = None, None, None
        
        # Convert to PyTorch tensors
        text_train_tensor = torch.tensor(X_text_train, dtype=torch.float32)
        pose_train_tensor = torch.tensor(X_pose_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        text_test_tensor = torch.tensor(X_text_test, dtype=torch.float32)
        pose_test_tensor = torch.tensor(X_pose_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        if val_size > 0:
            text_val_tensor = torch.tensor(X_text_val, dtype=torch.float32)
            pose_val_tensor = torch.tensor(X_pose_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        else:
            text_val_tensor, pose_val_tensor, y_val_tensor = None, None, None
        
        # Create TensorDatasets
        train_dataset = TensorDataset(text_train_tensor, pose_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(text_test_tensor, pose_test_tensor, y_test_tensor)
        
        if val_size > 0:
            val_dataset = TensorDataset(text_val_tensor, pose_val_tensor, y_val_tensor)
        else:
            val_dataset = None
        
        # Store class weights for imbalanced datasets
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        # Store feature dimensions
        text_input_dim = X_text_scaled.shape[1]
        pose_input_dim = X_pose_scaled.shape[1]
        
        # Return prepared data
        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "class_weights": class_weights_tensor,
            "num_classes": num_classes,
            "text_input_dim": text_input_dim,
            "pose_input_dim": pose_input_dim,
            "encoder": encoder,
            "feature_names": {
                "text": text_features,
                "pose": pose_features
            }
        }
    
    def create_model(self, 
                    text_input_dim: int, 
                    pose_input_dim: int,
                    num_classes: int,
                    hidden_dim: int = 128) -> MultimodalExerciseClassifier:
        """
        Create a new model instance
        
        Args:
            text_input_dim: Dimension of text features
            pose_input_dim: Dimension of pose features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            
        Returns:
            Initialized model
        """
        model = MultimodalExerciseClassifier(
            text_input_dim=text_input_dim,
            pose_input_dim=pose_input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        return model.to(self.device)
    
    def train(self, 
             data: Dict[str, Any],
             target_col: str,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             weight_decay: float = 1e-5,
             epochs: int = 50,
             early_stopping_patience: int = 10,
             scheduler_factor: float = 0.5,
             scheduler_patience: int = 5,
             class_weights: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Train a model
        
        Args:
            data: Prepared data dictionary
            target_col: Target column name
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            scheduler_factor: Factor for learning rate scheduler
            scheduler_patience: Patience for learning rate scheduler
            class_weights: Optional class weights for imbalanced datasets
            
        Returns:
            Dictionary with training results
        """
        # Extract data
        train_dataset = data["train_dataset"]
        val_dataset = data["val_dataset"]
        num_classes = data["num_classes"]
        text_input_dim = data["text_input_dim"]
        pose_input_dim = data["pose_input_dim"]
        
        # Use provided class weights or the ones from data
        if class_weights is None and "class_weights" in data:
            class_weights = data["class_weights"]
        
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Create model
        model = self.create_model(
            text_input_dim=text_input_dim,
            pose_input_dim=pose_input_dim,
            num_classes=num_classes
        )
        
        # Define loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True
        )
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        early_stopping_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # MLflow logging
        if self.log_mlflow:
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_param("target_column", target_col)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("weight_decay", weight_decay)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("text_features", text_input_dim)
                mlflow.log_param("pose_features", pose_input_dim)
                mlflow.log_param("num_classes", num_classes)
                
                # Training loop
                for epoch in range(epochs):
                    # Training phase
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    # Progress bar
                    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)")
                    
                    for text_features, pose_features, labels in pbar:
                        # Move data to device
                        text_features = text_features.to(self.device)
                        pose_features = pose_features.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Zero gradients
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(text_features, pose_features)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        # Update statistics
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        # Update progress bar
                        pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})
                    
                    # Calculate epoch statistics
                    epoch_loss = running_loss / len(train_loader)
                    epoch_acc = 100 * correct / total
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc)
                    
                    # Log metrics
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
                    
                    # Validation phase
                    if val_loader:
                        val_loss, val_acc = self._validate(model, val_loader, criterion, target_col)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)
                        
                        # Log metrics
                        mlflow.log_metric("val_loss", val_loss, step=epoch)
                        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                        
                        # Print epoch results
                        print(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                        
                        # Learning rate scheduler step
                        scheduler.step(val_loss)
                        
                        # Early stopping check
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stopping_counter = 0
                            
                            # Save best model
                            torch.save(model.state_dict(), f"{self.model_dir}/{target_col}_best_model.pth")
                        else:
                            early_stopping_counter += 1
                            print(f"EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}")
                            
                            if early_stopping_counter >= early_stopping_patience:
                                print("Early stopping triggered")
                                break
                    else:
                        # No validation set, save model at each epoch
                        torch.save(model.state_dict(), f"{self.model_dir}/{target_col}_latest_model.pth")
                        print(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
                
                # Plot and log training curves
                fig = plt.figure(figsize=(12, 5))
                
                # Plot loss curves
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                if val_losses:
                    plt.plot(val_losses, label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'{target_col} - Loss Curves')
                
                # Plot accuracy curves
                plt.subplot(1, 2, 2)
                plt.plot(train_accs, label='Train Accuracy')
                if val_accs:
                    plt.plot(val_accs, label='Val Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.title(f'{target_col} - Accuracy Curves')
                
                plt.tight_layout()
                
                # Save plot
                plot_path = f"{self.model_dir}/{target_col}_training_curves.png"
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path)
                
                # Save model to MLflow
                mlflow.pytorch.log_model(model, f"{target_col}_model")
                
                # Log scalers
                if f"text_{target_col}" in self.scalers:
                    joblib.dump(self.scalers[f"text_{target_col}"], f"{self.model_dir}/{target_col}_text_scaler.pkl")
                    mlflow.log_artifact(f"{self.model_dir}/{target_col}_text_scaler.pkl")
                    
                if f"pose_{target_col}" in self.scalers:
                    joblib.dump(self.scalers[f"pose_{target_col}"], f"{self.model_dir}/{target_col}_pose_scaler.pkl")
                    mlflow.log_artifact(f"{self.model_dir}/{target_col}_pose_scaler.pkl")
                
                # Log encoder
                joblib.dump(self.target_encoders[target_col], f"{self.model_dir}/{target_col}_encoder.pkl")
                mlflow.log_artifact(f"{self.model_dir}/{target_col}_encoder.pkl")
        else:
            # Training without MLflow logging
            # Similar code as above, but without MLflow calls
            for epoch in range(epochs):
                # Training phase
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                # Progress bar
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)")
                
                for text_features, pose_features, labels in pbar:
                    # Move data to device
                    text_features = text_features.to(self.device)
                    pose_features = pose_features.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(text_features, pose_features)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})
                
                # Calculate epoch statistics
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100 * correct / total
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
                
                # Validation phase
                if val_loader:
                    val_loss, val_acc = self._validate(model, val_loader, criterion, target_col)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    
                    # Print epoch results
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                    
                    # Learning rate scheduler step
                    scheduler.step(val_loss)
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stopping_counter = 0
                        
                        # Save best model
                        torch.save(model.state_dict(), f"{self.model_dir}/{target_col}_best_model.pth")
                    else:
                        early_stopping_counter += 1
                        print(f"EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}")
                        
                        if early_stopping_counter >= early_stopping_patience:
                            print("Early stopping triggered")
                            break
                else:
                    # No validation set, save model at each epoch
                    torch.save(model.state_dict(), f"{self.model_dir}/{target_col}_latest_model.pth")
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
            
            # Plot training curves
            plot_training_curves(
                train_losses, 
                val_losses if val_loader else None,
                train_accs,
                val_accs if val_loader else None,
                title=f"{target_col} Training",
                save_path=f"{self.model_dir}/{target_col}_training_curves.png"
            )
            
            # Save scalers
            if f"text_{target_col}" in self.scalers:
                joblib.dump(self.scalers[f"text_{target_col}"], f"{self.model_dir}/{target_col}_text_scaler.pkl")
                
            if f"pose_{target_col}" in self.scalers:
                joblib.dump(self.scalers[f"pose_{target_col}"], f"{self.model_dir}/{target_col}_pose_scaler.pkl")
            
            # Save encoder
            joblib.dump(self.target_encoders[target_col], f"{self.model_dir}/{target_col}_encoder.pkl")
            
        # Load best model for further use
        if os.path.exists(f"{self.model_dir}/{target_col}_best_model.pth"):
            model.load_state_dict(torch.load(f"{self.model_dir}/{target_col}_best_model.pth"))
        
        # Store model
        self.models[target_col] = model
        
        # Return training results
        return {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses if val_loader else None,
            "train_accs": train_accs,
            "val_accs": val_accs if val_loader else None,
            "best_val_loss": best_val_loss if val_loader else None
        }
    
    def _validate(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module, target_col: str) -> Tuple[float, float]:
        """
        Validate model on a dataloader
        
        Args:
            model: Model to validate
            dataloader: Validation dataloader
            criterion: Loss function
            target_col: Target column name
            
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text_features, pose_features, labels in dataloader:
                # Move data to device
                text_features = text_features.to(self.device)
                pose_features = pose_features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(text_features, pose_features)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions
                all_preds.extend(predicted.cpu().numpy())
        
        # Convert to numpy array
        all_preds = np.array(all_preds)
        
        # Decode predictions if encoder exists
        if target_col in self.target_encoders:
            encoder = self.target_encoders[target_col]
            decoded_preds = encoder.inverse_transform(all_preds)
            return decoded_preds
        else:
            return all_preds
    
    def save(self, target_col: str) -> None:
        """
        Save model and associated data
        
        Args:
            target_col: Target column name (model to save)
        """
        # Check if model exists
        if target_col not in self.models:
            raise ValueError(f"Model for target '{target_col}' not found. Train the model first.")
        
        model = self.models[target_col]
        
        # Create save directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), f"{self.model_dir}/{target_col}_model.pth")
        
        # Save scalers
        if f"text_{target_col}" in self.scalers:
            joblib.dump(self.scalers[f"text_{target_col}"], f"{self.model_dir}/{target_col}_text_scaler.pkl")
            
        if f"pose_{target_col}" in self.scalers:
            joblib.dump(self.scalers[f"pose_{target_col}"], f"{self.model_dir}/{target_col}_pose_scaler.pkl")
        
        # Save encoder
        if target_col in self.target_encoders:
            joblib.dump(self.target_encoders[target_col], f"{self.model_dir}/{target_col}_encoder.pkl")
        
        print(f"Model and associated data for '{target_col}' saved to {self.model_dir}")
    
    def load(self, 
            target_col: str,
            text_input_dim: Optional[int] = None,
            pose_input_dim: Optional[int] = None,
            num_classes: Optional[int] = None) -> bool:
        """
        Load model and associated data
        
        Args:
            target_col: Target column name (model to load)
            text_input_dim: Text input dimension (required if not previously loaded)
            pose_input_dim: Pose input dimension (required if not previously loaded)
            num_classes: Number of classes (required if not previously loaded)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if model files exist
        model_path = f"{self.model_dir}/{target_col}_model.pth"
        text_scaler_path = f"{self.model_dir}/{target_col}_text_scaler.pkl"
        pose_scaler_path = f"{self.model_dir}/{target_col}_pose_scaler.pkl"
        encoder_path = f"{self.model_dir}/{target_col}_encoder.pkl"
        
        if not os.path.exists(model_path):
            print(f"Model file for '{target_col}' not found at {model_path}")
            return False
        
        # Load encoder first to get num_classes
        if os.path.exists(encoder_path):
            self.target_encoders[target_col] = joblib.load(encoder_path)
            loaded_num_classes = len(self.target_encoders[target_col].classes_)
        else:
            if num_classes is None:
                raise ValueError("Number of classes must be provided if encoder file doesn't exist")
            loaded_num_classes = num_classes
        
        # Load scalers
        if os.path.exists(text_scaler_path):
            self.scalers[f"text_{target_col}"] = joblib.load(text_scaler_path)
            loaded_text_dim = self.scalers[f"text_{target_col}"].n_features_in_
        else:
            if text_input_dim is None:
                raise ValueError("Text input dimension must be provided if text scaler file doesn't exist")
            loaded_text_dim = text_input_dim
        
        if os.path.exists(pose_scaler_path):
            self.scalers[f"pose_{target_col}"] = joblib.load(pose_scaler_path)
            loaded_pose_dim = self.scalers[f"pose_{target_col}"].n_features_in_
        else:
            if pose_input_dim is None:
                raise ValueError("Pose input dimension must be provided if pose scaler file doesn't exist")
            loaded_pose_dim = pose_input_dim
        
        # Create model
        model = self.create_model(
            text_input_dim=loaded_text_dim,
            pose_input_dim=loaded_pose_dim,
            num_classes=loaded_num_classes
        )
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Store model
        self.models[target_col] = model
        
        print(f"Model and associated data for '{target_col}' loaded from {self.model_dir}")
        return True
    
    def train_all_models(self, 
                        df: pd.DataFrame, 
                        target_columns: List[str],
                        text_feature_extractor: Callable[[pd.DataFrame], List[str]] = None,
                        pose_feature_extractor: Callable[[pd.DataFrame], List[str]] = None,
                        **train_kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Train models for all target columns
        
        Args:
            df: Input DataFrame
            target_columns: List of target column names
            text_feature_extractor: Function to extract text features
            pose_feature_extractor: Function to extract pose features
            **train_kwargs: Additional arguments for train method
            
        Returns:
            Dictionary of training results for each target
        """
        results = {}
        
        # Default feature extractors
        if text_feature_extractor is None:
            text_feature_extractor = lambda df: [col for col in df.columns if col.startswith(('tfidf_', 'embedding_', 'topic_'))]
        
        if pose_feature_extractor is None:
            pose_feature_extractor = lambda df: [col for col in df.columns if col.startswith(('landmark_', 'angle_', 'height_', 'width_'))]
        
        for target_col in target_columns:
            print(f"\n{'='*50}")
            print(f"Training model for {target_col}")
            print(f"{'='*50}")
            
            # Extract features
            text_features = text_feature_extractor(df)
            pose_features = pose_feature_extractor(df)
            
            print(f"Using {len(text_features)} text features and {len(pose_features)} pose features")
            
            # Prepare data
            data = self.prepare_data(
                df=df,
                target_col=target_col,
                text_features=text_features,
                pose_features=pose_features
            )
            
            # Train model
            train_result = self.train(
                data=data,
                target_col=target_col,
                **train_kwargs
            )
            
            # Evaluate model
            eval_result = self.evaluate(
                target_col=target_col,
                test_dataset=data["test_dataset"]
            )
            
            # Save model
            self.save(target_col)
            
            # Store results
            results[target_col] = {
                "train_result": train_result,
                "eval_result": eval_result,
                "feature_info": {
                    "text_features": text_features,
                    "pose_features": pose_features,
                    "num_classes": data["num_classes"]
                }
            }
            
            print(f"\nModel for {target_col}:")
            print(f"  Accuracy: {eval_result['accuracy']:.4f}")
            print(f"  Precision: {eval_result['precision']:.4f}")
            print(f"  Recall: {eval_result['recall']:.4f}")
            print(f"  F1 Score: {eval_result['f1']:.4f}")
        
        return results
    
    def evaluate(self, 
                target_col: str,
                test_dataset: Optional[TensorDataset] = None,
                batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            target_col: Target column name
            test_dataset: Test dataset (if None, uses the one from prepare_data)
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation results
        """
        # Get model
        if target_col not in self.models:
            raise ValueError(f"Model for target '{target_col}' not found. Train the model first.")
        
        model = self.models[target_col]
        
        # Use provided dataset or the one from data preparation
        if test_dataset is None:
            raise ValueError("Test dataset must be provided")
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Run evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for text_features, pose_features, labels in test_loader:
                # Move data to device
                text_features = text_features.to(self.device)
                pose_features = pose_features.to(self.device)
                
                # Forward pass
                outputs = model(text_features, pose_features)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Get class names
        encoder = self.target_encoders[target_col]
        class_names = encoder.classes_
        
        # Plot confusion matrix
        cm_fig_path = f"{self.model_dir}/{target_col}_confusion_matrix.png"
        plot_confusion_matrix(
            all_labels, all_preds, 
            labels=list(range(len(class_names))),
            title=f"{target_col} Confusion Matrix",
            save_path=cm_fig_path
        )
        
        # Plot classification report
        cr_fig_path = f"{self.model_dir}/{target_col}_classification_report.png"
        plot_classification_report(
            all_labels, all_preds,
            labels=list(range(len(class_names))),
            title=f"{target_col} Classification Report",
            save_path=cr_fig_path
        )
        
        # Log to MLflow if enabled
        if self.log_mlflow:
            with mlflow.start_run() as run:
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1", f1)
                
                mlflow.log_artifact(cm_fig_path)
                mlflow.log_artifact(cr_fig_path)
        
        # Return evaluation results
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_preds,
            "true_labels": all_labels,
            "class_names": class_names
        }
    
    def predict(self, 
               df: pd.DataFrame,
               target_col: str,
               text_features: List[str],
               pose_features: List[str],
               batch_size: int = 32) -> np.ndarray:
        """
        Make predictions with a trained model
        
        Args:
            df: Input DataFrame
            target_col: Target column name (model to use)
            text_features: List of text feature columns
            pose_features: List of pose feature columns
            batch_size: Batch size
            
        Returns:
            Array of predictions
        """
        # Check if model exists
        if target_col not in self.models:
            raise ValueError(f"Model for target '{target_col}' not found. Train the model first.")
        
        model = self.models[target_col]
        
        # Extract features
        if not text_features:
            # Use dummy feature if no text features provided
            X_text = np.zeros((len(df), 1))
        else:
            X_text = df[text_features].fillna(0).values
            
        if not pose_features:
            # Use dummy feature if no pose features provided
            X_pose = np.zeros((len(df), 1))
        else:
            X_pose = df[pose_features].fillna(0).values
        
        # Scale features
        if f"text_{target_col}" in self.scalers:
            text_scaler = self.scalers[f"text_{target_col}"]
            X_text_scaled = text_scaler.transform(X_text)
        else:
            X_text_scaled = X_text
            
        if f"pose_{target_col}" in self.scalers:
            pose_scaler = self.scalers[f"pose_{target_col}"]
            X_pose_scaled = pose_scaler.transform(X_pose)
        else:
            X_pose_scaled = X_pose
        
        # Convert to PyTorch tensors
        text_tensor = torch.tensor(X_text_scaled, dtype=torch.float32)
        pose_tensor = torch.tensor(X_pose_scaled, dtype=torch.float32)
        
        # Create dataset and loader
        dataset = TensorDataset(text_tensor, pose_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        # Make predictions
        model.eval()
        all_preds = []
        
        with torch.no_grad():
            for text_features, pose_features in loader:
                # Move data to device
                text_features = text_features.to(self.device)
                pose_features = pose_features.to(self.device)
                
                # Forward pass
                outputs = model(text_features, pose_features)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions
                all_preds.extend(predicted.cpu().numpy())
        
        return np.array(all_preds)