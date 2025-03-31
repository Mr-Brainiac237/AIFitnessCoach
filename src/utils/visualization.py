# src/utils/visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Union
import cv2

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         labels: List[str] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix for classification results
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    
    # Labels and title
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    # Ensure labels are visible
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_classification_report(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              labels: List[str] = None,
                              title: str = "Classification Report",
                              figsize: Tuple[int, int] = (12, 10),
                              save_path: Optional[str] = None) -> None:
    """
    Plot classification report as a heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Get classification report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    
    # Convert to DataFrame for easier plotting
    report_df = pd.DataFrame(report).T
    
    # Drop unnecessary rows
    if 'accuracy' in report_df.index:
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu")
    
    # Labels and title
    plt.title(title)
    
    # Ensure labels are visible
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Classification report saved to {save_path}")
    
    plt.show()

def plot_feature_importance(feature_names: List[str], 
                           importances: np.ndarray,
                           title: str = "Feature Importance",
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: Names of features
        importances: Importance scores
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    
    # Labels and title
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    # Ensure labels are visible
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def plot_training_curves(train_loss: List[float], 
                        val_loss: List[float],
                        train_metric: Optional[List[float]] = None,
                        val_metric: Optional[List[float]] = None,
                        metric_name: str = "Accuracy",
                        title: str = "Training Curves",
                        figsize: Tuple[int, int] = (14, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Plot training and validation curves
    
    Args:
        train_loss: Training loss values
        val_loss: Validation loss values
        train_metric: Optional training metric values
        val_metric: Optional validation metric values
        metric_name: Name of the metric
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=figsize)
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metric if provided
    if train_metric and val_metric:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_metric, 'b-', label=f'Training {metric_name}')
        plt.plot(epochs, val_metric, 'r-', label=f'Validation {metric_name}')
        plt.title(f'{title} - {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
    
    # Ensure labels are visible
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def visualize_pca(features: np.ndarray, 
                 labels: np.ndarray,
                 label_names: Optional[List[str]] = None,
                 n_components: int = 2,
                 title: str = "PCA Visualization",
                 figsize: Tuple[int, int] = (10, 8),
                 save_path: Optional[str] = None) -> None:
    """
    Visualize data using PCA
    
    Args:
        features: Feature matrix
        labels: Labels
        label_names: Optional list of label names
        n_components: Number of PCA components
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    
    # Create DataFrame
    df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1] if n_components >= 2 else np.zeros(len(pca_result)),
        'Label': labels
    })
    
    # Plot
    plt.figure(figsize=figsize)
    
    if n_components >= 2:
        ax = sns.scatterplot(x='PCA1', y='PCA2', hue='Label', data=df, palette='viridis')
        if label_names:
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, label_names)
    else:
        sns.histplot(x='PCA1', hue='Label', data=df, palette='viridis', element='step')
    
    # Labels and title
    plt.title(f'{title} (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
    
    # Ensure labels are visible
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"PCA visualization saved to {save_path}")
    
    plt.show()

def visualize_exercise_landmarks(image_path: str, 
                               landmarks: np.ndarray,
                               figsize: Tuple[int, int] = (8, 8),
                               save_path: Optional[str] = None) -> None:
    """
    Visualize pose landmarks on an exercise image
    
    Args:
        image_path: Path to the image
        landmarks: Landmark coordinates (shape: [33, 3] or [33, 4])
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width, _ = image.shape
    
    # Reshape landmarks if needed
    if landmarks.ndim == 1:
        if len(landmarks) == 33 * 4:  # x, y, z, visibility format
            landmarks = landmarks.reshape(33, 4)
        elif len(landmarks) == 33 * 3:  # x, y, z format
            landmarks = landmarks.reshape(33, 3)
        else:
            raise ValueError("Unexpected landmark shape")
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(landmarks.shape[0]):
        x = int(landmarks[i, 0] * width)
        y = int(landmarks[i, 1] * height)
        vis = landmarks[i, 3] if landmarks.shape[1] >= 4 else 1.0
        points.append((x, y, vis))
    
    # Define connections for visualization
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Shoulders to hips
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Arms
        (11, 13), (13, 15), (12, 14), (14, 16),
        # Hands
        (15, 17), (15, 19), (15, 21), (17, 19),
        (16, 18), (16, 20), (16, 22), (18, 20),
        # Legs
        (23, 25), (25, 27), (24, 26), (26, 28),
        # Feet
        (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32)
    ]
    
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    
    # Draw landmarks
    for i, (x, y, vis) in enumerate(points):
        if vis > 0.5:  # Only draw visible landmarks
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(annotated_image, str(i), (x + 5, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw connections
    for connection in connections:
        idx1, idx2 = connection
        if idx1 < len(points) and idx2 < len(points):
            x1, y1, vis1 = points[idx1]
            x2, y2, vis2 = points[idx2]
            if vis1 > 0.5 and vis2 > 0.5:  # Only connect visible landmarks
                cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Display the image
    plt.figure(figsize=figsize)
    plt.imshow(annotated_image)
    plt.title('Exercise Pose Landmarks')
    plt.axis('off')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Landmark visualization saved to {save_path}")
    
    plt.show()

def plot_attribute_distribution(df: pd.DataFrame, 
                              attribute: str,
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None) -> None:
    """
    Plot distribution of an exercise attribute
    
    Args:
        df: DataFrame with exercise data
        attribute: Attribute column name
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in DataFrame")
    
    plt.figure(figsize=figsize)
    
    # Count values
    value_counts = df[attribute].value_counts()
    
    # Plot
    ax = sns.barplot(x=value_counts.index, y=value_counts.values)
    
    # Add count labels
    for i, count in enumerate(value_counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    # Labels and title
    plt.title(title or f'Distribution of {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Ensure labels are visible
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Distribution plot saved to {save_path}")
    
    plt.show()

def plot_relationship(df: pd.DataFrame, 
                     x: str, 
                     y: str,
                     hue: Optional[str] = None,
                     kind: str = 'scatter',
                     title: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 6),
                     save_path: Optional[str] = None) -> None:
    """
    Plot relationship between two attributes
    
    Args:
        df: DataFrame with exercise data
        x: X-axis attribute
        y: Y-axis attribute
        hue: Optional attribute for color grouping
        kind: Plot type ('scatter', 'line', 'bar', 'box', 'violin', 'heatmap')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    
    if kind == 'scatter':
        sns.scatterplot(x=x, y=y, hue=hue, data=df)
    elif kind == 'line':
        sns.lineplot(x=x, y=y, hue=hue, data=df)
    elif kind == 'bar':
        sns.barplot(x=x, y=y, hue=hue, data=df)
    elif kind == 'box':
        sns.boxplot(x=x, y=y, hue=hue, data=df)
    elif kind == 'violin':
        sns.violinplot(x=x, y=y, hue=hue, data=df)
    elif kind == 'heatmap':
        if hue:
            pivot_table = pd.crosstab(df[x], df[y], df[hue], aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
        else:
            pivot_table = pd.crosstab(df[x], df[y])
            sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
    else:
        raise ValueError(f"Unsupported plot kind: {kind}")
    
    # Labels and title
    plt.title(title or f'Relationship between {x} and {y}')
    
    # Ensure labels are visible
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        print(f"Relationship plot saved to {save_path}")
    
    plt.show()

def plot_feature_correlations(df: pd.DataFrame, 
                             features: List[str],
                             target: Optional[str] = None,
                             top_n: int = 15,
                             figsize: Tuple[int, int] = (12, 10),
                             save_path: Optional[str] = None) -> None:
    """
    Plot feature correlations
    
    Args:
        df: DataFrame with feature data
        features: List of feature columns
        target: Optional target column
        top_n: Number of top correlations to show
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Select features
    if target:
        selected_columns = features + [target]
    else:
        selected_columns = features
    
    # Calculate correlation matrix
    corr_matrix = df[selected_columns].corr()
    
    if target:
        # Get correlations with target
        target_corr = corr_matrix[target].drop(target)
        
        # Get top N correlations
        top_corr = target_corr.abs().sort_values(ascending=False).head(top_n)
        
        # Plot correlations with target
        plt.figure(figsize=(figsize[0], figsize[1] // 2))
        sns.barplot(x=top_corr.values, y=top_corr.index)
        plt.title(f'Top {top_n} Correlations with {target}')
        plt.xlabel('Correlation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_target_corr.png")
            print(f"Target correlation plot saved to {save_path}_target_corr.png")
        
        plt.show()
    
    # Plot correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_corr_matrix.png")
        print(f"Correlation matrix saved to {save_path}_corr_matrix.png")
    
    plt.show()
