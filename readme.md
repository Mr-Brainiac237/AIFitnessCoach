# Exercise Classification AI

A deep learning system for classifying exercises based on multiple attributes including target muscle group, equipment requirements, movement patterns, intensity levels, exercise types, quality metrics, and risk assessment.

## Features

- **Data Collection**: Fetches exercise data from ExerciseDB and Wger APIs
- **Data Preprocessing**: Cleans and processes exercise data, extracts features
- **Feature Engineering**: 
  - Text-based features from exercise names and instructions
  - Pose-based features from exercise images/GIFs
  - Derived attributes like movement patterns and joints used
- **Multi-attribute Classification**: 
  - Target Muscle Group
  - Equipment Necessary
  - Intensity/Experience Level
  - Exercise Type (Compound vs. isolation, strength vs. hypertrophy, etc.)
  - Quality of Movement
  - Movement Pattern
  - Joints Used
  - Risk Assessment
- **Visualization**: Tools for exploring and visualizing the data and model results

## Project Structure

```
project/
├── data/
│   ├── raw/                  # Raw data from APIs
│   ├── processed/            # Processed data ready for training
│   └── external/             # Additional datasets
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data/                 # Data collection and processing scripts
│   │   ├── __init__.py
│   │   ├── api_fetcher.py    # Scripts to fetch data from exercise APIs
│   │   └── preprocessor.py   # Data preprocessing utilities
│   ├── features/             # Feature extraction scripts
│   │   ├── __init__.py
│   │   ├── text_features.py  # Text-based feature extraction
│   │   ├── pose_features.py  # Pose/movement based features
│   │   └── combined.py       # Feature combination utilities
│   ├── models/               # Model definition and training
│   │   ├── __init__.py
│   │   ├── classifier.py     # Classification model definitions
│   │   └── trainer.py        # Training loops and utilities
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── visualization.py  # Visualization utilities
├── models/                   # Saved models
├── .env                      # Environment variables (API keys etc.)
├── requirements.txt          # Project dependencies
├── main.py                   # Main script
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mr-Brainiac237/AIFitnessCoach
   cd AIFitnessCoach
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   EXERCISEDB_API_KEY=your_exercisedb_api_key
   WGER_API_KEY=your_wger_api_key
   ```

## Usage

### 1. Data Collection

Fetch data from the exercise APIs:

```bash
python main.py --fetch
```

### 2. Data Preprocessing

Process the fetched data and extract features:

```bash
python main.py --preprocess
```

To also process exercise images (takes longer):

```bash
python main.py --preprocess --process_images --image_limit 1325
```

### 3. Model Training

Train classification models for different attributes:

```bash
python main.py --train
```

### 4. Making Predictions

Run predictions on new exercises:

```bash
python main.py --predict
```

### 5. Data Exploration

Explore the data and model results using Jupyter notebooks:

```bash
jupyter lab
```

Then open the notebooks in the `notebooks/` directory.

## Model Architecture

The system uses a multimodal neural network architecture that combines:

1. **Text Features**: Processes exercise names and instructions using TF-IDF and text embeddings
2. **Pose Features**: Analyzes exercise form and movements using MediaPipe pose estimation
3. **Combined Classification**: Joins the features to classify exercises on multiple dimensions

## Evaluation Metrics

The model's performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrices

## Future Improvements

- Implement more sophisticated NLP techniques (transformers)
- Add temporal pose analysis for better movement understanding
- Create a web interface for easy classification of new exercises
- Expand the database with more exercises and attributes
- Add explainability tools to understand model decisions

## Requirements

- Python 3.8+
- PyTorch 1.9+
- MediaPipe 0.8.7+
- Other requirements in `requirements.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ExerciseDB API](https://rapidapi.com/justin-WFnsXH_t6/api/exercisedb/) for exercise data
- [Wger REST API](https://wger.de/en/software/api) for additional exercise information
- [MediaPipe](https://developers.google.com/mediapipe) for pose estimation
