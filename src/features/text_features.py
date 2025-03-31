# src/features/text_features.py
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import spacy
from sentence_transformers import SentenceTransformer

class TextFeatureExtractor:
    """
    Extract features from exercise text data (names and instructions)
    """
    def __init__(self, 
                 use_tfidf: bool = True,
                 use_embeddings: bool = False,
                 use_topic_modeling: bool = False,
                 max_features: int = 100,
                 embedding_model: str = 'paraphrase-MiniLM-L6-v2'):
        """
        Initialize the text feature extractor
        
        Args:
            use_tfidf: Whether to use TF-IDF features
            use_embeddings: Whether to use sentence embeddings
            use_topic_modeling: Whether to use topic modeling
            max_features: Maximum number of TF-IDF features
            embedding_model: Sentence transformer model name
        """
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings
        self.use_topic_modeling = use_topic_modeling
        self.max_features = max_features
        self.embedding_model_name = embedding_model
        
        # Initialize feature extractors
        if self.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
        
        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except:
                print(f"Warning: Could not load embedding model {embedding_model}. Disabling embeddings.")
                self.use_embeddings = False
        
        if self.use_topic_modeling:
            self.count_vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english'
            )
            self.lda_model = LatentDirichletAllocation(
                n_components=10,
                random_state=42
            )
        
        # Load spaCy model for text preprocessing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Warning: Could not load spaCy model. Attempting to download...")
            try:
                import os
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print("Warning: Could not download spaCy model. Using basic preprocessing.")
                self.nlp = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using spaCy
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Basic preprocessing if spaCy is not available
        if self.nlp is None:
            return text.lower()
        
        # SpaCy preprocessing
        doc = self.nlp(text)
        
        # Extract lemmas, filter stop words and punctuation
        tokens = [token.lemma_.lower() for token in doc 
                  if not token.is_stop and not token.is_punct and token.lemma_.strip()]
        
        return " ".join(tokens)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract text features from a DataFrame
        
        Args:
            df: Input DataFrame with 'name' and 'instructions' columns
            
        Returns:
            DataFrame with added text features
        """
        # Prepare combined text
        if 'instructions' in df.columns and 'name' in df.columns:
            df['text_combined'] = df['name'] + ' ' + df['instructions'].fillna('')
        elif 'name' in df.columns:
            df['text_combined'] = df['name']
        else:
            raise ValueError("DataFrame must contain 'name' column")
        
        # Preprocess text
        print("Preprocessing text...")
        df['text_preprocessed'] = df['text_combined'].apply(self.preprocess_text)
        
        result_df = df.copy()
        
        # Extract TF-IDF features
        if self.use_tfidf:
            print("Extracting TF-IDF features...")
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text_preprocessed'])
            
            # Convert to DataFrame
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(), 
                columns=[f"tfidf_{f}" for f in feature_names]
            )
            
            # Add to result DataFrame
            result_df = pd.concat([result_df, tfidf_df], axis=1)
        
        # Extract sentence embeddings
        if self.use_embeddings:
            print("Generating sentence embeddings...")
            embeddings = self.embedding_model.encode(
                df['text_combined'].tolist(), 
                show_progress_bar=True
            )
            
            # Convert to DataFrame
            embedding_df = pd.DataFrame(
                embeddings, 
                columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
            )
            
            # Add to result DataFrame
            result_df = pd.concat([result_df, embedding_df], axis=1)
        
        # Extract topic modeling features
        if self.use_topic_modeling:
            print("Extracting topic modeling features...")
            count_matrix = self.count_vectorizer.fit_transform(df['text_preprocessed'])
            
            # Fit LDA model
            self.lda_model.fit(count_matrix)
            
            # Transform data
            topic_matrix = self.lda_model.transform(count_matrix)
            
            # Convert to DataFrame
            topic_df = pd.DataFrame(
                topic_matrix,
                columns=[f"topic_{i}" for i in range(topic_matrix.shape[1])]
            )
            
            # Add to result DataFrame
            result_df = pd.concat([result_df, topic_df], axis=1)
        
        return result_df
    
    def extract_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract keyword-based features from text
        
        Args:
            df: Input DataFrame with 'name' and 'instructions' columns
            
        Returns:
            DataFrame with added keyword features
        """
        result_df = df.copy()
        
        # Define keyword dictionaries for various categories
        keywords = {
            'compound_keywords': ['compound', 'multi-joint', 'full body', 'complex'],
            'isolation_keywords': ['isolation', 'single-joint', 'isolate', 'focus'],
            'beginner_keywords': ['beginner', 'basic', 'simple', 'easy', 'starter'],
            'advanced_keywords': ['advanced', 'difficult', 'challenging', 'expert'],
            'high_intensity_keywords': ['high intensity', 'explosive', 'power', 'maximum'],
            'low_intensity_keywords': ['low intensity', 'gentle', 'moderate', 'light'],
            'strength_keywords': ['strength', 'power', 'force', 'heavy'],
            'hypertrophy_keywords': ['hypertrophy', 'muscle growth', 'build muscle', 'volume'],
            'cardio_keywords': ['cardio', 'cardiovascular', 'aerobic', 'heart rate'],
            'flexibility_keywords': ['flexibility', 'stretch', 'mobility', 'range of motion'],
            'risky_keywords': ['caution', 'careful', 'risk', 'injury', 'dangerous'],
            'safe_keywords': ['safe', 'beginner-friendly', 'low-impact']
        }
        
        # Create keyword feature columns
        for category, words in keywords.items():
            result_df[f'kw_{category}'] = result_df['text_combined'].apply(
                lambda x: sum(1 for word in words if word in str(x).lower())
            )
        
        return result_df
    
    def extract_exercise_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract exercise attributes based on text analysis
        
        Args:
            df: Input DataFrame with text features
            
        Returns:
            DataFrame with extracted exercise attributes
        """
        result_df = df.copy()
        
        # Extract movement patterns from exercise names and descriptions
        result_df['movement_pattern'] = result_df.apply(
            lambda row: self._extract_movement_pattern(
                row['name'] if 'name' in row else "", 
                row['instructions'] if 'instructions' in row else ""
            ), 
            axis=1
        )
        
        # Extract joints used from descriptions
        result_df['joints_used'] = result_df.apply(
            lambda row: self._extract_joints_used(
                row['instructions'] if 'instructions' in row else ""
            ), 
            axis=1
        )
        
        # Determine exercise type
        result_df['exercise_type'] = result_df.apply(
            lambda row: self._determine_exercise_type(
                row['text_combined'] if 'text_combined' in row else "",
                row['kw_strength_keywords'] if 'kw_strength_keywords' in row else 0,
                row['kw_hypertrophy_keywords'] if 'kw_hypertrophy_keywords' in row else 0,
                row['kw_cardio_keywords'] if 'kw_cardio_keywords' in row else 0
            ),
            axis=1
        )
        
        # Determine intensity level
        result_df['intensity_level'] = result_df.apply(
            lambda row: self._determine_intensity(
                row['text_combined'] if 'text_combined' in row else "",
                row['kw_beginner_keywords'] if 'kw_beginner_keywords' in row else 0,
                row['kw_advanced_keywords'] if 'kw_advanced_keywords' in row else 0,
                row['kw_high_intensity_keywords'] if 'kw_high_intensity_keywords' in row else 0,
                row['kw_low_intensity_keywords'] if 'kw_low_intensity_keywords' in row else 0
            ),
            axis=1
        )
        
        # Determine risk assessment
        result_df['risk_assessment'] = result_df.apply(
            lambda row: self._assess_risk(
                row['text_combined'] if 'text_combined' in row else "",
                row['kw_risky_keywords'] if 'kw_risky_keywords' in row else 0,
                row['kw_safe_keywords'] if 'kw_safe_keywords' in row else 0,
                row['movement_pattern'] if 'movement_pattern' in row else "",
                row['joints_used'] if 'joints_used' in row else []
            ),
            axis=1
        )
        
        return result_df
    
    def _extract_movement_pattern(self, name: str, instructions: str) -> str:
        """
        Extract movement pattern from exercise name and instructions
        
        Args:
            name: Exercise name
            instructions: Exercise instructions
            
        Returns:
            Movement pattern category
        """
        name = name.lower() if isinstance(name, str) else ""
        instructions = instructions.lower() if isinstance(instructions, str) else ""
        text = name + " " + instructions
        
        # Basic pattern matching
        if any(word in text for word in ['squat', 'lunge', 'step', 'leg press']):
            return 'squat'
        elif any(word in text for word in ['hinge', 'deadlift', 'hip thrust', 'good morning', 'swing']):
            return 'hinge'
        elif any(word in text for word in ['push', 'press', 'bench', 'chest', 'shoulder press', 'overhead']):
            return 'push'
        elif any(word in text for word in ['pull', 'row', 'chin', 'pull-up', 'pullup', 'pulldown', 'lat']):
            return 'pull'
        elif any(word in text for word in ['rotation', 'twist', 'turn', 'russian twist', 'woodchop']):
            return 'rotation'
        elif any(word in text for word in ['carry', 'walk', 'farmer', 'suitcase', 'turkish']):
            return 'carry'
        elif any(word in text for word in ['plank', 'bridge', 'hold', 'isometric', 'hollow']):
            return 'isometric'
        elif any(word in text for word in ['jump', 'hop', 'bound', 'plyometric', 'explosive']):
            return 'plyometric'
        elif any(word in text for word in ['curl', 'extension', 'raise', 'isolation']):
            return 'isolation'
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
            'hip': ['hip', 'glute', 'buttock', 'gluteal'],
            'knee': ['knee', 'quad', 'hamstring'],
            'ankle': ['ankle', 'calf', 'shin'],
            'spine': ['spine', 'back', 'lumbar', 'thoracic', 'cervical', 'neck', 'core', 'abdominal']
        }
        
        for joint, keywords in joint_keywords.items():
            if any(keyword in instructions for keyword in keywords):
                joints.append(joint)
                
        return joints
    
    def _determine_exercise_type(self, text: str, strength_score: int, 
                                hypertrophy_score: int, cardio_score: int) -> str:
        """
        Determine exercise type based on text and keyword scores
        
        Args:
            text: Combined text
            strength_score: Strength keyword score
            hypertrophy_score: Hypertrophy keyword score
            cardio_score: Cardio keyword score
            
        Returns:
            Exercise type
        """
        text = text.lower() if isinstance(text, str) else ""
        
        # Check for explicit mentions
        if 'strength' in text or 'power' in text:
            return 'strength'
        elif 'hypertrophy' in text or 'muscle growth' in text or 'build muscle' in text:
            return 'hypertrophy'
        elif 'cardio' in text or 'aerobic' in text or 'endurance' in text:
            return 'cardio'
        elif 'flexibility' in text or 'mobility' in text or 'stretch' in text:
            return 'flexibility'
        
        # Use keyword scores
        scores = {
            'strength': strength_score,
            'hypertrophy': hypertrophy_score,
            'cardio': cardio_score
        }
        
        max_type = max(scores, key=scores.get)
        if scores[max_type] > 0:
            return max_type
        
        # Default based on common exercise patterns
        if any(word in text for word in ['rep', 'weight', 'resistance', 'dumbbell', 'barbell']):
            # Default to hypertrophy for weighted exercises with no clear indication
            return 'hypertrophy'
        elif any(word in text for word in ['minute', 'sprint', 'run', 'jumping']):
            return 'cardio'
        else:
            return 'strength'  # Default
    
    def _determine_intensity(self, text: str, beginner_score: int, advanced_score: int, 
                            high_intensity_score: int, low_intensity_score: int) -> str:
        """
        Determine intensity level based on text and keyword scores
        
        Args:
            text: Combined text
            beginner_score: Beginner keyword score
            advanced_score: Advanced keyword score
            high_intensity_score: High intensity keyword score
            low_intensity_score: Low intensity keyword score
            
        Returns:
            Intensity level (beginner, intermediate, advanced)
        """
        text = text.lower() if isinstance(text, str) else ""
        
        # Check for explicit mentions
        if 'beginner' in text or 'easy' in text or 'basic' in text:
            return 'beginner'
        elif 'advanced' in text or 'difficult' in text or 'challenging' in text:
            return 'advanced'
        elif 'intermediate' in text or 'moderate' in text:
            return 'intermediate'
        
        # Use keyword scores
        if beginner_score > advanced_score or low_intensity_score > high_intensity_score:
            return 'beginner'
        elif advanced_score > beginner_score or high_intensity_score > low_intensity_score:
            return 'advanced'
        else:
            return 'intermediate'  # Default
    
    def _assess_risk(self, text: str, risky_score: int, safe_score: int, 
                    movement_pattern: str, joints_used: List[str]) -> str:
        """
        Assess exercise risk based on text, keyword scores, and other factors
        
        Args:
            text: Combined text
            risky_score: Risky keyword score
            safe_score: Safe keyword score
            movement_pattern: Movement pattern category
            joints_used: List of joints used
            
        Returns:
            Risk assessment (low, medium, high)
        """
        text = text.lower() if isinstance(text, str) else ""
        joints_used = joints_used if isinstance(joints_used, list) else []
        
        # Check for explicit mentions
        if any(word in text for word in ['dangerous', 'caution', 'careful', 'injury']):
            return 'high'
        elif any(word in text for word in ['safe', 'beginner-friendly', 'low impact']):
            return 'low'
        
        # Use keyword scores
        if risky_score > safe_score:
            return 'high'
        elif safe_score > risky_score:
            return 'low'
        
        # Assess based on movement pattern and joints used
        high_risk_patterns = ['hinge', 'rotation']
        medium_risk_patterns = ['squat', 'push', 'pull', 'plyometric']
        low_risk_patterns = ['isometric', 'carry', 'isolation']
        
        high_risk_joints = ['spine', 'shoulder']
        
        if movement_pattern in high_risk_patterns or any(joint in high_risk_joints for joint in joints_used):
            return 'high'
        elif movement_pattern in medium_risk_patterns:
            return 'medium'
        elif movement_pattern in low_risk_patterns:
            return 'low'
        else:
            return 'medium'  # Default
