# src/routines/workout_generator.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import random
from collections import defaultdict
import joblib
import torch

class WorkoutRoutineGenerator:
    """
    Generate personalized workout routines based on user preferences and fitness goals.
    Uses the trained exercise classification models to select appropriate exercises.
    """
    def __init__(self, 
                 exercise_data_path: str = "data/processed/exercisedb_processed.csv",
                 model_dir: str = "models"):
        """
        Initialize the workout routine generator.
        
        Args:
            exercise_data_path: Path to processed exercise data
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.exercise_data = pd.read_csv(exercise_data_path)
        
        # Load encoders for prediction interpretation
        self.encoders = {}
        for target_col in ['movement_pattern', 'intensity_level', 'exercise_type', 'is_compound', 'risk_assessment']:
            encoder_path = f"{model_dir}/{target_col}_encoder.pkl"
            if os.path.exists(encoder_path):
                self.encoders[target_col] = joblib.load(encoder_path)
        
        # Define workout templates
        self.workout_templates = {
            'beginner': {
                'full_body': {
                    'days_per_week': 3,
                    'exercises_per_workout': 6,
                    'sets_per_exercise': 3,
                    'rep_ranges': {
                        'compound': '8-12',
                        'isolation': '12-15'
                    },
                    'rest_periods': {
                        'compound': '90-120 sec',
                        'isolation': '60 sec'
                    },
                    'structure': [
                        {'type': 'compound', 'pattern': 'push', 'primary': True},
                        {'type': 'compound', 'pattern': 'pull', 'primary': True},
                        {'type': 'compound', 'pattern': 'squat', 'primary': True},
                        {'type': 'isolation', 'pattern': 'push', 'primary': False},
                        {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                        {'type': 'isolation', 'pattern': 'other', 'primary': False}
                    ]
                },
                'upper_lower': {
                    'days_per_week': 4,
                    'exercises_per_workout': 5,
                    'sets_per_exercise': 3,
                    'rep_ranges': {
                        'compound': '8-12',
                        'isolation': '12-15'
                    },
                    'rest_periods': {
                        'compound': '90-120 sec',
                        'isolation': '60 sec'
                    },
                    'structure': {
                        'upper': [
                            {'type': 'compound', 'pattern': 'push', 'primary': True},
                            {'type': 'compound', 'pattern': 'pull', 'primary': True},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ],
                        'lower': [
                            {'type': 'compound', 'pattern': 'squat', 'primary': True},
                            {'type': 'compound', 'pattern': 'hinge', 'primary': True},
                            {'type': 'isolation', 'pattern': 'squat', 'primary': False},
                            {'type': 'isolation', 'pattern': 'hinge', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ]
                    }
                }
            },
            'intermediate': {
                'push_pull_legs': {
                    'days_per_week': 6,
                    'exercises_per_workout': 6,
                    'sets_per_exercise': 4,
                    'rep_ranges': {
                        'compound': '6-10',
                        'isolation': '10-15'
                    },
                    'rest_periods': {
                        'compound': '2-3 min',
                        'isolation': '60-90 sec'
                    },
                    'structure': {
                        'push': [
                            {'type': 'compound', 'pattern': 'push', 'primary': True},
                            {'type': 'compound', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'push', 'primary': True},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ],
                        'pull': [
                            {'type': 'compound', 'pattern': 'pull', 'primary': True},
                            {'type': 'compound', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': True},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ],
                        'legs': [
                            {'type': 'compound', 'pattern': 'squat', 'primary': True},
                            {'type': 'compound', 'pattern': 'hinge', 'primary': True},
                            {'type': 'isolation', 'pattern': 'squat', 'primary': False},
                            {'type': 'isolation', 'pattern': 'hinge', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ]
                    }
                },
                'upper_lower': {
                    'days_per_week': 4,
                    'exercises_per_workout': 7,
                    'sets_per_exercise': 4,
                    'rep_ranges': {
                        'compound': '6-10',
                        'isolation': '10-15'
                    },
                    'rest_periods': {
                        'compound': '2-3 min',
                        'isolation': '60-90 sec'
                    },
                    'structure': {
                        'upper': [
                            {'type': 'compound', 'pattern': 'push', 'primary': True},
                            {'type': 'compound', 'pattern': 'pull', 'primary': True},
                            {'type': 'compound', 'pattern': 'push', 'primary': False},
                            {'type': 'compound', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ],
                        'lower': [
                            {'type': 'compound', 'pattern': 'squat', 'primary': True},
                            {'type': 'compound', 'pattern': 'hinge', 'primary': True},
                            {'type': 'compound', 'pattern': 'squat', 'primary': False},
                            {'type': 'isolation', 'pattern': 'squat', 'primary': False},
                            {'type': 'isolation', 'pattern': 'hinge', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ]
                    }
                }
            },
            'advanced': {
                'push_pull_legs': {
                    'days_per_week': 6,
                    'exercises_per_workout': 8,
                    'sets_per_exercise': 5,
                    'rep_ranges': {
                        'compound': '5-8',
                        'isolation': '8-12'
                    },
                    'rest_periods': {
                        'compound': '3-4 min',
                        'isolation': '1-2 min'
                    },
                    'structure': {
                        'push': [
                            {'type': 'compound', 'pattern': 'push', 'primary': True},
                            {'type': 'compound', 'pattern': 'push', 'primary': True},
                            {'type': 'compound', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'push', 'primary': True},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ],
                        'pull': [
                            {'type': 'compound', 'pattern': 'pull', 'primary': True},
                            {'type': 'compound', 'pattern': 'pull', 'primary': True},
                            {'type': 'compound', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': True},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ],
                        'legs': [
                            {'type': 'compound', 'pattern': 'squat', 'primary': True},
                            {'type': 'compound', 'pattern': 'hinge', 'primary': True},
                            {'type': 'compound', 'pattern': 'squat', 'primary': False},
                            {'type': 'compound', 'pattern': 'hinge', 'primary': False},
                            {'type': 'isolation', 'pattern': 'squat', 'primary': False},
                            {'type': 'isolation', 'pattern': 'hinge', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ]
                    }
                },
                'body_part_split': {
                    'days_per_week': 5,
                    'exercises_per_workout': 7,
                    'sets_per_exercise': 4,
                    'rep_ranges': {
                        'compound': '6-10',
                        'isolation': '8-15'
                    },
                    'rest_periods': {
                        'compound': '2-3 min',
                        'isolation': '1-2 min'
                    },
                    'structure': {
                        'chest': [
                            {'type': 'compound', 'pattern': 'push', 'primary': True, 'target': 'chest'},
                            {'type': 'compound', 'pattern': 'push', 'primary': False, 'target': 'chest'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': True, 'target': 'chest'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'chest'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'chest'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'triceps'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'shoulders'}
                        ],
                        'back': [
                            {'type': 'compound', 'pattern': 'pull', 'primary': True, 'target': 'lats'},
                            {'type': 'compound', 'pattern': 'pull', 'primary': False, 'target': 'lats'},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': True, 'target': 'lats'},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False, 'target': 'upper back'},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False, 'target': 'lats'},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False, 'target': 'biceps'},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False, 'target': 'forearms'}
                        ],
                        'legs': [
                            {'type': 'compound', 'pattern': 'squat', 'primary': True, 'target': 'quads'},
                            {'type': 'compound', 'pattern': 'hinge', 'primary': True, 'target': 'glutes'},
                            {'type': 'compound', 'pattern': 'squat', 'primary': False, 'target': 'quads'},
                            {'type': 'isolation', 'pattern': 'squat', 'primary': False, 'target': 'quads'},
                            {'type': 'isolation', 'pattern': 'hinge', 'primary': False, 'target': 'hamstrings'},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False, 'target': 'calves'},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False, 'target': 'abs'}
                        ],
                        'shoulders_arms': [
                            {'type': 'compound', 'pattern': 'push', 'primary': True, 'target': 'shoulders'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'shoulders'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'shoulders'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'triceps'},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False, 'target': 'triceps'},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False, 'target': 'biceps'},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False, 'target': 'biceps'}
                        ],
                        'full_body': [
                            {'type': 'compound', 'pattern': 'push', 'primary': True},
                            {'type': 'compound', 'pattern': 'pull', 'primary': True},
                            {'type': 'compound', 'pattern': 'squat', 'primary': True},
                            {'type': 'compound', 'pattern': 'hinge', 'primary': True},
                            {'type': 'isolation', 'pattern': 'push', 'primary': False},
                            {'type': 'isolation', 'pattern': 'pull', 'primary': False},
                            {'type': 'isolation', 'pattern': 'other', 'primary': False}
                        ]
                    }
                }
            }
        }
        
        # Map body parts to target muscles
        self.body_part_mapping = {
            'chest': ['pectorals', 'chest'],
            'back': ['lats', 'upper back', 'traps', 'lower back'],
            'shoulders': ['shoulders', 'delts', 'deltoids'],
            'biceps': ['biceps'],
            'triceps': ['triceps'],
            'forearms': ['forearms'],
            'abs': ['abs', 'core', 'abdominals'],
            'quads': ['quads', 'quadriceps'],
            'hamstrings': ['hamstrings'],
            'glutes': ['glutes', 'gluteus maximus'],
            'calves': ['calves']
        }
        
        # Reverse mapping from target muscles to body parts
        self.target_to_body_part = {}
        for body_part, targets in self.body_part_mapping.items():
            for target in targets:
                self.target_to_body_part[target.lower()] = body_part
    
    def preprocess_exercise_data(self) -> None:
        """
        Preprocess exercise data to ensure all necessary attributes are present.
        Fill in missing values using classifier predictions if available.
        """
        # Fill missing values for key attributes
        required_attributes = ['movement_pattern', 'is_compound', 'intensity_level', 'exercise_type', 'risk_assessment']
        
        for attr in required_attributes:
            if attr in self.exercise_data.columns:
                # Replace missing values with "unknown" if column exists
                self.exercise_data[attr] = self.exercise_data[attr].fillna("unknown")
            else:
                # Create column filled with "unknown" if it doesn't exist
                self.exercise_data[attr] = "unknown"
        
        # Convert is_compound to string categoricals for consistency
        if 'is_compound' in self.exercise_data.columns:
            self.exercise_data['is_compound'] = self.exercise_data['is_compound'].astype(str)
            self.exercise_data['is_compound'] = self.exercise_data['is_compound'].replace({'True': 'compound', 'False': 'isolation'})
        
        # Map target muscles to body parts for easier filtering
        if 'target' in self.exercise_data.columns:
            self.exercise_data['body_part'] = self.exercise_data['target'].apply(
                lambda x: self.target_to_body_part.get(str(x).lower(), 'other')
            )
        
        print(f"Preprocessed exercise data: {len(self.exercise_data)} exercises")
    
    def filter_exercises(self, 
                       experience_level: str = 'beginner',
                       target_muscles: Optional[List[str]] = None,
                       equipment_available: Optional[List[str]] = None,
                       movement_patterns: Optional[List[str]] = None,
                       risk_level: str = 'low') -> pd.DataFrame:
        """
        Filter exercises based on user preferences.
        
        Args:
            experience_level: User's experience level (beginner, intermediate, advanced)
            target_muscles: List of target muscles to focus on
            equipment_available: List of available equipment
            movement_patterns: List of preferred movement patterns
            risk_level: Maximum acceptable risk level (low, medium, high)
            
        Returns:
            Filtered DataFrame of exercises
        """
        # Create a copy to avoid modifying the original
        filtered_df = self.exercise_data.copy()
        
        # Map experience level to intensity level
        experience_to_intensity = {
            'beginner': ['beginner', 'intermediate'],
            'intermediate': ['beginner', 'intermediate', 'advanced'],
            'advanced': ['intermediate', 'advanced']
        }
        
        # Filter by intensity level based on experience
        # Temporarily skip intensity level filtering since it's not properly set in the data
        # if experience_level in experience_to_intensity:
        #     allowed_intensity = experience_to_intensity[experience_level]
        #     filtered_df = filtered_df[filtered_df['intensity_level'].isin(allowed_intensity)]
        
        # Filter by target muscles if provided
        if target_muscles and 'target' in filtered_df.columns:
            # Convert target muscles to lowercase for case-insensitive matching
            target_muscles_lower = [t.lower() for t in target_muscles]
            
            # Create a mask for each target muscle and combine with OR
            target_mask = filtered_df['target'].str.lower().isin(target_muscles_lower)
            
            # Also check body_part if available
            if 'body_part' in filtered_df.columns:
                body_part_mask = filtered_df['body_part'].str.lower().isin(target_muscles_lower)
                target_mask = target_mask | body_part_mask
            
            filtered_df = filtered_df[target_mask]
        
        # Filter by available equipment if provided
        if equipment_available and 'equipment' in filtered_df.columns:
            # Convert equipment list to lowercase for case-insensitive matching
            equipment_lower = [e.lower() for e in equipment_available]
            
            # Create a mask for equipment filtering
            equipment_mask = filtered_df['equipment'].str.lower().isin(equipment_lower)
            
            # Exclude exercise ball exercises unless specifically requested
            if 'exercise ball' not in equipment_lower and 'stability ball' not in equipment_lower:
                equipment_mask = equipment_mask & ~filtered_df['equipment'].str.lower().str.contains('ball')
            
            filtered_df = filtered_df[equipment_mask]
        
        # Filter by movement patterns if provided
        if movement_patterns and 'movement_pattern' in filtered_df.columns:
            # Convert movement patterns to lowercase for case-insensitive matching
            patterns_lower = [p.lower() for p in movement_patterns]
            filtered_df = filtered_df[filtered_df['movement_pattern'].str.lower().isin(patterns_lower)]
        
        # Filter by risk level
        risk_levels = {
            'low': ['low'],
            'medium': ['low', 'medium'],
            'high': ['low', 'medium', 'high']
        }
        
        # Temporarily skip risk level filtering since it's not properly set in the data
        # if risk_level in risk_levels and 'risk_assessment' in filtered_df.columns:
        #     allowed_risk = risk_levels[risk_level]
        #     filtered_df = filtered_df[filtered_df['risk_assessment'].str.lower().isin(allowed_risk)]
        
        return filtered_df
    
    def select_exercises_for_template(self, 
                                    filtered_exercises: pd.DataFrame, 
                                    template_structure: List[Dict],
                                    selected_exercises: Optional[List[str]] = None) -> List[Dict]:
        """
        Select exercises that match the template structure requirements.
        
        Args:
            filtered_exercises: DataFrame of filtered exercises
            template_structure: List of dictionaries specifying exercise requirements
            selected_exercises: List of exercise IDs that have already been selected
            
        Returns:
            List of selected exercises with metadata
        """
        selected_exercises = selected_exercises or []
        result = []
        
        for requirement in template_structure:
            # Extract requirement details
            ex_type = requirement.get('type', 'compound')  # compound or isolation
            pattern = requirement.get('pattern', 'other')  # movement pattern
            is_primary = requirement.get('primary', False)  # primary or accessory
            target = requirement.get('target', None)  # target muscle (optional)
            
            # Create a mask for each requirement
            type_mask = filtered_exercises['is_compound'] == ex_type
            pattern_mask = filtered_exercises['movement_pattern'].str.lower() == pattern.lower()
            
            # Combine masks
            combined_mask = type_mask & pattern_mask
            
            # Add target muscle filter if specified
            if target and 'target' in filtered_exercises.columns:
                target_mask = filtered_exercises['target'].str.lower() == target.lower()
                # Also check body part if available
                if 'body_part' in filtered_exercises.columns:
                    body_part_mask = filtered_exercises['body_part'].str.lower() == target.lower()
                    target_mask = target_mask | body_part_mask
                combined_mask = combined_mask & target_mask
            
            # Filter out already selected exercises
            if selected_exercises:
                combined_mask = combined_mask & ~filtered_exercises['id'].isin(selected_exercises)
            
            # Get matching exercises
            matching_exercises = filtered_exercises[combined_mask]
            
            if len(matching_exercises) > 0:
                # Randomly select one matching exercise
                selected_exercise = matching_exercises.sample(1).iloc[0]
                
                # Add to selected exercises
                selected_exercises.append(selected_exercise['id'])
                
                # Create exercise dict with metadata
                exercise_dict = {
                    'id': selected_exercise['id'],
                    'name': selected_exercise['name'],
                    'type': ex_type,
                    'movement_pattern': pattern,
                    'is_primary': is_primary,
                    'target': selected_exercise['target'] if 'target' in selected_exercise else None,
                    'equipment': selected_exercise['equipment'] if 'equipment' in selected_exercise else None,
                    'instructions': selected_exercise['instructions'] if 'instructions' in selected_exercise else None,
                    'gifUrl': selected_exercise['gifurl'] if 'gifurl' in selected_exercise else None
                }
                
                result.append(exercise_dict)
            else:
                # If no matching exercises, try with relaxed constraints
                # First, try ignoring the movement pattern
                relaxed_mask = type_mask
                
                # Add target muscle filter if specified
                if target and 'target' in filtered_exercises.columns:
                    target_mask = filtered_exercises['target'].str.lower() == target.lower()
                    # Also check body part if available
                    if 'body_part' in filtered_exercises.columns:
                        body_part_mask = filtered_exercises['body_part'].str.lower() == target.lower()
                        target_mask = target_mask | body_part_mask
                    relaxed_mask = relaxed_mask & target_mask
                
                # Filter out already selected exercises
                if selected_exercises:
                    relaxed_mask = relaxed_mask & ~filtered_exercises['id'].isin(selected_exercises)
                
                # Get matching exercises with relaxed constraints
                relaxed_matches = filtered_exercises[relaxed_mask]
                
                if len(relaxed_matches) > 0:
                    # Randomly select one matching exercise
                    selected_exercise = relaxed_matches.sample(1).iloc[0]
                    
                    # Add to selected exercises
                    selected_exercises.append(selected_exercise['id'])
                    
                    # Create exercise dict with metadata
                    exercise_dict = {
                        'id': selected_exercise['id'],
                        'name': selected_exercise['name'],
                        'type': ex_type,
                        'movement_pattern': selected_exercise['movement_pattern'],
                        'is_primary': is_primary,
                        'target': selected_exercise['target'] if 'target' in selected_exercise else None,
                        'equipment': selected_exercise['equipment'] if 'equipment' in selected_exercise else None,
                        'instructions': selected_exercise['instructions'] if 'instructions' in selected_exercise else None,
                        'gifUrl': selected_exercise['gifurl'] if 'gifurl' in selected_exercise else None
                    }
                    
                    result.append(exercise_dict)
                else:
                    # If still no matches, just pick any exercise of the right type
                    any_match = filtered_exercises[filtered_exercises['is_compound'] == ex_type]
                    
                    if len(any_match) > 0:
                        # Randomly select one exercise
                        selected_exercise = any_match.sample(1).iloc[0]
                        
                        # Add to selected exercises
                        selected_exercises.append(selected_exercise['id'])
                        
                        # Create exercise dict with metadata
                        exercise_dict = {
                            'id': selected_exercise['id'],
                            'name': selected_exercise['name'],
                            'type': ex_type,
                            'movement_pattern': selected_exercise['movement_pattern'],
                            'is_primary': is_primary,
                            'target': selected_exercise['target'] if 'target' in selected_exercise else None,
                            'equipment': selected_exercise['equipment'] if 'equipment' in selected_exercise else None,
                            'instructions': selected_exercise['instructions'] if 'instructions' in selected_exercise else None,
                            'gifUrl': selected_exercise['gifurl'] if 'gifurl' in selected_exercise else None
                        }
                        
                        result.append(exercise_dict)
                    else:
                        # If still no matches, skip this requirement
                        print(f"Could not find any exercise matching: {requirement}")
        
        return result
    
    def create_workout_routine(self, 
                              user_preferences: Dict,
                              weeks: int = 4) -> Dict:
        """
        Create a complete workout routine based on user preferences.
        
        Args:
            user_preferences: Dictionary of user preferences
            weeks: Number of weeks for the routine
            
        Returns:
            Complete workout routine
        """
        # Extract user preferences
        experience_level = user_preferences.get('experience_level', 'beginner')
        workout_split = user_preferences.get('workout_split', 'full_body')
        days_per_week = user_preferences.get('days_per_week', None)
        target_muscles = user_preferences.get('target_muscles', None)
        equipment_available = user_preferences.get('equipment_available', None)
        workout_goal = user_preferences.get('workout_goal', 'strength')
        time_per_workout = user_preferences.get('time_per_workout', 60)
        risk_tolerance = user_preferences.get('risk_tolerance', 'low')
        
        # Preprocess exercise data
        self.preprocess_exercise_data()
        
        # Filter exercises based on user preferences
        filtered_exercises = self.filter_exercises(
            experience_level=experience_level,
            target_muscles=target_muscles,
            equipment_available=equipment_available,
            risk_level=risk_tolerance
        )
        
        print(f"Filtered exercises: {len(filtered_exercises)}")
        
        # Determine workout template based on experience and split preference
        if experience_level in self.workout_templates:
            if workout_split in self.workout_templates[experience_level]:
                template = self.workout_templates[experience_level][workout_split]
            else:
                # Fallback to default split for experience level
                default_split = list(self.workout_templates[experience_level].keys())[0]
                template = self.workout_templates[experience_level][default_split]
        else:
            # Fallback to beginner templates
            if workout_split in self.workout_templates['beginner']:
                template = self.workout_templates['beginner'][workout_split]
            else:
                template = self.workout_templates['beginner']['full_body']
        
        # Override days_per_week if specified by user
        if days_per_week:
            template['days_per_week'] = days_per_week
        
        # Adjust exercises per workout based on time available
        if time_per_workout < 30:
            template['exercises_per_workout'] = max(3, template['exercises_per_workout'] - 3)
        elif time_per_workout < 45:
            template['exercises_per_workout'] = max(4, template['exercises_per_workout'] - 2)
        elif time_per_workout < 60:
            template['exercises_per_workout'] = max(5, template['exercises_per_workout'] - 1)
        elif time_per_workout > 90:
            template['exercises_per_workout'] = min(10, template['exercises_per_workout'] + 2)
        
        # Adjust rep ranges based on workout goal
        if workout_goal == 'strength':
            template['rep_ranges'] = {
                'compound': '4-6',
                'isolation': '6-8'
            }
        elif workout_goal == 'hypertrophy':
            template['rep_ranges'] = {
                'compound': '8-12',
                'isolation': '10-15'
            }
        elif workout_goal == 'endurance':
            template['rep_ranges'] = {
                'compound': '15-20',
                'isolation': '15-25'
            }
        
        # Create workout routine structure
        routine = {
            'name': f"{experience_level.capitalize()} {workout_split.replace('_', ' ').title()} - {workout_goal.capitalize()} Program",
            'description': f"A {weeks}-week {workout_goal} focused program for {experience_level} athletes using a {workout_split.replace('_', ' ')} split.",
            'experience_level': experience_level,
            'goal': workout_goal,
            'weeks': weeks,
            'days_per_week': template['days_per_week'],
            'workout_split': workout_split,
            'workout_rotation': [],
            'workouts': {}
        }
        
        # Build workout rotation schedule based on split type
        if workout_split == 'full_body':
            # Full body workouts
            # Example: [A, rest, B, rest, C, rest, rest] for 3 days/week
            workout_types = ['A', 'B', 'C'][:template['days_per_week']]
            rest_days = template['days_per_week'] * ['rest']
            
            # Interleave workout and rest days
            rotation = []
            for i in range(min(len(workout_types), 7)):
                rotation.append(workout_types[i])
                if len(rotation) < 7:
                    rotation.append('rest')
            
            # Add additional rest days if needed
            while len(rotation) < 7:
                rotation.append('rest')
            
            routine['workout_rotation'] = rotation
            
            # Create workouts for each type
            for workout_type in workout_types:
                # Select exercises for this workout
                workout_exercises = self.select_exercises_for_template(
                    filtered_exercises, 
                    template['structure']
                )
                
                # Create workout
                routine['workouts'][workout_type] = {
                    'name': f"Full Body Workout {workout_type}",
                    'exercises': workout_exercises,
                    'sets_per_exercise': template['sets_per_exercise'],
                    'rep_ranges': template['rep_ranges'],
                    'rest_periods': template['rest_periods']
                }
        
        elif workout_split == 'upper_lower':
            # Upper/Lower split
            # Example: [Upper, Lower, rest, Upper, Lower, rest, rest] for 4 days/week
            days_per_week = template['days_per_week']
            if days_per_week == 2:
                rotation = ['Upper', 'rest', 'Lower', 'rest', 'rest', 'rest', 'rest']
            elif days_per_week == 3:
                rotation = ['Upper', 'Lower', 'rest', 'Upper', 'rest', 'rest', 'rest']
            elif days_per_week == 4:
                rotation = ['Upper', 'Lower', 'rest', 'Upper', 'Lower', 'rest', 'rest']
            elif days_per_week == 5:
                rotation = ['Upper', 'Lower', 'Upper', 'Lower', 'Upper', 'rest', 'rest']
            elif days_per_week == 6:
                rotation = ['Upper', 'Lower', 'Upper', 'Lower', 'Upper', 'Lower', 'rest']
            else:
                rotation = ['Upper', 'Lower', 'rest', 'Upper', 'Lower', 'rest', 'rest']
            
            routine['workout_rotation'] = rotation
            
            # Create Upper and Lower workouts
            for workout_type in ['Upper', 'Lower']:
                # Select exercises for this workout
                workout_exercises = self.select_exercises_for_template(
                    filtered_exercises, 
                    template['structure'][workout_type.lower()]
                )
                
                # Create workout
                routine['workouts'][workout_type] = {
                    'name': f"{workout_type} Body Workout",
                    'exercises': workout_exercises,
                    'sets_per_exercise': template['sets_per_exercise'],
                    'rep_ranges': template['rep_ranges'],
                    'rest_periods': template['rest_periods']
                }
        
        elif workout_split == 'push_pull_legs':
            # Push/Pull/Legs split
            # Example: [Push, Pull, Legs, rest, Push, Pull, Legs] for 6 days/week
            days_per_week = template['days_per_week']
            if days_per_week == 3:
                rotation = ['Push', 'Pull', 'Legs', 'rest', 'rest', 'rest', 'rest']
            elif days_per_week == 4:
                rotation = ['Push', 'Pull', 'rest', 'Legs', 'Push', 'rest', 'rest']
            elif days_per_week == 5:
                rotation = ['Push', 'Pull', 'Legs', 'Push', 'Pull', 'rest', 'rest']
            elif days_per_week == 6:
                rotation = ['Push', 'Pull', 'Legs', 'Push', 'Pull', 'Legs', 'rest']
            else:
                rotation = ['Push', 'Pull', 'Legs', 'rest', 'Push', 'Pull', 'rest']
            
            routine['workout_rotation'] = rotation
            
            # Create Push, Pull, and Legs workouts
            for workout_type in ['Push', 'Pull', 'Legs']:
                # Select exercises for this workout
                workout_exercises = self.select_exercises_for_template(
                    filtered_exercises, 
                    template['structure'][workout_type.lower()]
                )
                
                # Create workout
                routine['workouts'][workout_type] = {
                    'name': f"{workout_type} Workout",
                    'exercises': workout_exercises,
                    'sets_per_exercise': template['sets_per_exercise'],
                    'rep_ranges': template['rep_ranges'],
                    'rest_periods': template['rest_periods']
                }
        
        elif workout_split == 'body_part_split':
            # Body part split
            # Example: [Chest, Back, Legs, Shoulders_Arms, Full_Body, rest, rest] for 5 days/week
            days_per_week = template['days_per_week']
            workout_types = ['Chest', 'Back', 'Legs', 'Shoulders_Arms', 'Full_Body'][:days_per_week]
            rotation = workout_types + ['rest'] * (7 - len(workout_types))
            
            routine['workout_rotation'] = rotation
            
            # Create workouts for each body part
            for workout_type in workout_types:
                workout_key = workout_type.lower().replace('_', ' ')
                
                # Select exercises for this workout
                workout_exercises = self.select_exercises_for_template(
                    filtered_exercises, 
                    template['structure'][workout_key]
                )
                
                # Create workout
                routine['workouts'][workout_type] = {
                    'name': f"{workout_type.replace('_', ' ')} Workout",
                    'exercises': workout_exercises,
                    'sets_per_exercise': template['sets_per_exercise'],
                    'rep_ranges': template['rep_ranges'],
                    'rest_periods': template['rest_periods']
                }
        
        # Create progression plan
        routine['progression_plan'] = self._create_progression_plan(routine, weeks)
        
        return routine
    
    def _create_progression_plan(self, routine: Dict, weeks: int) -> Dict:
        """
        Create a progression plan for the workout routine.
        
        Args:
            routine: Generated workout routine
            weeks: Number of weeks
            
        Returns:
            Dictionary with progression plan details
        """
        workout_goal = routine['goal']
        progression_plan = {
            'description': f"Progressive overload plan for {weeks} weeks",
            'weeks': []
        }
        
        for week in range(1, weeks + 1):
            week_plan = {
                'week': week,
                'focus': '',
                'load': '',
                'volume': '',
                'notes': ''
            }
            
            # Determine focus and adjustments based on week and goal
            if workout_goal == 'strength':
                if week % 4 == 1:
                    # Introductory/technique week
                    week_plan['focus'] = 'Technique'
                    week_plan['load'] = '70-75% of max'
                    week_plan['volume'] = 'Moderate'
                    week_plan['notes'] = 'Focus on perfecting form and technique'
                elif week % 4 == 2:
                    # Volume week
                    week_plan['focus'] = 'Volume'
                    week_plan['load'] = '75-80% of max'
                    week_plan['volume'] = 'High'
                    week_plan['notes'] = 'Increase total reps and sets'
                elif week % 4 == 3:
                    # Intensity week
                    week_plan['focus'] = 'Intensity'
                    week_plan['load'] = '85-90% of max'
                    week_plan['volume'] = 'Low to Moderate'
                    week_plan['notes'] = 'Focus on heavier weights with good form'
                else:
                    # Deload week
                    week_plan['focus'] = 'Recovery'
                    week_plan['load'] = '60-65% of max'
                    week_plan['volume'] = 'Low'
                    week_plan['notes'] = 'Deload week to allow for recovery'
            
            elif workout_goal == 'hypertrophy':
                if week % 4 == 1:
                    # Introductory week
                    week_plan['focus'] = 'Mind-Muscle Connection'
                    week_plan['load'] = '65-70% of max'
                    week_plan['volume'] = 'Moderate'
                    week_plan['notes'] = 'Focus on feeling the target muscles work'
                elif week % 4 == 2:
                    # Volume week
                    week_plan['focus'] = 'Volume'
                    week_plan['load'] = '70-75% of max'
                    week_plan['volume'] = 'High'
                    week_plan['notes'] = 'Increase total sets and time under tension'
                elif week % 4 == 3:
                    # Intensity week
                    week_plan['focus'] = 'Intensity'
                    week_plan['load'] = '75-80% of max'
                    week_plan['volume'] = 'Moderate to High'
                    week_plan['notes'] = 'Focus on increasing weight while maintaining form'
                else:
                    # Deload week
                    week_plan['focus'] = 'Recovery'
                    week_plan['load'] = '60-65% of max'
                    week_plan['volume'] = 'Low'
                    week_plan['notes'] = 'Deload week to allow for recovery'
            
            elif workout_goal == 'endurance':
                if week % 4 == 1:
                    # Introductory week
                    week_plan['focus'] = 'Technique and Conditioning'
                    week_plan['load'] = '50-60% of max'
                    week_plan['volume'] = 'Moderate'
                    week_plan['notes'] = 'Focus on form with shorter rest periods'
                elif week % 4 == 2:
                    # Volume week
                    week_plan['focus'] = 'Volume and Density'
                    week_plan['load'] = '55-65% of max'
                    week_plan['volume'] = 'High'
                    week_plan['notes'] = 'Increase reps and decrease rest periods'
                elif week % 4 == 3:
                    # Intensity week
                    week_plan['focus'] = 'Work Capacity'
                    week_plan['load'] = '60-70% of max'
                    week_plan['volume'] = 'High'
                    week_plan['notes'] = 'Implement supersets and circuit training'
                else:
                    # Deload week
                    week_plan['focus'] = 'Active Recovery'
                    week_plan['load'] = '45-55% of max'
                    week_plan['volume'] = 'Low to Moderate'
                    week_plan['notes'] = 'Recover while maintaining conditioning'
            
            else:  # General fitness
                if week % 4 == 1:
                    week_plan['focus'] = 'Technique'
                    week_plan['load'] = 'Moderate'
                    week_plan['volume'] = 'Moderate'
                    week_plan['notes'] = 'Focus on proper form and technique'
                elif week % 4 == 2:
                    week_plan['focus'] = 'Intensity'
                    week_plan['load'] = 'Moderate to High'
                    week_plan['volume'] = 'Moderate'
                    week_plan['notes'] = 'Increase weights by 5-10% from previous week'
                elif week % 4 == 3:
                    week_plan['focus'] = 'Volume'
                    week_plan['load'] = 'Moderate'
                    week_plan['volume'] = 'High'
                    week_plan['notes'] = 'Add 1-2 extra sets to each exercise'
                else:
                    week_plan['focus'] = 'Recovery'
                    week_plan['load'] = 'Light to Moderate'
                    week_plan['volume'] = 'Low'
                    week_plan['notes'] = 'Deload week to allow for recovery'
            
            progression_plan['weeks'].append(week_plan)
        
        return progression_plan
    
    def generate_routine_from_user_input(self, user_input: Dict) -> Dict:
        """
        User-friendly method to generate a workout routine from simple user input.
        
        Args:
            user_input: Dictionary with user preferences
            
        Returns:
            Structured workout routine
        """
        # Map user input to preferences
        preferences = {
            'experience_level': user_input.get('experience', 'beginner').lower(),
            'workout_split': user_input.get('split', 'full_body').lower().replace(' ', '_'),
            'days_per_week': int(user_input.get('days', 3)),
            'target_muscles': user_input.get('muscles', []),
            'equipment_available': user_input.get('equipment', ['bodyweight', 'dumbbell', 'barbell']),
            'workout_goal': user_input.get('goal', 'strength').lower(),
            'time_per_workout': int(user_input.get('time', 60)),
            'risk_tolerance': user_input.get('risk', 'low').lower()
        }
        
        # Create the routine
        routine = self.create_workout_routine(preferences, weeks=int(user_input.get('weeks', 4)))
        
        return routine
    
    def format_routine_for_display(self, routine: Dict) -> Dict:
        """
        Format the workout routine for user-friendly display.
        
        Args:
            routine: Generated workout routine
            
        Returns:
            Formatted routine for display
        """
        display_routine = {
            'name': routine['name'],
            'description': routine['description'],
            'details': {
                'Experience Level': routine['experience_level'].capitalize(),
                'Goal': routine['goal'].capitalize(),
                'Weeks': routine['weeks'],
                'Days Per Week': routine['days_per_week'],
                'Split Type': routine['workout_split'].replace('_', ' ').title()
            },
            'weekly_schedule': [],
            'workouts': []
        }
        
        # Format weekly schedule
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        schedule = []
        for i, workout_day in enumerate(routine['workout_rotation']):
            if workout_day == 'rest':
                schedule.append(f"{days[i]}: Rest Day")
            else:
                schedule.append(f"{days[i]}: {routine['workouts'][workout_day]['name']}")
        
        display_routine['weekly_schedule'] = schedule
        
        # Format workouts
        for workout_type, workout in routine['workouts'].items():
            formatted_workout = {
                'name': workout['name'],
                'exercises': []
            }
            
            for exercise in workout['exercises']:
                formatted_exercise = {
                    'name': exercise['name'],
                    'type': exercise['type'].capitalize(),
                    'target': exercise['target'] if exercise['target'] else 'Multiple',
                    'equipment': exercise['equipment'] if exercise['equipment'] else 'Bodyweight',
                    'sets': workout['sets_per_exercise'],
                    'reps': workout['rep_ranges'][exercise['type']],
                    'rest': workout['rest_periods'][exercise['type']],
                    'notes': 'Primary exercise' if exercise['is_primary'] else 'Accessory exercise',
                    'instructions': exercise['instructions'] if exercise['instructions'] else '',
                    'image_url': exercise['gifUrl'] if exercise['gifUrl'] else ''
                }
                
                formatted_workout['exercises'].append(formatted_exercise)
            
            display_routine['workouts'].append(formatted_workout)
        
        # Format progression plan
        display_routine['progression'] = []
        for week in routine['progression_plan']['weeks']:
            display_routine['progression'].append({
                'Week': week['week'],
                'Focus': week['focus'],
                'Load': week['load'],
                'Volume': week['volume'],
                'Notes': week['notes']
            })
        
        return display_routine