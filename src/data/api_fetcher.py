# src/data/api_fetcher.py
import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union

# Load environment variables
load_dotenv()

class ExerciseDBFetcher:
    """
    Class to fetch exercise data from ExerciseDB API
    """
    def __init__(self):
        self.api_key = os.getenv("EXERCISEDB_API_KEY")
        self.base_url = "https://exercisedb.p.rapidapi.com/exercises"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "exercisedb.p.rapidapi.com"
        }
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the API
        
        Args:
            endpoint: API endpoint to query
            params: Query parameters
            
        Returns:
            JSON response
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        # Handle rate limiting
        if response.status_code == 429:
            wait_time = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            return self._make_request(endpoint, params)
        
        response.raise_for_status()
        return response.json()
    
    def get_all_exercises(self) -> List[Dict]:
        """
        Fetch all exercises from the API with pagination
        
        Returns:
            List of exercise dictionaries
        """
        all_exercises = []
        limit = 100  # Maximum number of exercises per request
        offset = 0
        
        while True:
            params = {
                "limit": limit,
                "offset": offset
            }
            exercises = self._make_request("", params)
            
            if not exercises:  # No more exercises to fetch
                break
                
            all_exercises.extend(exercises)
            
            if len(exercises) < limit:  # We've reached the end
                break
                
            offset += limit
            time.sleep(1)  # Add a small delay to avoid rate limiting
            
        return all_exercises
    
    def get_exercise_by_id(self, exercise_id: str) -> Dict:
        """
        Fetch a specific exercise by ID
        
        Args:
            exercise_id: Exercise ID
            
        Returns:
            Exercise dictionary
        """
        return self._make_request(f"/exercise/{exercise_id}")
    
    def get_exercises_by_bodypart(self, bodypart: str) -> List[Dict]:
        """
        Fetch exercises by bodypart
        
        Args:
            bodypart: Body part to filter by
            
        Returns:
            List of exercise dictionaries
        """
        return self._make_request(f"/bodyPart/{bodypart}")
    
    def get_exercises_by_target(self, target: str) -> List[Dict]:
        """
        Fetch exercises by target muscle
        
        Args:
            target: Target muscle to filter by
            
        Returns:
            List of exercise dictionaries
        """
        return self._make_request(f"/target/{target}")
    
    def get_exercises_by_equipment(self, equipment: str) -> List[Dict]:
        """
        Fetch exercises by equipment
        
        Args:
            equipment: Equipment to filter by
            
        Returns:
            List of exercise dictionaries
        """
        return self._make_request(f"/equipment/{equipment}")
    
    def get_bodyparts_list(self) -> List[str]:
        """
        Get list of all bodyparts
        
        Returns:
            List of bodyparts
        """
        return self._make_request("/bodyPartList")
    
    def get_target_muscles_list(self) -> List[str]:
        """
        Get list of all target muscles
        
        Returns:
            List of target muscles
        """
        return self._make_request("/targetList")
    
    def get_equipment_list(self) -> List[str]:
        """
        Get list of all equipment
        
        Returns:
            List of equipment
        """
        return self._make_request("/equipmentList")
    
    def save_all_data(self, output_dir: str = "data/raw") -> None:
        """
        Fetch and save all exercise data to files
        
        Args:
            output_dir: Directory to save data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all exercises
        all_exercises = self.get_all_exercises()
        with open(f"{output_dir}/all_exercises.json", "w") as f:
            json.dump(all_exercises, f, indent=2)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(all_exercises)
        df.to_csv(f"{output_dir}/all_exercises.csv", index=False)
        
        # Save lists of bodyparts, targets, and equipment
        bodyparts = self.get_bodyparts_list()
        with open(f"{output_dir}/bodyparts.json", "w") as f:
            json.dump(bodyparts, f, indent=2)
            
        targets = self.get_target_muscles_list()
        with open(f"{output_dir}/target_muscles.json", "w") as f:
            json.dump(targets, f, indent=2)
            
        equipment = self.get_equipment_list()
        with open(f"{output_dir}/equipment.json", "w") as f:
            json.dump(equipment, f, indent=2)
        
        print(f"All data saved to {output_dir}")


class WgerFetcher:
    """
    Class to fetch exercise data from Wger API
    """
    def __init__(self):
        self.base_url = "https://wger.de/api/v2"
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Token {os.getenv('WGER_API_KEY', '')}"
        }
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the API
        
        Args:
            endpoint: API endpoint to query
            params: Query parameters
            
        Returns:
            JSON response
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_exercises(self, language: int = 2, limit: int = 100, offset: int = 0) -> Dict:
        """
        Fetch exercises from the API
        
        Args:
            language: Language ID (2 for English)
            limit: Number of results per page
            offset: Pagination offset
            
        Returns:
            Dictionary containing exercise data
        """
        params = {
            "language": language,
            "limit": limit,
            "offset": offset
        }
        return self._make_request("/exercise/", params)
    
    def get_all_exercises(self, language: int = 2) -> List[Dict]:
        """
        Fetch all exercises with pagination
        
        Args:
            language: Language ID (2 for English)
            
        Returns:
            List of exercise dictionaries
        """
        all_exercises = []
        offset = 0
        limit = 100
        
        while True:
            response = self.get_exercises(language, limit, offset)
            results = response.get("results", [])
            all_exercises.extend(results)
            
            if not response.get("next"):
                break
                
            offset += limit
            
        return all_exercises
    
    def get_exercise_images(self, exercise_id: int) -> Dict:
        """
        Fetch images for a specific exercise
        
        Args:
            exercise_id: Exercise ID
            
        Returns:
            Dictionary containing image data
        """
        return self._make_request(f"/exerciseimage/?exercise={exercise_id}")
    
    def get_muscles(self) -> Dict:
        """
        Fetch muscle data
        
        Returns:
            Dictionary containing muscle data
        """
        return self._make_request("/muscle/")
    
    def get_equipment(self) -> Dict:
        """
        Fetch equipment data
        
        Returns:
            Dictionary containing equipment data
        """
        return self._make_request("/equipment/")
    
    def save_all_data(self, output_dir: str = "data/raw/wger") -> None:
        """
        Fetch and save all exercise data to files
        
        Args:
            output_dir: Directory to save data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all exercises
        all_exercises = self.get_all_exercises()
        with open(f"{output_dir}/all_exercises.json", "w") as f:
            json.dump(all_exercises, f, indent=2)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(all_exercises)
        df.to_csv(f"{output_dir}/all_exercises.csv", index=False)
        
        # Save muscles and equipment data
        muscles = self.get_muscles()
        with open(f"{output_dir}/muscles.json", "w") as f:
            json.dump(muscles, f, indent=2)
            
        equipment = self.get_equipment()
        with open(f"{output_dir}/equipment.json", "w") as f:
            json.dump(equipment, f, indent=2)
        
        print(f"All Wger data saved to {output_dir}")


def fetch_and_merge_data():
    """
    Fetch data from both APIs and merge them
    """
    # Create output directories
    exercisedb_dir = "data/raw/exercisedb"
    wger_dir = "data/raw/wger"
    merged_dir = "data/processed"
    
    os.makedirs(exercisedb_dir, exist_ok=True)
    os.makedirs(wger_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    
    # Fetch data from ExerciseDB
    print("Fetching data from ExerciseDB...")
    exercisedb = ExerciseDBFetcher()
    exercisedb.save_all_data(exercisedb_dir)
    
    # Fetch data from Wger
    print("Fetching data from Wger...")
    wger = WgerFetcher()
    wger.save_all_data(wger_dir)
    
    # Load the data
    exercisedb_df = pd.read_csv(f"{exercisedb_dir}/all_exercises.csv")
    wger_df = pd.read_csv(f"{wger_dir}/all_exercises.csv")
    
    # TODO: Implement merging logic based on exercise names or other identifiers
    
    print("Data fetching complete. Merging will require additional processing.")
    
    return {
        "exercisedb": exercisedb_df,
        "wger": wger_df
    }


if __name__ == "__main__":
    fetch_and_merge_data()
