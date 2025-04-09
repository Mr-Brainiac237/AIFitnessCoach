# src/routines/cli.py
import argparse
import json
import os
from src.routines.workout_generator import WorkoutRoutineGenerator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate personalized workout routines')
    
    # User preferences
    parser.add_argument('--experience', choices=['beginner', 'intermediate', 'advanced'], 
                        default='beginner', help='User experience level')
    parser.add_argument('--split', choices=['full_body', 'upper_lower', 'push_pull_legs', 'body_part_split'], 
                        default='full_body', help='Workout split type')
    parser.add_argument('--days', type=int, default=3, help='Number of workout days per week')
    parser.add_argument('--goal', choices=['strength', 'hypertrophy', 'endurance', 'general'], 
                        default='strength', help='Workout goal')
    parser.add_argument('--time', type=int, default=60, help='Time available per workout (minutes)')
    parser.add_argument('--weeks', type=int, default=4, help='Program duration in weeks')
    parser.add_argument('--risk', choices=['low', 'medium', 'high'], 
                        default='low', help='Risk tolerance level')
    
    # Target muscles (multiple selection)
    parser.add_argument('--muscles', nargs='+', 
                        help='Target muscles to focus on (e.g., chest back legs)')
    
    # Equipment (multiple selection)
    parser.add_argument('--equipment', nargs='+', 
                        default=['bodyweight', 'dumbbell', 'barbell'],
                        help='Available equipment (e.g., bodyweight dumbbell barbell)')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file path to save the routine (JSON)')
    parser.add_argument('--pretty', action='store_true', help='Pretty print the routine (text format)')
    
    return parser.parse_args()

def pretty_print_routine(routine):
    """Format and print the routine in a user-friendly way"""
    print("\n" + "="*80)
    print(f" {routine['name']} ".center(80, "="))
    print("="*80)
    
    print(f"\n{routine['description']}\n")
    
    print("PROGRAM DETAILS:")
    print(f"Experience Level: {routine['details']['Experience Level']}")
    print(f"Goal: {routine['details']['Goal']}")
    print(f"Duration: {routine['details']['Weeks']} weeks")
    print(f"Days Per Week: {routine['details']['Days Per Week']}")
    print(f"Split Type: {routine['details']['Split Type']}")
    
    print("\nWEEKLY SCHEDULE:")
    for day in routine['weekly_schedule']:
        print(f"  {day}")
    
    print("\nWORKOUTS:")
    for i, workout in enumerate(routine['workouts']):
        print(f"\n{i+1}. {workout['name']}:")
        print("-" * 80)
        
        for j, exercise in enumerate(workout['exercises']):
            print(f"  {j+1}. {exercise['name']}")
            print(f"     {exercise['sets']} sets × {exercise['reps']} reps | Rest: {exercise['rest']}")
            print(f"     Target: {exercise['target']} | Equipment: {exercise['equipment']}")
            print(f"     Notes: {exercise['notes']}")
            if exercise['instructions']:
                print(f"     Instructions: {exercise['instructions'][:100]}...")
            print()
    
    print("\nPROGRESSION PLAN:")
    print("-" * 80)
    for week in routine['progression']:
        print(f"Week {week['Week']}: {week['Focus']}")
        print(f"  Load: {week['Load']} | Volume: {week['Volume']}")
        print(f"  Notes: {week['Notes']}")
        print()
    
    print("="*80)
    print(" End of Program ".center(80, "="))
    print("="*80 + "\n")

def main():
    """Main function to run the CLI workout generator"""
    args = parse_args()
    
    # Create user preferences dict from args
    user_input = {
        'experience': args.experience,
        'split': args.split,
        'days': args.days,
        'muscles': args.muscles,
        'equipment': args.equipment,
        'goal': args.goal,
        'time': args.time,
        'weeks': args.weeks,
        'risk': args.risk
    }
    
    # Initialize the workout generator
    generator = WorkoutRoutineGenerator()
    
    # Generate the routine
    routine = generator.generate_routine_from_user_input(user_input)
    
    # Format for display
    display_routine = generator.format_routine_for_display(routine)
    
    # Output options
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(display_routine, f, indent=2)
        print(f"Workout routine saved to {args.output}")
    
    if args.pretty or not args.output:
        pretty_print_routine(display_routine)
    
    print("Workout routine generated successfully!")

if __name__ == "__main__":
    main()
