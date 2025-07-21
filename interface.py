#!/usr/bin/env python3
"""
College Recommendation System - Command Line Interface
Interactive CLI for getting personalized college recommendations
"""

import sys
import os
from typing import Optional, Dict, Any
import pandas as pd

# Add the current directory to path to import recommender
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from recommender import recommend_colleges
except ImportError as e:
    print(f"❌ Error importing recommender module: {e}")
    print("Please ensure recommender.py is in the same directory.")
    sys.exit(1)

class CollegeRecommendationCLI:
    """Command line interface for college recommendations."""
    
    def __init__(self):
        self.banner = """
╔══════════════════════════════════════════════════════╗
║        🎓 College Recommendation System 🎓           ║
║                                                      ║
║     Find the perfect college match for you!          ║
╚══════════════════════════════════════════════════════╝
"""
        self.help_text = """
📖 How to use:
• Enter your preferences for each field (or press Enter to skip)
• The more specific you are, the better the recommendations
• You can leave fields blank if you have no preference
• Type 'help' anytime for assistance or 'quit' to exit
"""

    def display_banner(self) -> None:
        """Display welcome banner."""
        print(self.banner)
        print(self.help_text)

    def get_user_input(self, prompt: str, field_name: str, examples: Optional[str] = None) -> str:
        """Get user input with validation and help."""
        full_prompt = f"\n{prompt}"
        if examples:
            full_prompt += f"\n💡 Examples: {examples}"
        full_prompt += f"\n🔸 {field_name}: "
        
        while True:
            try:
                user_input = input(full_prompt).strip()
                
                # Handle special commands
                if user_input.lower() == 'quit':
                    print("\n👋 Thanks for using the College Recommendation System!")
                    sys.exit(0)
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.display_banner()
                    continue
                
                return user_input
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the College Recommendation System!")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Error reading input: {e}")
                print("Please try again...")

    def get_number_input(self, prompt: str, default: int = 5, min_val: int = 1, max_val: int = 20) -> int:
        """Get numeric input with validation."""
        full_prompt = f"\n{prompt} (default: {default}, range: {min_val}-{max_val}): "
        
        while True:
            try:
                user_input = input(full_prompt).strip()
                
                if not user_input:
                    return default
                
                if user_input.lower() in ['quit', 'help', 'clear']:
                    if user_input.lower() == 'quit':
                        sys.exit(0)
                    elif user_input.lower() == 'help':
                        self.show_help()
                        continue
                    elif user_input.lower() == 'clear':
                        os.system('cls' if os.name == 'nt' else 'clear')
                        self.display_banner()
                        continue
                
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"⚠️ Please enter a number between {min_val} and {max_val}")
                    
            except ValueError:
                print("⚠️ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the College Recommendation System!")
                sys.exit(0)

    def show_help(self) -> None:
        """Display help information."""
        help_info = """
📚 Field Descriptions:

🔹 Course: The academic program you're interested in
   Examples: Computer Science, BBA, Civil Engineering, Medicine

🔹 Location: City or region where you want to study
   Examples: Kathmandu, Pokhara, Chitwan, Lalitpur

🔹 University: Specific university preference
   Examples: Tribhuvan University, Kathmandu University, Pokhara University

🔹 Ownership: Type of institution
   Examples: Public Institutions, Private Institutions, Community Institutions, Constituent Institutions

🔹 Number of Recommendations: How many colleges to show (1-20)

💡 Tips:
• Be as specific as possible for better matches
• You can skip any field by pressing Enter
• The system will find the best matches based on your preferences
• Type 'quit' to exit or 'clear' to restart

Press Enter to continue...
"""
        print(help_info)
        input()

    def collect_user_preferences(self) -> Dict[str, Any]:
        """Collect all user preferences."""
        preferences = {}
        
        # Course preference
        preferences['user_course'] = self.get_user_input(
            "What course are you interested in studying?",
            "Course",
            "Computer Science, BBA, Civil Engineering, Medicine"
        )
        
        # Location preference  
        preferences['user_location'] = self.get_user_input(
            "Which city or location do you prefer?",
            "Location", 
            "Kathmandu, Pokhara, Chitwan, Lalitpur"
        )
        
        # University preference
        preferences['user_university'] = self.get_user_input(
            "Do you have a preferred university?",
            "University",
            "Tribhuvan University, Kathmandu University, Pokhara University"
        )
        
        # Ownership preference
        preferences['user_ownership'] = self.get_user_input(
            "What type of institution do you prefer?",
            "Ownership",
            "Public Institutions, Private Institutions, Community Institutions, Constituent Institutions"
        )
        
        # Number of recommendations
        preferences['top_n'] = self.get_number_input(
            "How many college recommendations would you like?"
        )
        
        return preferences

    def display_preferences_summary(self, preferences: Dict[str, Any]) -> bool:
        """Display user preferences and ask for confirmation."""
        print("\n" + "="*60)
        print("📋 YOUR PREFERENCES SUMMARY")
        print("="*60)
        
        print(f"🎯 Course: {preferences['user_course'] or 'Any'}")
        print(f"📍 Location: {preferences['user_location'] or 'Any'}")
        print(f"🏛️ University: {preferences['user_university'] or 'Any'}")
        print(f"🏢 Ownership: {preferences['user_ownership'] or 'Any'}")
        print(f"📊 Number of recommendations: {preferences['top_n']}")
        
        print("\n" + "="*60)
        
        while True:
            confirm = input("\n✅ Is this correct? (y/n/edit): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                return False
            elif confirm == 'edit':
                return False
            elif confirm == 'quit':
                sys.exit(0)
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'edit' to modify")

    def display_recommendations(self, recommendations: pd.DataFrame) -> None:
        """Display recommendations in a formatted way."""
        if recommendations.empty:
            print("\n❌ No recommendations found.")
            print("💡 Try adjusting your preferences:")
            print("   • Use more general terms")
            print("   • Leave some fields blank")
            print("   • Check spelling of course/location names")
            return
        
        print("\n" + "="*80)
        print("🎯 YOUR PERSONALIZED COLLEGE RECOMMENDATIONS")
        print("="*80)
        
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"\n🏆 #{idx} - {row.get('college', 'N/A')}")
            print(f"   📍 Location: {row.get('location', 'N/A')}")
            print(f"   🏛️ University: {row.get('university', 'N/A')}")
            print(f"   🎓 Course: {row.get('course_offered', 'N/A')[:100]}{'...' if len(str(row.get('course_offered', ''))) > 100 else ''}")
            print(f"   🏢 Ownership: {row.get('ownership_type', 'N/A')}")
            
            if 'score' in row:
                score = row['score']
                if isinstance(score, (int, float)):
                    print(f"   ⭐ Match Score: {score:.2f}")
            
            print("   " + "-"*60)

    def ask_for_another_search(self) -> bool:
        """Ask if user wants to perform another search."""
        while True:
            choice = input("\n🔄 Would you like to search again? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            elif choice == 'quit':
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")

    def run(self) -> None:
        """Run the main CLI interface."""
        try:
            # Clear screen and show banner
            os.system('cls' if os.name == 'nt' else 'clear')
            self.display_banner()
            
            while True:
                # Collect user preferences
                preferences = self.collect_user_preferences()
                
                # Show summary and confirm
                if not self.display_preferences_summary(preferences):
                    print("\n🔄 Let's try again...")
                    continue
                
                print("\n🔍 Searching for the best college matches...")
                print("⏳ Please wait...")
                
                try:
                    # Get recommendations
                    recommendations = recommend_colleges(**preferences)
                    
                    # Display results
                    self.display_recommendations(recommendations)
                    
                except Exception as e:
                    print(f"\n❌ Error getting recommendations: {e}")
                    print("Please check if all required files are present and try again.")
                
                # Ask for another search
                if not self.ask_for_another_search():
                    break
                    
                # Clear for next search
                os.system('cls' if os.name == 'nt' else 'clear')
                self.display_banner()
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the College Recommendation System!")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            print("Please contact support if this issue persists.")
        
        print("\n🎓 Happy learning! Good luck with your college search! 🎓")

def main():
    """Main entry point for the CLI application."""
    cli = CollegeRecommendationCLI()
    cli.run()

if __name__ == "__main__":
    main()