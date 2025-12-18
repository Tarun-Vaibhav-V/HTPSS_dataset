"""
AI Symptom Analysis Assistant - Interactive Application
========================================================
Interactive command-line interface for the AI Symptom Analysis Assistant.

Features:
- Interactive conversation mode
- Batch symptom analysis
- Detailed medical explanations
- Risk level assessment

Author: AI Assistant
Date: 2025-12-18
"""

import sys
import os
from symptom_analyzer import SymptomAnalyzer


class InteractiveApp:
    """
    Interactive command-line application for symptom analysis.
    """
    
    def __init__(self):
        """Initialize the interactive application."""
        self.analyzer = None
        self.running = False
    
    def display_banner(self):
        """Display welcome banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        🏥 AI SYMPTOM ANALYSIS ASSISTANT (DSARG_4)                   ║
║                                                                      ║
║        An intelligent system for analyzing symptoms and              ║
║        suggesting possible medical conditions                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

⚠️  MEDICAL DISCLAIMER:
This tool is for educational and informational purposes only.
It is NOT a substitute for professional medical advice, diagnosis,
or treatment. Always seek the advice of qualified healthcare providers.

"""
        print(banner)
    
    def display_help(self):
        """Display help information."""
        help_text = """
📚 AVAILABLE COMMANDS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  analyze <symptoms>   - Analyze symptoms and get disease predictions
  
  help                 - Show this help message
  
  examples             - Show example usage
  
  clear                - Clear the screen
  
  quit / exit          - Exit the application

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 TIP: You can enter symptoms separated by commas, e.g.:
   "fever, headache, cough, fatigue"

"""
        print(help_text)
    
    def display_examples(self):
        """Display usage examples."""
        examples = """
📝 USAGE EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1: Common Cold Symptoms
  Input: fever, cough, runny nose, headache
  
Example 2: Dengue-like Symptoms
  Input: high fever, joint pain, skin rash, headache, back pain
  
Example 3: Diabetes Symptoms
  Input: increased thirst, frequent urination, fatigue, blurred vision
  
Example 4: Natural Language
  Input: I have been experiencing severe headache and vomiting
  
Example 5: Malaria Symptoms
  Input: chills, high fever, sweating, headache, muscle pain

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        print(examples)
    
    def initialize_analyzer(self):
        """Initialize the symptom analyzer."""
        try:
            print("🔧 Initializing AI Symptom Analyzer...")
            self.analyzer = SymptomAnalyzer()
            print("✅ Analyzer initialized successfully!\n")
            return True
        except FileNotFoundError:
            print("\n❌ ERROR: Model files not found!")
            print("\nPlease train the model first by running:")
            print("   python train.py")
            print("\nThis will:")
            print("   1. Preprocess the dataset")
            print("   2. Train the ML model")
            print("   3. Save model files to 'models/' directory")
            return False
        except Exception as e:
            print(f"\n❌ ERROR: Failed to initialize analyzer")
            print(f"   Details: {e}")
            return False
    
    def process_command(self, command):
        """
        Process user command.
        
        Args:
            command (str): User command
            
        Returns:
            bool: True to continue, False to exit
        """
        command = command.strip().lower()
        
        # Exit commands
        if command in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the AI Symptom Analysis Assistant!")
            print("   Stay healthy! 🌟\n")
            return False
        
        # Help command
        elif command == 'help' or command == '?':
            self.display_help()
        
        # Examples command
        elif command == 'examples':
            self.display_examples()
        
        # Clear screen command
        elif command == 'clear' or command == 'cls':
            os.system('cls' if os.name == 'nt' else 'clear')
            self.display_banner()
        
        # Empty command
        elif not command:
            pass
        
        # Analyze symptoms
        else:
            # Remove "analyze" prefix if present
            if command.startswith('analyze '):
                symptoms_input = command[8:]
            else:
                symptoms_input = command
            
            if symptoms_input:
                self.analyze_symptoms(symptoms_input)
            else:
                print("⚠️ Please provide symptoms to analyze.")
                print("   Example: fever, headache, cough")
        
        return True
    
    def analyze_symptoms(self, symptoms_input):
        """
        Analyze user-provided symptoms.
        
        Args:
            symptoms_input (str): User symptom input
        """
        print("\n" + "=" * 70)
        print("🔍 ANALYZING SYMPTOMS...")
        print("=" * 70)
        
        try:
            # Run analysis
            result = self.analyzer.analyze(symptoms_input, detailed=True)
            print(result)
            
            # Prompt for next action
            print("\n" + "=" * 70)
            print("💡 What would you like to do next?")
            print("   - Enter new symptoms to analyze")
            print("   - Type 'help' for available commands")
            print("   - Type 'exit' to quit")
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR during analysis: {e}")
            print("   Please try again with different symptoms.\n")
    
    def run_interactive_mode(self):
        """Run the interactive conversation mode."""
        self.running = True
        
        print("✅ Interactive mode started!")
        print("   Type 'help' for available commands\n")
        
        while self.running:
            try:
                # Get user input
                user_input = input("🩺 You: ").strip()
                
                # Process command
                continue_running = self.process_command(user_input)
                
                if not continue_running:
                    self.running = False
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                confirm = input("   Do you want to exit? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    print("\n👋 Goodbye!")
                    self.running = False
                else:
                    print("   Continuing...\n")
            
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("   Continuing...\n")
    
    def run_quick_analysis(self, symptoms_input):
        """
        Run a quick analysis without interactive mode.
        
        Args:
            symptoms_input (str): Symptoms to analyze
        """
        self.analyze_symptoms(symptoms_input)
    
    def start(self, quick_mode=None):
        """
        Start the application.
        
        Args:
            quick_mode (str): If provided, run quick analysis instead of interactive
        """
        # Display banner
        self.display_banner()
        
        # Initialize analyzer
        if not self.initialize_analyzer():
            return
        
        # Run in appropriate mode
        if quick_mode:
            # Quick analysis mode
            self.run_quick_analysis(quick_mode)
        else:
            # Interactive mode
            self.run_interactive_mode()


def main():
    """Main entry point."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='AI Symptom Analysis Assistant - Interactive Application'
    )
    
    parser.add_argument(
        '--symptoms',
        type=str,
        help='Symptoms to analyze (for quick mode)',
        default=None
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test cases instead of interactive mode'
    )
    
    args = parser.parse_args()
    
    # Create and start app
    app = InteractiveApp()
    
    if args.test:
        # Test mode - run predefined test cases
        print("🧪 Running test cases...\n")
        
        app.display_banner()
        
        if not app.initialize_analyzer():
            return
        
        test_cases = [
            ("Dengue", "high fever, joint pain, skin rash, headache, back pain, nausea"),
            ("Diabetes", "fatigue, weight loss, excessive hunger, increased appetite, frequent urination"),
            ("Common Cold", "fever, cough, runny nose, headache, congestion"),
            ("Malaria", "chills, high fever, sweating, headache, muscle pain"),
        ]
        
        for name, symptoms in test_cases:
            print(f"\n{'=' * 70}")
            print(f"🧪 TEST CASE: {name}")
            print(f"{'=' * 70}\n")
            app.run_quick_analysis(symptoms)
            print("\n")
        
        print("✅ All test cases completed!")
        
    elif args.symptoms:
        # Quick mode - analyze provided symptoms
        app.start(quick_mode=args.symptoms)
    else:
        # Interactive mode
        app.start()


if __name__ == "__main__":
    main()
