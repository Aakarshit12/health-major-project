import google.generativeai as genai
from PIL import Image
import os
import sys

class DiseaseAnalyzerChatbot:
    def __init__(self, api_key):
        """Initialize the chatbot with Gemini API"""
        try:
            # Clean the API key (remove any whitespace)
            api_key = str(api_key).strip()
            
            # Configure Gemini with the API key
            genai.configure(api_key=api_key)
            
            # Use Gemini 2.5 Flash - supports both text and vision
            model_name = 'models/gemini-2.5-flash'
            
            # Initialize the model for both chat and vision
            self.chat_model = genai.GenerativeModel(model_name)
            self.vision_model = genai.GenerativeModel(model_name)
            
            print(f"✅ Model initialized: {model_name}")
        except Exception as e:
            raise Exception(f"Failed to configure Gemini API: {str(e)}")
        
        # System prompt to restrict responses to disease-related topics only
        self.system_prompt = """You are a medical assistant specialized in diseases and health conditions. 
        You ONLY answer questions related to:
        - Diseases, symptoms, and medical conditions
        - Health-related queries
        - Medical terminology and explanations
        - Treatment information (general, not prescriptive)
        - Prevention and health tips
        
        If a user asks about anything NOT related to diseases or health, politely redirect them by saying:
        "I'm sorry, I can only assist with disease and health-related questions. Please ask me something about medical conditions, symptoms, or health concerns."
        
        Always provide helpful, accurate medical information but remind users to consult healthcare professionals for diagnosis and treatment.
        Keep responses clear, concise, and educational."""
        
        self.chat_history = []
        
    def chat(self, user_message):
        """Handle text-based chat about diseases"""
        try:
            # Combine system prompt with user message
            full_prompt = f"{self.system_prompt}\n\nUser: {user_message}\nAssistant:"
            
            # Generate response
            response = self.chat_model.generate_content(full_prompt)
            
            # Store in chat history
            self.chat_history.append({
                "user": user_message,
                "assistant": response.text
            })
            
            return response.text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_xray(self, image_path):
        """Analyze X-ray image for disease detection"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return "Error: Image file not found!"
            
            # Open and load the image
            img = Image.open(image_path)
            
            # Create a detailed prompt for medical image analysis
            analysis_prompt = """You are an expert medical imaging AI assistant. Analyze this X-ray image carefully and provide:

1. Image Quality Assessment: Comment on the quality and clarity of the X-ray
2. Anatomical Region: Identify what body part is shown
3. Findings: Describe any abnormalities, suspicious areas, or concerning patterns you observe
4. Potential Conditions: List possible diseases or conditions that might be indicated such as pneumonia, tuberculosis, fractures, tumors, etc
5. Confidence Level: Rate your confidence in the findings as Low, Medium, or High
6. Recommendation: Suggest whether immediate medical attention is needed

IMPORTANT DISCLAIMER: Remind the user that this is an AI analysis and NOT a replacement for professional medical diagnosis. They should consult a qualified radiologist or physician for accurate diagnosis.

Format your response clearly with these sections."""

            # Generate analysis using vision model
            response = self.vision_model.generate_content([analysis_prompt, img])
            
            return response.text
        
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def explain_ml_results(self, ml_results_json):
        """Explain ML model results in conversational language"""
        try:
            # Create a prompt to explain the ML results
            explanation_prompt = f"""You are a compassionate medical AI assistant. A machine learning model has analyzed an X-ray image and produced the following results:

{ml_results_json}

Please provide a clear, empathetic explanation of these results for the patient. Include:

1. Main Finding Summary: Explain the top finding in simple terms
2. Risk Assessment: Explain what the probability percentages mean
3. Key Concerns: Highlight findings that need immediate attention (High Risk and above)
4. Recommendations: Summarize the recommended actions in priority order
5. Next Steps: What the patient should do next
6. Reassurance: Provide appropriate reassurance while being honest

Use simple, non-technical language that a patient can understand. Be empathetic and supportive. Always emphasize that these are AI-generated results and they should consult with a healthcare professional for proper diagnosis and treatment.

Format your response in a friendly, conversational tone."""

            # Generate explanation
            response = self.chat_model.generate_content(explanation_prompt)
            
            return response.text
        
        except Exception as e:
            return f"Error explaining ML results: {str(e)}"
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("Chat history cleared.")

def print_header():
    """Print chatbot header"""
    print("\n" + "="*60)
    print("🏥  DISEASE ANALYZER CHATBOT  🏥")
    print("    Powered by Google Gemini AI")
    print("="*60)
    print("\nFeatures:")
    print("  • Ask questions about diseases and symptoms")
    print("  • Analyze X-ray images for disease detection")
    print("  • Explain ML model results in simple terms")
    print("="*60 + "\n")

def print_menu():
    """Print menu options"""
    print("\n📋 OPTIONS:")
    print("  1. Chat about diseases")
    print("  2. Analyze X-ray image with AI")
    print("  3. Explain ML model results (JSON)")
    print("  4. Clear chat history")
    print("  5. Exit")
    print("-" * 40)

def main():
    # Get API key from environment variable or user input
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("🔑 Google Gemini API Key not found in environment variables.")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        api_key = input("Please enter your Gemini API key: ").strip()
        
        if not api_key:
            print("❌ API key is required to run the chatbot!")
            sys.exit(1)
    
    # Clean the API key
    api_key = api_key.strip().replace(" ", "").replace("\n", "").replace("\t", "")
    
    # Validate API key format
    if not api_key.startswith('AIzaSy'):
        print("⚠️  Warning: API key format looks unusual. It should start with 'AIzaSy'")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            sys.exit(1)
    
    # Initialize chatbot
    print("\n⏳ Initializing Disease Analyzer Chatbot...")
    try:
        chatbot = DiseaseAnalyzerChatbot(api_key)
        print("✅ Chatbot initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {str(e)}")
        print("\n💡 Tips:")
        print("  • Make sure your API key is correct")
        print("  • Check your internet connection")
        print("  • Verify the API key at: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    print_header()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            # Chat mode
            print("\n💬 CHAT MODE")
            print("Ask me anything about diseases and health!")
            print("Type 'back' to return to main menu\n")
            print("-" * 40)
            
            while True:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if not user_input:
                    print("⚠️  Please enter a message!")
                    continue
                
                print("\n🤖 Assistant: ", end="")
                response = chatbot.chat(user_input)
                print(response)
        
        elif choice == '2':
            # Image analysis mode
            print("\n🔬 X-RAY ANALYSIS MODE")
            print("Enter the path to your X-ray image")
            print("Type 'back' to return to main menu\n")
            print("-" * 40)
            
            image_path = input("\n📁 Image path: ").strip()
            
            if image_path.lower() == 'back':
                continue
            
            if not image_path:
                print("⚠️  Please enter an image path!")
                continue
            
            print("\n⏳ Analyzing X-ray image... Please wait...\n")
            result = chatbot.analyze_xray(image_path)
            print("="*60)
            print("📊 ANALYSIS RESULTS")
            print("="*60)
            print(result)
            print("="*60)
        
        elif choice == '3':
            # ML Results explanation mode
            print("\n🧠 ML MODEL RESULTS EXPLANATION")
            print("Paste your ML model JSON results or provide file path")
            print("Type 'back' to return to main menu\n")
            print("-" * 40)
            
            print("\nChoose input method:")
            print("  1. Paste JSON directly")
            print("  2. Provide JSON file path")
            method = input("\nEnter choice (1-2): ").strip()
            
            if method == '1':
                print("\n📋 Paste your JSON results below (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == '' and lines:
                        break
                    lines.append(line)
                
                ml_results = '\n'.join(lines)
                
                if ml_results.strip():
                    print("\n⏳ Analyzing ML results... Please wait...\n")
                    explanation = chatbot.explain_ml_results(ml_results)
                    print("="*60)
                    print("💡 ML RESULTS EXPLANATION")
                    print("="*60)
                    print(explanation)
                    print("="*60)
                else:
                    print("⚠️  No JSON data provided!")
            
            elif method == '2':
                json_path = input("\n📁 JSON file path: ").strip()
                
                if json_path.lower() == 'back':
                    continue
                
                try:
                    with open(json_path, 'r') as f:
                        ml_results = f.read()
                    
                    print("\n⏳ Analyzing ML results... Please wait...\n")
                    explanation = chatbot.explain_ml_results(ml_results)
                    print("="*60)
                    print("💡 ML RESULTS EXPLANATION")
                    print("="*60)
                    print(explanation)
                    print("="*60)
                except FileNotFoundError:
                    print("❌ Error: File not found!")
                except Exception as e:
                    print(f"❌ Error reading file: {str(e)}")
            else:
                print("❌ Invalid choice!")
        
        elif choice == '4':
            # Clear history
            chatbot.clear_history()
            print("✅ Chat history has been cleared!")
        
        elif choice == '5':
            # Exit
            print("\n👋 Thank you for using Disease Analyzer Chatbot!")
            print("Stay healthy! 🏥\n")
            sys.exit(0)
        
        else:
            print("❌ Invalid choice! Please enter a number between 1 and 5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        sys.exit(1)