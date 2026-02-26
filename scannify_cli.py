"""
Scannify CLI - Simple Command Line PDF Scanner
Test the core AI functionality without web interface
"""

import os
import sys
from pathlib import Path

# Simple check for required packages
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    print("✅ AI packages loaded successfully")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Run: pip install langchain-google-genai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    sys.exit(1)

print(f"✅ API key loaded: {api_key[:10]}...")

# Initialize AI
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def test_ai_connection():
    """Test basic AI functionality"""
    print("\n🤖 Testing AI connection...")
    
    try:
        response = llm.invoke("Hello! Please respond with 'AI connection successful!'")
        print(f"🤖 AI Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ AI connection failed: {e}")
        return False

def analyze_text_content(text):
    """Analyze text content using AI"""
    prompt = f"""
    Analyze this document text and provide:
    
    1. SUMMARY: (2-3 sentences)
    2. MAIN TOPICS: (Key themes)
    3. DOCUMENT TYPE: (Report, Article, etc.)
    4. KEY INSIGHTS: (Important findings)
    5. IMPROVEMENT SUGGESTIONS: (How to make it better)
    
    Text: {text[:3000]}
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error analyzing content: {e}"

def enhance_content(text, enhancement_type="clarity"):
    """Enhance content using AI"""
    prompts = {
        "clarity": "Rewrite this text to be clearer and more readable while keeping all information:",
        "professional": "Make this text more professional and polished:",
        "summary": "Create a comprehensive summary with key points:",
        "structure": "Improve the structure and organization of this text:"
    }
    
    prompt = f"""
    {prompts.get(enhancement_type, prompts['clarity'])}
    
    Original text: {text[:3000]}
    
    Please provide the enhanced version:
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error enhancing content: {e}"

def simple_pdf_extract(pdf_path):
    """Simple PDF text extraction"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            return "❌ No PDF libraries available. Install: pip install PyPDF2 pdfplumber"

def main():
    """Main CLI interface"""
    print("=" * 60)
    print("🚀 SCANNIFY CLI - RAG-Powered PDF Scanner")
    print("=" * 60)
    
    # Test AI connection first
    if not test_ai_connection():
        return
    
    print("\n📋 Available Commands:")
    print("1. Test AI with sample text")
    print("2. Analyze PDF with RAG")
    print("3. Chat with PDF (RAG)")
    print("4. Enhance text content")
    print("5. Exit")
    
    while True:
        print("\n" + "-" * 40)
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            # Test with sample text
            sample_text = """
            This is a sample business report about quarterly sales performance.
            Our company achieved significant growth this quarter with revenue increasing by 15%.
            The main drivers were improved customer satisfaction and new product launches.
            However, we need to focus on reducing operational costs in the next quarter.
            Key metrics: Customer satisfaction: 95%, Revenue growth: 15%, New products launched: 3.
            """
            
            print("\n🧠 Analyzing sample text...")
            analysis = analyze_text_content(sample_text)
            print(f"\n📊 Analysis Result:\n{analysis}")
            
            print("\n✨ Enhancing for clarity...")
            enhanced = enhance_content(sample_text, "clarity")
            print(f"\n📄 Enhanced Version:\n{enhanced}")
            
        elif choice == "2" or choice == "3":
            # Analyze PDF with RAG
            pdf_path = input("Enter PDF file path: ").strip().strip('"')
            
            if not Path(pdf_path).exists():
                print("❌ File not found!")
                continue
            
            print("📖 Extracting text from PDF...")
            text = simple_pdf_extract(pdf_path)
            
            if "❌" in text:
                print(text)
                continue
                
            if not text.strip():
                print("❌ No text extracted from PDF")
                continue
            
            print(f"✅ Extracted {len(text)} characters")
            
            # Setup RAG system
            print("🧠 Setting up RAG system...")
            try:
                from pdf_scanner import pdf_scanner
                rag_success = pdf_scanner.setup_rag_system(text)
                
                if not rag_success:
                    print("❌ Failed to setup RAG system")
                    continue
                    
                print("✅ RAG system initialized!")
                
                if choice == "2":
                    # Analysis mode
                    print("\n🧠 Analyzing content with RAG...")
                    analysis = analyze_text_content(text)
                    print(f"\n📊 Analysis Result:\n{analysis}")
                    
                    # RAG-specific analysis
                    print("\n🔍 RAG Analysis - Key Topics:")
                    topics = pdf_scanner.analyze_with_rag("What are the main topics and themes in this document?")
                    print(topics)
                    
                elif choice == "3":
                    # Chat mode
                    print("\n💬 RAG Chat Mode - Ask questions about the document")
                    print("Type 'quit' to exit chat mode")
                    
                    while True:
                        question = input("\n🤔 Your question: ").strip()
                        
                        if question.lower() in ['quit', 'exit', 'q']:
                            break
                        
                        if not question:
                            continue
                        
                        print("🧠 Searching with RAG...")
                        response = pdf_scanner.ask_question_rag(question)
                        print(f"\n🤖 RAG Answer: {response['answer']}")
                        
                        # Show relevant context
                        if 'context' in response:
                            print(f"\n📄 Relevant Context Found:")
                            for i, ctx in enumerate(response['context'][:2]):
                                if hasattr(ctx, 'page_content'):
                                    content = ctx.page_content[:200] + "..."
                                else:
                                    content = str(ctx)[:200] + "..."
                                print(f"Context {i+1}: {content}")
                    
                    # Show chat history
                    history = pdf_scanner.get_chat_history()
                    print(f"\n📜 Total questions asked: {len(history)}")
                
            except ImportError:
                print("❌ RAG functionality not available. Using basic analysis...")
                analysis = analyze_text_content(text)
                print(f"\n📊 Analysis Result:\n{analysis}")
            
        elif choice == "4":
            # Enhance custom text
            print("Enter your text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            
            text = "\n".join(lines).strip()
            if not text:
                print("❌ No text entered!")
                continue
            
            enhancement_type = input("Enhancement type (clarity/professional/summary/structure): ").strip()
            if enhancement_type not in ["clarity", "professional", "summary", "structure"]:
                enhancement_type = "clarity"
            
            print(f"\n✨ Enhancing for {enhancement_type}...")
            enhanced = enhance_content(text, enhancement_type)
            print(f"\n📄 Enhanced Version:\n{enhanced}")
            
        elif choice == "5":
            print("\n👋 Thanks for using Scannify RAG CLI!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-5.")

if __name__ == "__main__":
    main()