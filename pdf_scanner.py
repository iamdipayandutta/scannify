"""
PDF Scanner and AI Enhancement System
Comprehensive PDF processing with Google AI integration
"""

import os
import io
import base64
from typing import List, Dict, Tuple, Optional
from PIL import Image
import streamlit as st
import PyPDF2
import pdfplumber
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ImportError:
    # Fallback for different langchain versions
    create_stuff_documents_chain = None
    create_retrieval_chain = None
import dotenv
import re

# Optional OCR dependencies
try:
    import pytesseract
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    # Don't show warning here, show it only when OCR is actually needed

# Optional PyMuPDF for better image extraction
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

dotenv.load_dotenv()

class PDFScanner:
    """Advanced PDF scanning and processing with AI capabilities and RAG"""
    
    def __init__(self):
        self.api_key = dotenv.get_key(dotenv.find_dotenv(), "GEMINI_API_KEY")
        os.environ["GOOGLE_API_KEY"] = self.api_key
        
        # Try multiple models in order of preference (quotas vary)
        self.models = [
            "gemini-2.5-flash",       # Working model with quota
            "gemini-2.0-flash",       # Backup
            "gemini-2.5-pro",         # Pro version
            "gemini-2.0-flash-lite"   # Lite version
        ]
        
        # Initialize with first available model
        self.current_model_index = 0
        self.llm = self._initialize_llm()
        
        # Initialize embeddings with multiple fallbacks
        self.embeddings = None
        embedding_models = [
            "models/embedding-001",
            "embedding-001", 
            "models/text-embedding-gecko-001",
            "text-embedding-gecko-001",
            "models/embedding-gecko-001"
        ]
        
        for model in embedding_models:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
                print(f"✅ Successfully initialized embeddings with model: {model}")
                break
            except Exception as e:
                print(f"❌ Failed to initialize {model}: {str(e)}")
                continue
        
        if not self.embeddings:
            print("⚠️ Could not initialize any embedding model. RAG features will be limited.")
        self.vector_store = None
        self.retriever = None
        self.retrieval_chain = None
        self.chat_history = []
        self.simple_search_text = ""  # Fallback for when embeddings fail
        self.retry_count = 0
        self.max_retries = 3
        
    def _initialize_llm(self):
        """Initialize LLM with fallback models"""
        for i, model in enumerate(self.models):
            try:
                llm = ChatGoogleGenerativeAI(model=model)
                # Test the model with a simple query
                test_response = llm.invoke("Hello")
                print(f"✅ Successfully initialized with model: {model}")
                self.current_model_index = i
                return llm
            except Exception as e:
                print(f"❌ Failed to initialize {model}: {str(e)}")
                continue
        
        # If all models fail, use the first one and handle errors later
        print("⚠️ All models failed during initialization, using fallback")
        return ChatGoogleGenerativeAI(model=self.models[0])
        
    def _switch_to_next_model(self):
        """Switch to next available model when quota is exhausted"""
        if self.current_model_index < len(self.models) - 1:
            self.current_model_index += 1
            new_model = self.models[self.current_model_index]
            print(f"🔄 Switching to model: {new_model}")
            self.llm = ChatGoogleGenerativeAI(model=new_model)
            return True
        return False
        
    def _handle_api_error(self, error):
        """Handle API errors with appropriate fallbacks"""
        error_str = str(error).lower()
        
        if "resource_exhausted" in error_str or "429" in error_str:
            if "retry in" in error_str:
                # Extract retry time
                import re
                retry_match = re.search(r'retry in ([\d.]+)s', error_str)
                retry_time = retry_match.group(1) if retry_match else "unknown"
                
                if self._switch_to_next_model():
                    return f"⚠️ Quota exhausted for current model. Switched to backup model. Please try again."
                else:
                    return f"⚠️ All models have quota issues. Please wait {retry_time} seconds and try again, or upgrade your Google AI plan."
            else:
                if self._switch_to_next_model():
                    return "⚠️ Quota exhausted. Switched to backup model. Please try again."
                else:
                    return "⚠️ All models are currently quota exhausted. Please wait and try again later."
        
        elif "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "❌ Invalid API key. Please check your Google AI API key in the .env file."
        
        elif "model not found" in error_str:
            if self._switch_to_next_model():
                return "⚠️ Model not available. Switched to backup model. Please try again."
            else:
                return "❌ No available models found. Please check your Google AI account."
        
        else:
            return f"❌ API Error: {str(error)}"
            
    def _safe_invoke(self, prompt, max_retries=2):
        """Safely invoke LLM with error handling and retries"""
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
                return response, None
            except Exception as e:
                error_msg = self._handle_api_error(e)
                
                # If we switched models, try again immediately
                if "switched to backup model" in error_msg.lower() and attempt < max_retries:
                    continue
                
                # For quota errors, don't retry immediately
                if "quota" in error_msg.lower():
                    return None, error_msg
                    
                # For other errors, retry once more
                if attempt < max_retries:
                    continue
                    
                return None, error_msg
                
        return None, "❌ All retry attempts failed"
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        try:
            # Method 1: Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                return text
            
            # Method 2: PyPDF2 as fallback
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            
        return text
    
    def extract_text_from_scanned_pdf(self, pdf_file) -> str:
        """Extract text from scanned PDFs using OCR"""
        if not OCR_AVAILABLE:
            return "To process scanned PDFs, please install OCR dependencies: pip install pytesseract opencv-python"
        
        text = ""
        
        try:    
            if PYMUPDF_AVAILABLE:
                # Convert PDF pages to images and apply OCR using PyMuPDF
                pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
                
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    
                    # Convert to image
                    mat = fitz.Matrix(2.0, 2.0)  # Scale up for better OCR
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Convert to OpenCV image
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Preprocess image for better OCR
                    img = self.preprocess_image_for_ocr(img)
                    
                    # Apply OCR
                    page_text = pytesseract.image_to_string(img, config='--psm 6')
                    text += page_text + "\n"
                    
                pdf_document.close()
            else:
                # Fallback message if PyMuPDF not available
                text = "For better OCR support, install PyMuPDF: pip install PyMuPDF"
            
        except Exception as e:
            st.error(f"OCR extraction failed: {str(e)}")
            
        return text
    
    def preprocess_image_for_ocr(self, image):
        """Preprocess image for better OCR results"""
        if not OCR_AVAILABLE:
            return image
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Increase contrast
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            return thresh
        except Exception:
            # Return original image if preprocessing fails
            return image
    
    def analyze_pdf_content(self, text: str) -> Dict:
        """Analyze PDF content using AI"""
        analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze this document text and provide a comprehensive analysis:
        
        Text: {text}
        
        Please provide analysis in this format:
        
        **SUMMARY:** (2-3 sentence summary)
        
        **MAIN TOPICS:** (List key topics covered)
        
        **DOCUMENT TYPE:** (Report, Article, Manual, etc.)
        
        **KEY INSIGHTS:** (Important findings or main points)
        
        **STRUCTURE QUALITY:** (Rate 1-10 and explain)
        
        **IMPROVEMENT SUGGESTIONS:** (How to make it more effective)
        
        **AUDIENCE:** (Target audience for this document)
        
        Be specific and actionable in your analysis.
        """)
        
        response, error = self._safe_invoke(analysis_prompt.format_prompt(text=text[:8000]))
        if error:
            return {"analysis": f"Analysis failed: {error}"}
        return {"analysis": response.content}
    
    def generate_enhanced_content(self, original_text: str, improvement_type: str) -> str:
        """Generate enhanced version of the content"""
        
        enhancement_prompts = {
            "structure": """
            Improve the structure and organization of this document:
            
            Original text: {text}
            
            Please rewrite with:
            - Clear headings and subheadings
            - Better paragraph organization
            - Logical flow
            - Executive summary at top
            - Conclusion at end
            
            Maintain all original information but improve readability and structure.
            """,
            
            "clarity": """
            Improve the clarity and readability of this document:
            
            Original text: {text}
            
            Please rewrite with:
            - Simpler, clearer language
            - Shorter sentences
            - Better explanations
            - Remove jargon where possible
            - Add transitions between ideas
            
            Keep all important information but make it more accessible.
            """,
            
            "professional": """
            Make this document more professional and polished:
            
            Original text: {text}
            
            Please rewrite with:
            - Professional tone
            - Proper formatting
            - Consistent style
            - Error corrections
            - Enhanced vocabulary where appropriate
            
            Maintain the core message but elevate the presentation quality.
            """,
            
            "summary": """
            Create a comprehensive summary with key insights:
            
            Original text: {text}
            
            Create:
            - Executive Summary
            - Key Points (bullet format)
            - Main Conclusions
            - Action Items (if any)
            - Recommendations
            
            Make it concise yet comprehensive.
            """
        }
        
        prompt_template = enhancement_prompts.get(improvement_type, enhancement_prompts["structure"])
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        response, error = self._safe_invoke(prompt.format_prompt(text=original_text[:6000]))
        if error:
            return f"Enhancement failed: {error}. Original text: {original_text}"
        return response.content
    
    def create_enhanced_pdf(self, enhanced_content: str, filename: str = "enhanced_document.pdf") -> str:
        """Create a new beautifully formatted PDF from enhanced content"""
        
        output_path = f"enhanced_{filename}"
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            leftIndent=0
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            textColor=colors.black,
            leftIndent=0,
            rightIndent=0,
            wordWrap='breakWord'
        )
        
        # Parse enhanced content and format
        lines = enhanced_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 12))
                continue
                
            # Headers
            if line.startswith('**') and line.endswith('**'):
                header_text = line.replace('**', '')
                story.append(Paragraph(header_text, heading_style))
                story.append(Spacer(1, 6))
            
            # Bullet points
            elif line.startswith('-') or line.startswith('•'):
                bullet_text = line[1:].strip()
                story.append(Paragraph(f"• {bullet_text}", body_style))
            
            # Regular paragraphs
            else:
                story.append(Paragraph(line, body_style))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def extract_tables_and_figures(self, pdf_file) -> Dict:
        """Extract tables and figures information from PDF"""
        tables = []
        figures_info = []
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables, 1):
                        if table:
                            tables.append({
                                'page': page_num,
                                'table_number': table_num,
                                'data': table,
                                'rows': len(table),
                                'columns': len(table[0]) if table else 0
                            })
                    
                    # Extract images/figures info
                    images = page.images
                    for img_num, img in enumerate(images, 1):
                        figures_info.append({
                            'page': page_num,
                            'figure_number': img_num,
                            'bbox': img['bbox'],
                            'width': img['width'],
                            'height': img['height']
                        })
                        
        except Exception as e:
            st.error(f"Error extracting tables/figures: {str(e)}")
            
        return {
            'tables': tables,
            'figures': figures_info
        }
    
    def generate_smart_insights(self, text: str, analysis: Dict) -> List[str]:
        """Generate actionable insights for document improvement"""
        
        insights_prompt = ChatPromptTemplate.from_template("""
        Based on this document analysis, provide 5-7 specific, actionable insights for improvement:
        
        Document text: {text}
        Analysis: {analysis}
        
        Provide insights in this format:
        1. [SPECIFIC IMPROVEMENT AREA]: [Actionable suggestion]
        
        Focus on:
        - Content gaps
        - Structural improvements  
        - Clarity enhancements
        - Professional presentation
        - Audience engagement
        - Visual elements
        - Call-to-action improvements
        
        Make each insight specific and implementable.
        """)
        
        response, error = self._safe_invoke(
            insights_prompt.format_prompt(
                text=text[:4000], 
                analysis=str(analysis)
            )
        )
        
        if error:
            return [f"Error generating insights: {error}"]
        
        # Parse insights into list
        insights = []
        for line in response.content.split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                insights.append(line.strip())
                
        return insights
    
    def suggest_visual_improvements(self, text: str) -> List[str]:
        """Suggest visual improvements for the document"""
        
        visual_prompt = ChatPromptTemplate.from_template("""
        Based on this document content, suggest specific visual improvements:
        
        Content: {text}
        
        Suggest improvements for:
        1. Charts/graphs that could be added
        2. Infographics opportunities  
        3. Table formatting improvements
        4. Layout enhancements
        5. Color scheme suggestions
        6. Typography improvements
        
        Be specific about what visual elements would enhance understanding.
        """)
        
        response, error = self._safe_invoke(visual_prompt.format_prompt(text=text[:4000]))
        
        if error:
            return [f"Error generating visual suggestions: {error}"]
        
        suggestions = []
        for line in response.content.split('\n'):
            if line.strip() and (line.startswith('-') or re.match(r'^\d+\.', line.strip())):
                suggestions.append(line.strip())
                
        return suggestions

    def setup_rag_system(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> bool:
        """Setup RAG system with vector embeddings and retrieval"""
        try:
            # Check if embeddings are available
            if not self.embeddings:
                st.info("💡 Embedding models not available. Using simple text-based search...")
                # Store text for simple search fallback
                self.simple_search_text = text
                return False
                
            # Create document object
            doc = Document(page_content=text, metadata={"source": "uploaded_pdf"})
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            split_docs = text_splitter.split_documents([doc])
            
            # Create vector store
            if split_docs:
                # Test embedding with a small sample first
                test_text = split_docs[0].page_content[:100]
                test_embedding = self.embeddings.embed_query(test_text)
                
                # If test succeeds, create full vector store
                self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 4}  # Return top 4 most relevant chunks
                )
                
                # Setup retrieval chain if available
                if create_stuff_documents_chain and create_retrieval_chain:
                    prompt = ChatPromptTemplate.from_template("""
                    Answer the following question based only on the provided context from the document:
                    
                    Context: {context}
                    
                    Question: {input}
                    
                    Answer: Provide a detailed answer based on the context. If the answer is not in the context, say "I cannot find this information in the document."
                    """)
                    
                    document_chain = create_stuff_documents_chain(self.llm, prompt)
                    self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
                else:
                    # Simple RAG without chains
                    self.retrieval_chain = None
                
                st.success("✅ RAG system setup successful!")
                return True
            else:
                st.warning("No text chunks created for RAG system")
                return False
                
        except Exception as e:
            error_msg = str(e).lower()
            if "not_found" in error_msg or "404" in error_msg:
                st.info("💡 Falling back to simple text-based search...")
                # Store text for simple search fallback
                self.simple_search_text = text
                return False
            else:
                st.error(f"Error setting up RAG system: {str(e)}")
                st.info("💡 Falling back to simple text-based search...")
                # Store text for simple search fallback 
                self.simple_search_text = text
                return False
    
    def ask_question_rag(self, question: str) -> Dict:
        """Ask question using RAG system or simple text search"""
        # Check what's available for answering questions
        search_text = getattr(self, 'simple_search_text', '')
        
        if not self.vector_store and not search_text:
            return {"answer": "No document content available. Please process a document first."}
        
        try:
            if self.vector_store and self.retrieval_chain:
                # Use retrieval chain if available
                response = self.retrieval_chain.invoke({"input": question})
            elif self.vector_store:
                # Simple RAG without chains
                relevant_docs = self.vector_store.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                prompt = f"""
                Based on the following context from the document, answer the question:
                
                Context: {context}
                
                Question: {question}
                
                Answer: Provide a detailed answer based on the context. If the answer is not in the context, say "I cannot find this information in the document."
                """
                
                ai_response, error = self._safe_invoke(prompt)
                if error:
                    response = {
                        "answer": f"Error answering question: {error}",
                        "context": relevant_docs
                    }
                else:
                    response = {
                        "answer": ai_response.content,
                        "context": relevant_docs
                    }
            else:
                # Fallback to simple text search
                if not search_text:
                    return {"answer": "No document text available for search"}
                    
                # Simple keyword matching for context
                words = question.lower().split()
                sentences = search_text.split('.')
                
                relevant_sentences = []
                for sentence in sentences:
                    if any(word in sentence.lower() for word in words):
                        relevant_sentences.append(sentence.strip())
                
                context = ". ".join(relevant_sentences[:3]) if relevant_sentences else search_text[:2000]
                
                prompt = f"""
                Based on the following text, answer the question:
                
                Text: {context}
                
                Question: {question}
                
                Answer: Provide an answer based on the text. If you cannot find relevant information, say "I cannot find this information in the document."
                """
                
                ai_response, error = self._safe_invoke(prompt)
                if error:
                    response = {
                        "answer": f"Error answering question: {error}",
                        "context": relevant_sentences
                    }
                else:
                    response = {
                        "answer": ai_response.content,
                        "context": relevant_sentences
                    }
            
            # Add to chat history
            self.chat_history.append({
                "question": question,
                "answer": response["answer"],
                "context": response.get("context", [])
            })
            
            return response
        except Exception as e:
            return {"answer": f"Error processing question: {str(e)}"}
    
    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """Get relevant chunks for a query using similarity search"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            st.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def analyze_with_rag(self, question: str) -> str:
        """Analyze document using RAG with custom question"""
        if not self.retrieval_chain:
            return "RAG system not initialized. Please process a document first."
        
        try:
            # Get relevant context
            relevant_chunks = self.get_relevant_chunks(question, k=3)
            
            if not relevant_chunks:
                return "No relevant information found in the document."
            
            # Create analysis prompt
            context = "\n\n".join(relevant_chunks)
            analysis_prompt = f"""
            Based on the following document excerpts, analyze and answer: {question}
            
            Document Context:
            {context}
            
            Provide a comprehensive analysis with specific examples from the text.
            """
            
            response, error = self._safe_invoke(analysis_prompt)
            if error:
                return f"RAG analysis failed: {error}"
            return response.content
            
        except Exception as e:
            return f"Error in RAG analysis: {str(e)}"
    
    def get_chat_history(self) -> List[Dict]:
        """Get chat history"""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = []
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Perform similarity search with relevance scores"""
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return []
    
    def rag_summarization(self, focus_area: str = "general") -> str:
        """Advanced summarization using RAG"""
        if not self.vector_store:
            return "RAG system not initialized."
        
        try:
            # Define different focus areas
            focus_queries = {
                "general": "What are the main topics and key points of this document?",
                "technical": "What are the technical details and specifications mentioned?",
                "business": "What are the business implications and opportunities discussed?",
                "action": "What action items and recommendations are provided?",
                "risks": "What risks, challenges, or concerns are identified?"
            }
            
            query = focus_queries.get(focus_area, focus_queries["general"])
            
            # Get comprehensive context
            relevant_chunks = self.get_relevant_chunks(query, k=6)
            
            if not relevant_chunks:
                return "No content available for summarization."
            
            context = "\n\n".join(relevant_chunks)
            
            summary_prompt = f"""
            Create a comprehensive summary of this document focusing on: {focus_area}
            
            Document Content:
            {context}
            
            Provide:
            1. Executive Summary (2-3 sentences)
            2. Key Points (bullet format)
            3. Main Findings/Insights
            4. Relevant Details
            
            Make it detailed yet concise.
            """
            
            response, error = self._safe_invoke(summary_prompt)
            if error:
                return f"RAG summarization failed: {error}"
            return response.content
            
        except Exception as e:
            return f"Error in RAG summarization: {str(e)}"

# Initialize the scanner
pdf_scanner = PDFScanner()