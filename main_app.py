"""
Scannify - AI-Powered PDF Scanner and Enhancer
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
from pdf_scanner import pdf_scanner
import os
import time
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Scannify - AI PDF Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        color: #2c3e50 !important;
    }
    .feature-card h4 {
        color: #1f77b4 !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .feature-card p {
        color: #34495e !important;
        margin-bottom: 0;
        line-height: 1.4;
    }
    .success-box {
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        color: #155724 !important;
        font-weight: 500;
    }
    .warning-box {
        background: linear-gradient(145deg, #fff3cd, #ffeaa7);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
        color: #856404 !important;
        font-weight: 500;
    }
    .info-box {
        background: linear-gradient(145deg, #d1ecf1, #bee5eb);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
        color: #0c5460 !important;
        font-weight: 500;
    }
    /* Fix sidebar text color */
    .css-1d391kg p, .css-1d391kg h4 {
        color: inherit !important;
    }
    /* Ensure all text is readable */
    .stMarkdown {
        color: #2c3e50;
    }
    .stButton > button {
        background: linear-gradient(145deg, #1f77b4, #0d5aa7);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'enhanced_content' not in st.session_state:
    st.session_state.enhanced_content = ""
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Scannify - AI PDF Scanner & Enhancer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 Features")
        st.markdown("""
        <div class="feature-card">
        <h4>🔍 Smart Scanning</h4>
        <p>Extract text from text-based PDFs with AI analysis</p>
        </div>
        
        <div class="feature-card">
        <h4>🧠 AI Analysis</h4>
        <p>Deep content analysis with actionable insights</p>
        </div>
        
        <div class="feature-card">
        <h4>✨ Enhancement</h4>
        <p>AI-powered content improvement and restructuring</p>
        </div>
        
        <div class="feature-card">
        <h4>💬 Smart Chat</h4>
        <p>Ask questions and get AI-powered answers</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # OCR setting with status
        st.markdown("#### 🔍 OCR for Scanned PDFs")
        from pdf_scanner import OCR_AVAILABLE
        
        if OCR_AVAILABLE:
            use_ocr = st.checkbox("Enable OCR for scanned PDFs", value=True)
            st.markdown("✅ OCR Ready - Can process scanned documents")
        else:
            use_ocr = st.checkbox("Enable OCR for scanned PDFs", value=False, disabled=True)
            with st.expander("ℹ️ OCR Status", expanded=False):
                st.markdown("**OCR not available** - Text-based PDFs work fine")
                st.markdown("To enable OCR for scanned PDFs, install:")
                st.code("pip install pytesseract opencv-python")
                st.markdown("*OCR allows scanning handwritten/image-based PDFs*")
        
        # Enhancement type
        enhancement_type = st.selectbox(
            "Enhancement Focus",
            ["structure", "clarity", "professional", "summary"],
            format_func=lambda x: {
                "structure": "🏗️ Structure & Organization",
                "clarity": "🔍 Clarity & Readability", 
                "professional": "💼 Professional Polish",
                "summary": "📋 Summary & Key Points"
            }.get(x, x)
        )
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        if st.session_state.pdf_text:
            word_count = len(st.session_state.pdf_text.split())
            char_count = len(st.session_state.pdf_text)
            st.metric("Words", f"{word_count:,}")
            st.metric("Characters", f"{char_count:,}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Your PDF")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload any PDF file - text-based or scanned images"
        )
        
        if uploaded_file is not None:
            st.markdown('<div class="success-box">✅ PDF file uploaded successfully!</div>', unsafe_allow_html=True)
            
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "File Size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File Type": uploaded_file.type
            }
            
            st.markdown("#### 📋 File Information")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            
            # Processing button
            if st.button("🔄 Start AI Processing", type="primary"):
                with st.spinner("🤖 AI is analyzing your PDF..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Extract text
                        status_text.text("📖 Extracting text from PDF...")
                        progress_bar.progress(20)
                        
                        pdf_text = pdf_scanner.extract_text_from_pdf(uploaded_file)
                        
                        # If no text found and OCR enabled, try OCR
                        if not pdf_text.strip() and use_ocr:
                            from pdf_scanner import OCR_AVAILABLE
                            if OCR_AVAILABLE:
                                status_text.text("🔍 Running OCR on scanned content...")
                                progress_bar.progress(40)
                                try:
                                    pdf_text = pdf_scanner.extract_text_from_scanned_pdf(uploaded_file)
                                except Exception as ocr_error:
                                    st.warning(f"⚠️ OCR failed: {str(ocr_error)}. Proceeding with available text.")
                            else:
                                st.info("💡 This appears to be a scanned PDF, but OCR is not available. Install pytesseract and opencv-python for scanned PDF support.")
                        
                        if not pdf_text.strip():
                            st.error("❌ Could not extract text from the PDF. Please ensure it contains readable text or enable OCR for scanned documents.")
                            return
                        
                        st.session_state.pdf_text = pdf_text
                        progress_bar.progress(50)
                        
                        # Step 2: Setup RAG system
                        status_text.text("🧠 Setting up RAG system...")
                        rag_success = pdf_scanner.setup_rag_system(pdf_text)
                        if rag_success:
                            st.session_state.rag_initialized = True
                            st.session_state.chat_history = []  # Reset chat history
                        else:
                            st.session_state.rag_initialized = False
                            st.info("💡 RAG system setup failed, but you can still use basic AI features!")
                        progress_bar.progress(70)
                        
                        # Step 3: Analyze content
                        status_text.text("🔍 Analyzing content with AI...")
                        analysis = pdf_scanner.analyze_pdf_content(pdf_text)
                        st.session_state.analysis_results = analysis
                        progress_bar.progress(85)
                        
                        # Step 4: Generate enhancements
                        status_text.text("✨ Creating enhanced version...")
                        enhanced_content = pdf_scanner.generate_enhanced_content(pdf_text, enhancement_type)
                        st.session_state.enhanced_content = enhanced_content
                        progress_bar.progress(100)
                        
                        status_text.text("✅ Processing complete!")
                        st.session_state.processing_complete = True
                        
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
    
    with col2:
        if st.session_state.processing_complete:
            st.markdown("### 🎉 Processing Complete!")
            
            # Analysis Results
            st.markdown("#### 📊 AI Analysis")
            if st.session_state.analysis_results:
                st.markdown(st.session_state.analysis_results['analysis'])
            
            # Enhanced content preview
            with st.expander("📄 Enhanced Content Preview", expanded=False):
                st.markdown(st.session_state.enhanced_content[:1000] + "..." if len(st.session_state.enhanced_content) > 1000 else st.session_state.enhanced_content)
        else:
            st.markdown('<div class="info-box">👆 Upload a PDF file and click "Start AI Processing" to begin analysis</div>', unsafe_allow_html=True)
            
            # Feature preview
            st.markdown("### 🌟 What You'll Get")
            st.markdown("""
            - **📝 Complete Text Extraction** - From any PDF type
            - **🔍 Deep Content Analysis** - AI-powered insights
            - **📊 Structure Assessment** - Quality evaluation
            - **🎯 Improvement Suggestions** - Actionable recommendations
            - **✨ Enhanced Version** - AI-improved content
            - **📄 Beautiful PDF Output** - Professional formatting
            """)
    
    # Results section (full width)
    if st.session_state.processing_complete:
        st.markdown("---")
        st.markdown("## 📊 Detailed Analysis & RAG Q&A")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📖 Original Text", "🧠 AI Insights", "✨ Enhanced Content", "💬 RAG Chat", "📄 Download", "🔧 Advanced Tools"])
        
        with tab1:
            st.markdown("### 📖 Extracted Text")
            with st.expander("View Full Text", expanded=False):
                st.text_area("Original PDF Text", st.session_state.pdf_text, height=400)
            
            # Text statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Pages", "Estimated", delta=len(st.session_state.pdf_text) // 2000)
            with col2:
                st.metric("📝 Words", len(st.session_state.pdf_text.split()))
            with col3:
                st.metric("🔤 Characters", len(st.session_state.pdf_text))
            with col4:
                st.metric("📏 Reading Time", f"{len(st.session_state.pdf_text.split()) // 200} min")
        
        with tab2:
            st.markdown("### 🧠 AI-Generated Insights")
            
            # Generate smart insights
            if st.button("🔮 Generate Smart Insights"):
                with st.spinner("🧠 Generating actionable insights..."):
                    insights = pdf_scanner.generate_smart_insights(
                        st.session_state.pdf_text, 
                        st.session_state.analysis_results
                    )
                    
                    st.markdown("#### 💡 Actionable Improvement Insights")
                    for insight in insights:
                        st.markdown(f'<div class="info-box">{insight}</div>', unsafe_allow_html=True)
            
            # Visual improvement suggestions
            if st.button("🎨 Suggest Visual Improvements"):
                with st.spinner("🎨 Analyzing visual enhancement opportunities..."):
                    visual_suggestions = pdf_scanner.suggest_visual_improvements(st.session_state.pdf_text)
                    
                    st.markdown("#### 🎨 Visual Enhancement Suggestions")
                    for suggestion in visual_suggestions:
                        st.markdown(f"- {suggestion}")
        
        with tab3:
            st.markdown("### ✨ AI-Enhanced Content")
            st.markdown(st.session_state.enhanced_content)
            
            # Option to re-enhance with different focus
            st.markdown("#### 🔄 Re-enhance with Different Focus")
            new_enhancement = st.selectbox(
                "Choose new enhancement focus:",
                ["structure", "clarity", "professional", "summary"],
                format_func=lambda x: {
                    "structure": "🏗️ Structure & Organization",
                    "clarity": "🔍 Clarity & Readability", 
                    "professional": "💼 Professional Polish",
                    "summary": "📋 Summary & Key Points"
                }.get(x, x),
                key="new_enhancement"
            )
            
            if st.button("🔄 Re-enhance Content"):
                with st.spinner("✨ Creating new enhanced version..."):
                    new_enhanced = pdf_scanner.generate_enhanced_content(st.session_state.pdf_text, new_enhancement)
                    st.session_state.enhanced_content = new_enhanced
                    st.rerun()
        
        with tab4:
            st.markdown("### 💬 AI-Powered Chat")
            
            # Check if chat system is available
            chat_available = (
                st.session_state.rag_initialized or 
                hasattr(pdf_scanner, 'simple_search_text') and pdf_scanner.simple_search_text or
                st.session_state.pdf_text
            )
            
            if chat_available:
                if st.session_state.rag_initialized:
                    st.markdown("🧠 **RAG System Active** - Ask questions about your document using semantic search")
                else:
                    st.markdown("🔍 **Simple Chat Mode** - Ask questions about your document using text analysis")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    question = st.text_input("Ask anything about your document:", key="rag_question")
                with col2:
                    st.markdown("") # Spacing
                    clear_history = st.button("🗑️ Clear History")
                
                if clear_history:
                    pdf_scanner.clear_chat_history()
                    st.session_state.chat_history = []
                    st.rerun()
                
                if question and st.button("🤖 Ask AI", type="primary"):
                    with st.spinner("🧠 Analyzing your question..."):
                        response = pdf_scanner.ask_question_rag(question)
                        st.session_state.chat_history = pdf_scanner.get_chat_history()
                
                # Display chat history
                if st.session_state.chat_history or pdf_scanner.get_chat_history():
                    st.markdown("#### 📜 Chat History")
                    chat_history = pdf_scanner.get_chat_history()
                    
                    for i, chat in enumerate(reversed(chat_history[-5:])):  # Show last 5
                        with st.expander(f"Q{len(chat_history)-i}: {chat['question'][:50]}...", expanded=(i==0)):
                            st.markdown(f"**🤔 Question:** {chat['question']}")
                            st.markdown(f"**🤖 Answer:** {chat['answer']}")
                            
                            # Show relevant context
                            if 'context' in chat and chat['context']:
                                st.markdown("**📄 Relevant Context:**")
                                for j, ctx in enumerate(chat['context'][:2]):  # Show top 2 contexts
                                    if hasattr(ctx, 'page_content'):
                                        content = ctx.page_content[:200] + "..."
                                    else:
                                        content = str(ctx)[:200] + "..."
                                    st.markdown(f"```\n{content}\n```")
                
                # RAG Analytics
                if st.session_state.rag_initialized:
                    st.markdown("#### 📊 RAG Analytics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🔍 Similarity Search Test"):
                            test_query = st.text_input("Enter test query:", key="sim_test")
                            if test_query:
                                results = pdf_scanner.similarity_search_with_score(test_query, k=3)
                                for i, (content, score) in enumerate(results):
                                    st.markdown(f"**Result {i+1} (Score: {score:.3f})**")
                                    st.markdown(f"```\n{content[:150]}...\n```")
                    
                    with col2:
                        focus_area = st.selectbox("RAG Summary Focus:", 
                                                ["general", "technical", "business", "action", "risks"])
                        if st.button("📋 Generate RAG Summary"):
                            summary = pdf_scanner.rag_summarization(focus_area)
                            st.markdown(summary)
                    
                    with col3:
                        st.metric("Total Questions", len(pdf_scanner.get_chat_history()))
                        if pdf_scanner.vector_store:
                            # Get number of chunks in vector store
                            st.metric("Document Chunks", len(pdf_scanner.vector_store.docstore._dict))
                else:
                    st.markdown("#### 💬 Simple Chat Analytics")
                    st.metric("Total Questions", len(pdf_scanner.get_chat_history()))
                
                # Predefined RAG questions
                st.markdown("#### 🎯 Quick Analysis Questions")
                quick_questions = [
                    "What are the main topics of this document?",
                    "What are the key findings or conclusions?", 
                    "Are there any recommendations or action items?",
                    "What are the most important details to remember?",
                    "What problems or challenges are discussed?"
                ]
                
                cols = st.columns(3)
                for i, q in enumerate(quick_questions):
                    with cols[i % 3]:
                        if st.button(f"❓ {q}", key=f"quick_{i}"):
                            with st.spinner("🧠 Analyzing with RAG..."):
                                response = pdf_scanner.ask_question_rag(q)
                                st.session_state.chat_history = pdf_scanner.get_chat_history()
                                st.rerun()
            else:
                st.warning("⚠️ Chat system not available. Please process a PDF first.")
                
                # Demo mode for testing
                with st.expander("🔬 Try Demo Chat Mode", expanded=False):
                    st.markdown("**Test the AI chat system with sample content:**")
                    demo_text = st.text_area("Enter some text to chat about:", 
                                            value="This is a sample document about artificial intelligence and machine learning techniques.",
                                            height=100)
                    
                    if st.button("🚀 Initialize Demo Chat"):
                        if demo_text.strip():
                            # Setup demo mode
                            pdf_scanner.simple_search_text = demo_text
                            st.session_state.pdf_text = demo_text
                            st.success("✅ Demo chat mode activated! You can now ask questions about the text above.")
                            st.rerun()
                        else:
                            st.error("Please enter some text first!")
                
                st.markdown("**Available Features:**")
                st.markdown("- 🧠 **AI-powered Q&A** - Get answers based on document content")
                st.markdown("- 📚 **Simple text search** - Find relevant information")
                st.markdown("- 🔍 **Content analysis** - Deep document understanding")
                st.markdown("- 📜 **Chat History** - Track your questions and answers")
        
        with tab5:
            st.markdown("### 📄 Download Enhanced PDF")
            
            if st.button("📄 Create Enhanced PDF", type="primary"):
                with st.spinner("📄 Creating beautiful PDF..."):
                    try:
                        output_file = pdf_scanner.create_enhanced_pdf(
                            st.session_state.enhanced_content,
                            uploaded_file.name if 'uploaded_file' in locals() else "document.pdf"
                        )
                        
                        # Read the created PDF for download
                        with open(output_file, "rb") as file:
                            pdf_bytes = file.read()
                        
                        st.download_button(
                            label="📥 Download Enhanced PDF",
                            data=pdf_bytes,
                            file_name=f"enhanced_{uploaded_file.name if 'uploaded_file' in locals() else 'document.pdf'}",
                            mime="application/pdf"
                        )
                        
                        st.markdown('<div class="success-box">✅ Enhanced PDF created successfully!</div>', unsafe_allow_html=True)
                        
                        # Clean up temporary file
                        if os.path.exists(output_file):
                            os.remove(output_file)
                            
                    except Exception as e:
                        st.error(f"❌ Error creating PDF: {str(e)}")
            
            # Download enhanced text
            st.download_button(
                label="📄 Download Enhanced Text",
                data=st.session_state.enhanced_content,
                file_name=f"enhanced_{uploaded_file.name if 'uploaded_file' in locals() else 'document'}.txt",
                mime="text/plain"
            )
        
        with tab6:
            st.markdown("### 🔧 Advanced Analysis Tools")
            
            # Extract tables and figures
            if st.button("📊 Extract Tables & Figures"):
                with st.spinner("📊 Analyzing document structure..."):
                    if 'uploaded_file' in locals():
                        uploaded_file.seek(0)  # Reset file pointer
                        tables_figures = pdf_scanner.extract_tables_and_figures(uploaded_file)
                        
                        st.markdown("#### 📋 Tables Found")
                        if tables_figures['tables']:
                            for i, table in enumerate(tables_figures['tables']):
                                with st.expander(f"Table {i+1} (Page {table['page']})"):
                                    if table['data']:
                                        df = pd.DataFrame(table['data'][1:], columns=table['data'][0])
                                        st.dataframe(df)
                        else:
                            st.info("No tables detected in this document.")
                        
                        st.markdown("#### 🖼️ Figures Found")
                        if tables_figures['figures']:
                            for i, fig in enumerate(tables_figures['figures']):
                                st.write(f"Figure {i+1} - Page {fig['page']}, Size: {fig['width']}x{fig['height']}")
                        else:
                            st.info("No figures detected in this document.")
            
            # RAG-based advanced analysis
            if st.session_state.rag_initialized:
                st.markdown("#### 🧠 Advanced RAG Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Deep Topic Analysis"):
                        topics_analysis = pdf_scanner.analyze_with_rag(
                            "What are all the topics, themes, and subjects covered in this document? Provide a comprehensive breakdown with details."
                        )
                        st.markdown(topics_analysis)
                
                    if st.button("📊 Key Statistics & Numbers"):
                        stats_analysis = pdf_scanner.analyze_with_rag(
                            "What numerical data, statistics, percentages, dates, quantities, and metrics are mentioned? List and explain their significance."
                        )
                        st.markdown(stats_analysis)
                
                with col2:
                    if st.button("🎯 Critical Information"):
                        critical_analysis = pdf_scanner.analyze_with_rag(
                            "What are the most critical, important, and essential points that someone must know from this document?"
                        )
                        st.markdown(critical_analysis)
                
                    if st.button("🔗 Relationships & Connections"):
                        relationships = pdf_scanner.analyze_with_rag(
                            "What relationships, connections, dependencies, and associations are described between different concepts or elements?"
                        )
                        st.markdown(relationships)
            
            # Vector Store Analytics
            if st.session_state.rag_initialized and pdf_scanner.vector_store:
                st.markdown("#### 📊 Vector Store Analytics")
                
                total_chunks = len(pdf_scanner.vector_store.docstore._dict)
                st.metric("Document Chunks", total_chunks)
                
                # Show sample chunks
                if st.button("📄 View Document Chunks"):
                    st.markdown("**Sample Document Chunks:**")
                    sample_docs = list(pdf_scanner.vector_store.docstore._dict.values())[:3]
                    for i, doc in enumerate(sample_docs):
                        with st.expander(f"Chunk {i+1} ({len(doc.page_content)} chars)"):
                            st.markdown(f"```\n{doc.page_content[:300]}...\n```")
            
            else:
                st.info("🔧 Advanced RAG tools require document processing first.")

if __name__ == "__main__":
    main()