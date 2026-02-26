# 📄 Scannify - AI-Powered PDF Scanner & Enhancer

> Transform your PDFs with the power of AI! Scan, analyze, summarize, and enhance any PDF document using Google's generative AI.
>Video Demonstration - https://youtu.be/WlriddTdNNI

## ✨ Features

🔍 **Smart PDF Scanning**
- Extract text from any PDF (text-based or scanned)
- Advanced OCR for scanned documents
- Support for complex layouts and tables

🧠 **AI-Powered Analysis**
- Deep content analysis and insights
- Document type detection
- Structure quality assessment
- Audience identification

📊 **Content Enhancement**
- AI-driven content restructuring
- Professional formatting improvements
- Clarity and readability enhancement
- Summary generation

📄 **Beautiful PDF Generation**
- Create enhanced PDFs with professional formatting
- Improved typography and layout
- Custom styling and branding options

🎯 **Advanced Features**
- Table and figure extraction
- Visual improvement suggestions
- Interactive document chat
- Multiple enhancement modes

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# 1. Run the setup script
python setup.py

# 2. Configure your API key in .env file
# Get your key from: https://makersuite.google.com/app/apikey

# 3. Run the application
streamlit run main_app.py
```

### Option 2: Windows Quick Launch
```bash
# Double-click run.bat or run in terminal:
run.bat
```

### Option 3: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Install Tesseract OCR (for scanned PDFs)
# Windows: Download from https://github.com/UB-Mannheim/tesseract
# macOS: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr

# 4. Run the application
streamlit run main_app.py
```

## 🔧 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection (for AI processing)

### Dependencies
- **Streamlit** - Web interface
- **Google Generative AI** - AI processing
- **PyPDF2/PDFPlumber** - PDF processing
- **Tesseract OCR** - Scanned document processing
- **ReportLab** - PDF generation
- **OpenCV/Pillow** - Image processing

## 📚 Usage Guide

### 1. Upload Your PDF
- Click "Choose a PDF file" to upload your document
- Supports both text-based and scanned PDFs
- Files up to 50MB are supported

### 2. AI Processing
- Click "Start AI Processing" to begin analysis
- The AI will extract, analyze, and enhance your content
- Processing typically takes 30-60 seconds

### 3. Review Results
- **Original Text**: View extracted content
- **AI Insights**: Get analysis and improvement suggestions
- **Enhanced Content**: See the AI-improved version
- **Download**: Get your enhanced PDF

### 4. Advanced Features
- **Extract Tables & Figures**: Analyze document structure
- **Chat with Document**: Ask questions about your PDF
- **Visual Improvements**: Get suggestions for charts and graphics

## 🎛️ Enhancement Modes

**🏗️ Structure & Organization**
- Improved headings and sections
- Better paragraph organization
- Logical content flow

**🔍 Clarity & Readability**
- Simplified language
- Shorter sentences
- Better explanations

**💼 Professional Polish**
- Professional tone
- Enhanced vocabulary
- Error corrections

**📋 Summary & Key Points**
- Executive summaries
- Key insights extraction
- Action items identification

## 🔒 Privacy & Security

- **Local Processing**: Most operations run locally
- **API Usage**: Only text analysis sent to Google AI
- **No Data Storage**: Files are not permanently stored
- **Temporary Files**: Automatically cleaned up

## 🛠️ Configuration

### Environment Variables (.env)
```env
# Required: Google AI API Key
GEMINI_API_KEY=your_api_key_here

# Optional: Tesseract Path (if not in system PATH)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# App Settings
MAX_FILE_SIZE_MB=50
DEFAULT_DPI=300
OCR_LANGUAGE=eng
```

### API Key Setup
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file
4. Restart the application

## 📊 Supported File Types

**Input Formats:**
- PDF (text-based)
- PDF (scanned/image-based)
- Multi-page PDFs
- Password-protected PDFs (with password)

**Output Formats:**
- Enhanced PDF
- Text files
- Markdown format
- Analysis reports

## 🎨 Customization

### Custom Styling
Modify the CSS in `main_app.py` to change:
- Color scheme
- Fonts and typography
- Layout and spacing
- Brand elements

### Enhancement Templates
Add custom enhancement prompts in `pdf_scanner.py`:
```python
enhancement_prompts = {
    "custom": "Your custom prompt here...",
    # Add more custom modes
}
```

## 🔧 Troubleshooting

### Common Issues

**📋 "Could not extract text from PDF"**
- Enable OCR for scanned documents
- Check if Tesseract is properly installed
- Verify PDF is not corrupted

**🔑 "API Key Error"**
- Verify GEMINI_API_KEY is set in .env
- Check API key validity
- Ensure you have API quota remaining

**💾 "Import Error"**
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Try creating a virtual environment

**🖼️ "OCR Not Working"**
- Install Tesseract OCR
- Add Tesseract to system PATH
- Install language packs if needed

### Performance Optimization

**For Large PDFs:**
- Enable text extraction only (disable OCR)
- Process pages individually
- Use summary mode first

**For Better OCR:**
- Ensure high-resolution scans (300+ DPI)
- Use clean, high-contrast images
- Install additional language packs

## 📝 Development

### Project Structure
```
Scannify/
├── main_app.py          # Main Streamlit application
├── pdf_scanner.py       # Core PDF processing logic
├── chat_pdf.py          # Document chat functionality
├── requirements.txt     # Python dependencies
├── .env                 # Environment configuration
├── setup.py            # Automated setup script
├── run.bat             # Windows launcher
└── README.md           # This file
```

### Adding Custom Features
1. Modify `pdf_scanner.py` for core functionality
2. Update `main_app.py` for UI changes
3. Add new dependencies to `requirements.txt`
4. Test with various PDF types

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Include tests and documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google AI** for the Gemini API
- **Streamlit** for the amazing framework
- **Tesseract** for OCR capabilities
- **ReportLab** for PDF generation
- **LangChain** for AI orchestration

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check the Wiki
- **Community**: Join our Discord server
- **Email**: support@scannify.com

---

**Made with ❤️ and AI** | **© 2026 Scannify Project**

---

### 🚀 Ready to transform your PDFs? 

```bash
python setup.py && streamlit run main_app.py
```

Happy Scanning! 📄✨
