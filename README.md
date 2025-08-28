# 📰 Factify – AI-Powered Fake News Detection with Source Verification

Factify is an advanced machine learning system for **fake news detection** that combines **transformer-based deep learning (BERT)** with **AI-generated explanations** and **real-time source verification**. This project demonstrates the evolution from classical ML baselines to modern AI systems that provide both accurate predictions and transparent reasoning.

## 🌟 Key Features

- **🤖 AI-Powered Detection**: Fine-tuned BERT model with 95%+ accuracy
- **📝 Intelligent Explanations**: FLAN-T5 generated reasoning for each prediction
- **🔍 Source Verification**: Real-time web search and fact-checking integration
- **🎯 Confidence Scoring**: Calibrated confidence levels with temperature scaling
- **🌐 Production-Ready API**: FastAPI backend with comprehensive endpoints
- **📊 Multi-Model Support**: Baseline models + Advanced transformers

## 🚀 System Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   User Input Text   │───▶│  BERT Classification │───▶│   Prediction +      │
└─────────────────────┘    └──────────────────────┘    │   Confidence        │
                                      │                  └─────────────────────┘
                                      ▼                           │
┌─────────────────────┐    ┌──────────────────────┐              │
│  Web Search APIs    │◀───│   Claim Extraction   │              │
│  (Fact Checking)    │    │   (FLAN-T5)          │              │
└─────────────────────┘    └──────────────────────┘              │
           │                           │                          │
           ▼                           ▼                          ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Source Context    │───▶│  Explanation Model   │───▶│   Final Response    │
│   & Verification    │    │     (FLAN-T5)        │    │  with Explanation   │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## 🛠️ Tech Stack

**Core ML/AI:**
- **Classification**: Fine-tuned BERT (`naheelkk/fake-news-bert-model`)
- **Explanations**: Google FLAN-T5 for reasoning generation
- **Source Verification**: DuckDuckGo API integration
- **Framework**: PyTorch, HuggingFace Transformers

**Backend & API:**
- **FastAPI**: High-performance async API
- **Pydantic**: Request/response validation
- **CORS**: Cross-origin resource sharing
- **Logging**: Comprehensive error tracking

**Deployment:**
- **Docker**: Containerized deployment
- **Uvicorn**: ASGI server
- **Health Checks**: System monitoring endpoints

## 📊 Performance Metrics

| Model Component | Performance | Details |
|----------------|-------------|---------|
| **BERT Classification** | **95%+ Accuracy** | Fine-tuned on fake news dataset |
| **Baseline Models** | **94.18% (LogReg)** | TF-IDF + Classical ML |
| **Explanation Quality** | **Coherent & Contextual** | FLAN-T5 powered reasoning |
| **Source Verification** | **Real-time** | Web search integration |
| **API Response Time** | **<2 seconds** | Optimized inference pipeline |

## 🚦 API Endpoints

### Core Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "Your news article text here...",
  "explain": true,
  "search_sources": true
}
```

**Response:**
```json
{
  "text": "Input text",
  "prediction": "Real|Fake", 
  "confidence_score": 0.87,
  "raw_scores": {"real": 0.87, "fake": 0.13},
  "explanation": "AI-generated reasoning...",
  "sources": [
    {
      "title": "Fact-check source",
      "url": "https://...",
      "snippet": "Verification text...",
      "relevance_score": 0.9
    }
  ],
  "search_queries": ["fact check query 1", "..."]
}
```

### System Health
```http
GET /health
```

### Model Information
```http
GET /models/info
```

## 🏗️ Project Structure

```
factify/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── requirements.txt           # Python dependencies
│   └── debug_model.py            # Model testing utilities
├── frontend/
│   ├── .streamlit/               # Streamlit configuration
│   ├── config.toml              # App configuration
│   ├── requirements.txt         # Frontend dependencies
│   └── app.py                   # Streamlit interface
├── data/
│   ├── processed/               # Cleaned datasets
│   └── raw/                     # Original data files
├── models/                      # Saved model artifacts
├── notebooks/                   # Jupyter notebooks
├── src/                        # Source code modules
└── utils/                      # Utility functions
```

## 🐳 Quick Start with Docker

```bash
# Clone repository
git clone https://github.com/naheelkk/factify.git
cd factify

# Build and run with Docker Compose
docker-compose up --build

# API available at: http://localhost:8000
# Frontend at: http://localhost:8501
```

## 🖥️ Local Development

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set your own search API keys for enhanced source verification
SEARCH_API_KEY=your_google_search_api_key
SEARCH_ENGINE_ID=your_custom_search_engine_id

# Model configuration
MODEL_NAME=naheelkk/fake-news-bert-model
EXPLANATION_MODEL=google/flan-t5-small
MAX_LENGTH=512
```

### Model Customization
You can easily swap models by updating the configuration in `main.py`:
```python
MODEL_NAME = "your-huggingface-model"
EXPLANATION_MODEL = "google/flan-t5-base"  # or larger variants
```

## 🧠 AI Explanation System

The system uses a two-stage AI approach:

1. **Claim Extraction**: FLAN-T5 identifies key factual claims
2. **Source Verification**: Web search for fact-checking sources  
3. **Explanation Generation**: Context-aware reasoning using FLAN-T5
4. **Confidence Calibration**: Temperature scaling for realistic confidence scores

**Example Explanation Flow:**
```
News Text → Extract Claims → Search Sources → Generate Explanation
"Study shows..." → ["New study reveals X", "Experts claim Y"] → [Fact-check sources] → "This appears legitimate because..."
```

## 📈 Advanced Features

### Confidence Calibration
- **Temperature scaling** for realistic confidence scores
- **Conservative adjustments** to prevent overconfidence
- **Ambiguity detection** for close predictions

### Source Integration
- **Real-time web search** using DuckDuckGo API
- **Relevance scoring** for source quality assessment
- **Multiple verification queries** per article

### Robust Error Handling
- **Graceful degradation** when external services fail
- **Fallback explanations** when AI generation fails
- **Comprehensive logging** for debugging

## 🔬 Research & Development

This project demonstrates several advanced concepts:
- **Multi-modal AI**: Combining classification + generation models
- **Real-time fact-checking**: Integration with external verification sources
- **Explainable AI**: Transparent reasoning for model decisions
- **Production ML**: Scalable, robust deployment patterns

## 📋 API Testing

```bash
# Test basic prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover new planet", "explain": true}'

# Health check
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/models/info
```

## 🚀 Deployment Options

**Cloud Platforms:**
- **Render**: One-click deployment
- **Railway**: Container-based deployment  
- **AWS/GCP/Azure**: Full cloud deployment
- **Heroku**: Simple PaaS deployment

**Container Deployment:**
```bash
# Build Docker image
docker build -t factify-api .

# Run container
docker run -p 8000:8000 factify-api
```

## 📊 Monitoring & Analytics

The system includes comprehensive monitoring:
- **Health check endpoints** for system status
- **Detailed logging** for debugging
- **Performance metrics** tracking
- **Error handling** with graceful degradation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace** for transformer models and APIs
- **WELFake Dataset** for training data
- **Google Research** for FLAN-T5 model
- **FastAPI** and **Streamlit** teams for excellent frameworks

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/naheelkk/factify/issues)
- **Documentation**: Check `/docs` endpoint when API is running
- **Email**: [Your contact email]

---

**🎯 Built with ❤️ for combating misinformation through Advanced AI**

*Factify v2.0 - Now with AI-powered explanations and real-time source verification*
