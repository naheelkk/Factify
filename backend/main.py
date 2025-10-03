from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional
import logging
import numpy as np
import re
from datetime import datetime
from contextlib import asynccontextmanager
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class NewsRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    text: str
    explain: bool = True

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    text: str
    prediction: str
    confidence_score: float
    raw_scores: dict
    explanation: Optional[str] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    explanation_service_available: bool

# Global variables for model components
class ModelComponents:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.groq_client = None

model_components = ModelComponents()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "naheelkk/fake-news-bert-isot"
MAX_LENGTH = 512

class GroqExplanationService:
    """Groq-based explanation service with enhanced prompting and caching"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        
        self.groq_client = Groq(api_key=api_key)
        self.explanation_cache = {}
        logger.info("✅ Groq explanation service initialized")
    
    def get_cache_key(self, text: str, prediction: str, confidence: float) -> str:
        """Generate cache key for explanation"""
        text_hash = hash(text[:200])
        return f"{text_hash}_{prediction}_{int(confidence*100)}"
    
    def analyze_text_features(self, text: str) -> dict:
        """Analyze text characteristics for better explanations"""
        features = {
            'word_count': len(text.split()),
            'has_quotes': '"' in text or "'" in text,
            'has_numbers': bool(re.search(r'\d+', text)),
            'excessive_caps': len(re.findall(r'\b[A-Z]{3,}\b', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_sources': bool(re.search(r'(according to|source|study|research|report)', text, re.IGNORECASE)),
            'sensational_words': len(re.findall(r'\b(shocking|breaking|unbelievable|secret|bombshell|exposed)\b', text, re.IGNORECASE))
        }
        return features
    
    async def generate_explanation(self, text: str, prediction: str, confidence: float) -> str:
        """Generate enhanced explanation using Groq"""
        try:
            # Check cache first
            cache_key = self.get_cache_key(text, prediction, confidence)
            if cache_key in self.explanation_cache:
                logger.info("📋 Using cached explanation")
                return self.explanation_cache[cache_key]
            
            # Analyze text features
            features = self.analyze_text_features(text)
            
            # Build context for better explanations
            feature_context = []
            if features['excessive_caps'] > 3:
                feature_context.append("excessive capitalization")
            if features['exclamation_count'] > 3:
                feature_context.append("multiple exclamation marks")
            if features['sensational_words'] > 2:
                feature_context.append("sensational language")
            if features['has_sources']:
                feature_context.append("source attribution present")
            
            feature_text = f"\n\nText characteristics: {', '.join(feature_context)}" if feature_context else ""
            
            # Enhanced prompt for better explanations
            prompt = f"""You are an expert fact-checker analyzing news content credibility.

Article excerpt: "{text[:400]}{'...' if len(text) > 400 else ''}"

Classification: {prediction.upper()} (confidence: {confidence:.1%}){feature_text}

Provide a clear, professional explanation in 2-3 sentences that:
1. States whether the classification appears correct based on the content
2. Identifies specific indicators that support this classification (language patterns, source credibility, fact consistency)
3. Gives practical advice on verification

Focus on concrete evidence, not speculation. Be direct and informative."""

            # Call Groq API with optimized settings
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional fact-checking expert who provides clear, evidence-based analyses of news content. You focus on specific indicators like language patterns, source credibility, and factual consistency."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model="llama-3.1-8b-instant",  # Better model for more nuanced analysis
                max_tokens=150,
                temperature=0.3,  # Lower temperature for more consistent output
                top_p=0.9
            )
            
            explanation = chat_completion.choices[0].message.content.strip()
            
            if explanation and len(explanation) > 20:
                # Cache successful explanation
                self.explanation_cache[cache_key] = explanation
                logger.info("✅ Generated explanation using Groq")
                return explanation
            else:
                logger.warning("⚠️ Groq returned empty/short explanation, using fallback")
                return self._get_fallback_explanation(text, prediction, confidence, features)
            
        except Exception as e:
            logger.error(f"❌ Groq API error: {str(e)}")
            return self._get_fallback_explanation(text, prediction, confidence, features)
    
    def _get_fallback_explanation(self, text: str, prediction: str, confidence: float, features: dict) -> str:
        """Enhanced fallback explanation with feature analysis"""
        
        if prediction.lower() == "fake":
            indicators = []
            
            if features['excessive_caps'] > 3:
                indicators.append("excessive capitalization suggesting sensationalism")
            if features['exclamation_count'] > 3:
                indicators.append("overuse of exclamation marks")
            if features['sensational_words'] > 2:
                indicators.append("sensational language patterns")
            if not features['has_sources']:
                indicators.append("lack of credible source attribution")
            
            if indicators:
                explanation = f"This content exhibits {', '.join(indicators[:2])}, which are common in misinformation. "
            else:
                explanation = "This content shows linguistic patterns commonly associated with unreliable information. "
            
            if confidence > 0.85:
                explanation += "The model is highly confident in this classification. Always verify such claims with established fact-checking organizations."
            else:
                explanation += "Exercise caution and cross-reference with multiple trusted sources before accepting these claims."
        
        else:  # Real news
            positive_indicators = []
            
            if features['has_sources']:
                positive_indicators.append("proper source attribution")
            if features['has_quotes']:
                positive_indicators.append("direct quotes")
            if features['has_numbers'] and features['word_count'] > 100:
                positive_indicators.append("specific factual details and comprehensive reporting")
            
            if positive_indicators:
                explanation = f"This article demonstrates {' and '.join(positive_indicators[:2])}, typical of credible journalism. "
            else:
                explanation = "This content follows standard journalistic patterns and balanced reporting style. "
            
            if features['excessive_caps'] == 0 and features['sensational_words'] == 0:
                explanation += "The measured tone and lack of sensationalism further support its credibility."
            else:
                explanation += "However, always verify important claims through multiple reputable sources."
        
        return explanation

def get_enhanced_prediction(text: str):
    """Enhanced prediction with improved preprocessing and confidence calibration"""
    try:
        start_time = datetime.now()
        
        # Enhanced text preprocessing
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize with optimized settings
        inputs = model_components.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        
        inputs = {k: v.to(model_components.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_components.model(**inputs)
            logits = outputs.logits
            
            # Temperature scaling for better calibration
            temperature = 1.3
            scaled_logits = logits / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]
            
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            prediction = model_components.model.config.id2label[predicted_class]
            
            raw_scores = {
                "fake": float(probabilities[0]),
                "real": float(probabilities[1])
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"📊 Prediction: {prediction}, Confidence: {confidence:.3f}, Time: {processing_time:.3f}s")
            
            return prediction, confidence, raw_scores, processing_time
            
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise e

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Factify API...")
    try:
        logger.info(f"📥 Loading classification model: {MODEL_NAME}")
        
        # Set device
        model_components.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Using device: {model_components.device}")
        
        # Load model components
        model_components.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model_components.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model_components.model.config.id2label = {0: "fake", 1: "real"}
        model_components.model.config.label2id = {"fake": 0, "real": 1}
        model_components.model.to(model_components.device)
        model_components.model.eval()
        
        # Initialize Groq explanation service
        if GROQ_API_KEY:
            model_components.groq_client = GroqExplanationService(GROQ_API_KEY)
            logger.info("✅ Groq explanation service ready!")
        else:
            logger.warning("⚠️ GROQ_API_KEY not found - explanations will use fallback templates")
        
        logger.info("✅ Classification model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Factify API...")

# Create FastAPI app
app = FastAPI(
    title="Factify API",
    description="AI-powered fake news detection with Groq explanations",
    version="4.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_components.model is not None and model_components.tokenizer is not None,
        explanation_service_available=model_components.groq_client is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: NewsRequest):
    """
    Predict if news is fake or real with AI explanation
    
    - **text**: The news text to analyze (required)
    - **explain**: Whether to include AI explanation (default: True)
    """
    if model_components.model is None or model_components.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short for reliable analysis (minimum 20 characters)")
    
    try:
        # Get prediction
        prediction, confidence, raw_scores, processing_time = get_enhanced_prediction(request.text)
        
        explanation = None
        
        # Get explanation if requested
        if request.explain:
            if model_components.groq_client:
                explanation = await model_components.groq_client.generate_explanation(
                    request.text, prediction, confidence
                )
            else:
                # Fallback if Groq not available
                features = model_components.groq_client.analyze_text_features(request.text) if model_components.groq_client else {}
                explanation = f"Classification: {prediction} with {confidence:.1%} confidence. Note: Enhanced explanations require GROQ_API_KEY to be configured."
        
        return PredictionResponse(
            text=request.text,
            prediction=prediction,
            confidence_score=confidence,
            raw_scores=raw_scores,
            explanation=explanation,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    cache_size = len(model_components.groq_client.explanation_cache) if model_components.groq_client else 0
    
    return {
        "model_info": {
            "model_name": MODEL_NAME,
            "device": str(model_components.device) if model_components.device else "unknown",
            "max_length": MAX_LENGTH
        },
        "cache_info": {
            "explanation_cache_size": cache_size,
            "cache_enabled": model_components.groq_client is not None
        },
        "api_info": {
            "version": "4.0.0",
            "features": [
                "Enhanced BERT classification",
                "Groq-powered explanations",
                "Smart caching",
                "Text feature analysis",
                "Improved confidence calibration"
            ]
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🔍 Factify API - AI-Powered Fake News Detection",
        "version": "4.0.0",
        "features": [
            "🤖 BERT-based classification",
            "💡 Groq AI explanations",
            "⚡ Smart caching system",
            "🛡️ Enhanced confidence calibration",
            "📊 Text feature analysis"
        ],
        "explanation_service": "Groq (llama-3.1-70b)" if model_components.groq_client else "Fallback templates",
        "endpoints": {
            "health": "GET /health - Service health status",
            "predict": "POST /predict - Analyze news content",
            "stats": "GET /stats - API usage statistics",
            "docs": "GET /docs - Interactive API documentation"
        },
        "quick_test": {
            "method": "POST",
            "url": "/predict",
            "body": {
                "text": "Breaking: Scientists discover new species of dinosaur in Antarctica",
                "explain": True
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")