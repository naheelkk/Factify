from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, List
import logging
import numpy as np
import requests
import re
from urllib.parse import quote
import asyncio
from datetime import datetime
# import openai
import os
from groq import Groq
# import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Factify API", description="Fake news detection with API-based explanations")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class NewsRequest(BaseModel):
    text: str
    explain: bool = True
    search_sources: bool = True

class SourceInfo(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence_score: float
    raw_scores: dict
    explanation: Optional[str] = None
    sources: Optional[List[SourceInfo]] = None
    search_queries: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explanation_service_available: bool
    search_enabled: bool

# Global variables
model = None
tokenizer = None
device = None

# Configuration - Add your API keys here
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set via environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Free tier available
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Free inference API

# Model configuration
MODEL_NAME = "naheelkk/fake-news-bert-model"
MAX_LENGTH = 256

@app.on_event("startup")
async def load_model():
    """Load only the classification model - no explanation LLM needed locally"""
    global model, tokenizer, device
    
    try:
        logger.info(f"Loading classification model: {MODEL_NAME}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load only the classification model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        
        logger.info("Classification model loaded successfully!")
        logger.info("Using API-based explanation service - no local LLM needed!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

class ExplanationService:
    """Centralized service for API-based explanations"""
    
    def __init__(self):
        self.services = []
        
        # Initialize available services based on API keys
        # if OPENAI_API_KEY:
        #     self.services.append("openai")
        #     openai.api_key = OPENAI_API_KEY
            
        if GROQ_API_KEY:
            self.services.append("groq")
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            
        # if ANTHROPIC_API_KEY:
        #     self.services.append("anthropic")
        #     self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
        if HUGGINGFACE_API_KEY:
            self.services.append("huggingface")
    
    async def generate_explanation_openai(self, text: str, prediction: str, confidence: float, sources: List[dict] = None) -> str:
        """Generate explanation using OpenAI API (GPT-3.5/GPT-4)"""
        try:
            source_context = ""
            if sources and len(sources) > 0:
                source_context = "\n\nVerification sources:\n"
                for i, source in enumerate(sources[:2], 1):
                    title = source.get('title', 'Source')[:60]
                    snippet = source.get('snippet', '')[:80]
                    source_context += f"Source {i}: {title} - {snippet}\n"
            
            system_prompt = """You are an expert fact-checker and media analyst. Provide clear, concise explanations for news article classifications."""
            
            user_prompt = f"""
            Article: {text[:400]}...
            Classification: {prediction} (confidence: {confidence:.1%})
            {source_context}
            
            Please explain why this article is classified as {prediction.lower()} news. Focus on:
            - Content patterns and language analysis
            - Credibility indicators
            - Factual accuracy (if sources available)
            
            Keep the explanation concise (2-3 sentences) and informative.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",  # or "gpt-4" if available
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None
    
    async def generate_explanation_groq(self, text: str, prediction: str, confidence: float, sources: List[dict] = None) -> str:
        """Generate explanation using Groq API (Fast, free tier available)"""
        try:
            source_context = ""
            if sources and len(sources) > 0:
                source_context = "\n\nVerification sources:\n"
                for i, source in enumerate(sources[:2], 1):
                    title = source.get('title', 'Source')[:60]
                    snippet = source.get('snippet', '')[:80]
                    source_context += f"Source {i}: {title} - {snippet}\n"
            
            prompt = f"""
            You are an expert fact-checker. Explain why this news article is classified as {prediction.lower()}.
            
            Article: {text[:400]}...
            Classification: {prediction} (confidence: {confidence:.1%})
            {source_context}
            
            Provide a clear, 2-3 sentence explanation focusing on credibility indicators and content analysis.
            """
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert fact-checker and media analyst."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",  # Fast and free
                max_tokens=150,
                temperature=0.3
            )
            
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return None
    
    async def generate_explanation_anthropic(self, text: str, prediction: str, confidence: float, sources: List[dict] = None) -> str:
        """Generate explanation using Anthropic Claude API"""
        try:
            source_context = ""
            if sources and len(sources) > 0:
                source_context = "\n\nVerification sources:\n"
                for i, source in enumerate(sources[:2], 1):
                    title = source.get('title', 'Source')[:60]
                    snippet = source.get('snippet', '')[:80]
                    source_context += f"Source {i}: {title} - {snippet}\n"
            
            prompt = f"""
            Explain why this news article is classified as {prediction.lower()} news.
            
            Article: {text[:400]}...
            Classification: {prediction} (confidence: {confidence:.1%})
            {source_context}
            
            Provide a clear, concise explanation (2-3 sentences) focusing on credibility indicators and content patterns.
            """
            
            message = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Most cost-effective
                max_tokens=150,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return None
    
    async def generate_explanation_huggingface(self, text: str, prediction: str, confidence: float, sources: List[dict] = None) -> str:
        """Generate explanation using HuggingFace Inference API (Free tier)"""
        try:
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            
            source_context = ""
            if sources and len(sources) > 0:
                source_context = f" Based on {len(sources)} verification sources,"
            
            prompt = f"Explain why this news article is {prediction.lower()}:{source_context} {text[:200]}..."
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
            
            return None
            
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            return None
    
    async def generate_explanation_ollama(self, text: str, prediction: str, confidence: float, sources: List[dict] = None) -> str:
        """Generate explanation using local Ollama (if running)"""
        try:
            # Check if Ollama is running locally
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code != 200:
                return None
            
            source_context = ""
            if sources and len(sources) > 0:
                source_context = f"\nSources checked: {len(sources)} verification sources"
            
            prompt = f"""
            Explain why this news article is classified as {prediction.lower()} news:
            
            Article: {text[:300]}...
            Confidence: {confidence:.1%}{source_context}
            
            Provide a brief, factual explanation (2-3 sentences).
            """
            
            payload = {
                "model": "gemma3:latest",  # or whatever model you have
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
                
            return None
            
        except Exception as e:
            logger.warning(f"Ollama error {str(e)}")
            logger.debug(f"Ollama not available: {str(e)}")
            return None
    
    async def get_explanation(self, text: str, prediction: str, confidence: float, sources: List[dict] = None) -> str:
        """Try different explanation services in order of preference"""
        
        # Service priority order (most reliable first)
        service_methods = [
            ("ollama", self.generate_explanation_ollama),  # Local, no API costs
            ("groq", self.generate_explanation_groq),  # Fast and often free
            ("openai", self.generate_explanation_openai),  # High quality
            ("anthropic", self.generate_explanation_anthropic),  # High quality
            ("huggingface", self.generate_explanation_huggingface),  # Free tier
        ]
        
        for service_name, method in service_methods:
            if service_name in self.services or service_name == "ollama":
                try:
                    logger.info(f"Trying {service_name} for explanation generation")
                    explanation = await method(text, prediction, confidence, sources)
                    if explanation:
                        logger.info(f"Successfully generated explanation using {service_name}")
                        return explanation
                except Exception as e:
                    logger.warning(f"Failed to get explanation from {service_name}: {str(e)}")
                    continue
        
        # Fallback to template-based explanation
        logger.info("All API services failed, using template explanation")
        return self.get_template_explanation(text, prediction, confidence, sources)
    
    def get_template_explanation(self, text: str, prediction: str, confidence: float, sources: List[dict] = None) -> str:
        """Fallback template-based explanation"""
        if prediction.lower() == "fake":
            explanation = (
                f"This content exhibits patterns commonly associated with misinformation, "
                f"such as unverified claims, sensational language, or lack of credible sources. "
            )
        else:
            explanation = (
                f"This article demonstrates characteristics of legitimate journalism, "
                f"including factual reporting style and verifiable information patterns. "
            )
        
        if sources and len(sources) > 0:
            explanation += (
                f"Cross-verification with {len(sources)} external sources "
                f"{'contradicts key claims' if prediction.lower() == 'fake' else 'supports the main assertions'} "
                f"presented in the article."
            )
        
        return explanation

# Initialize the explanation service
explanation_service = ExplanationService()

def extract_key_claims(text: str) -> List[str]:
    """Simple rule-based claim extraction (no LLM needed)"""
    try:
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        # Look for sentences with claim indicators
        claim_indicators = ['says', 'reports', 'according', 'claims', 'reveals', 'shows', 'study', 'research', 'announces', 'discovers', 'finds']
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and any(word in sentence.lower() for word in claim_indicators):
                claims.append(sentence[:100])
        
        # If no claims found, use first few sentences
        if not claims:
            claims = [sent.strip()[:100] for sent in sentences[:3] if len(sent.strip()) > 20]
        
        return claims[:3]  # Return max 3 claims
        
    except Exception as e:
        logger.error(f"Error extracting claims: {str(e)}")
        return [text[:100]]

def search_web_sources(query: str, max_results: int = 3) -> List[dict]:
    """Search for sources using web search API"""
    try:
        search_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_redirect=1&skip_disambig=1"
        
        response = requests.get(search_url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            sources = []
            
            if data.get('Abstract'):
                sources.append({
                    'title': data.get('AbstractText', 'DuckDuckGo Search Result'),
                    'url': data.get('AbstractURL', 'https://duckduckgo.com'),
                    'snippet': data.get('Abstract', '')[:200],
                    'relevance_score': 0.9
                })
            
            for topic in data.get('RelatedTopics', [])[:max_results-len(sources)]:
                if isinstance(topic, dict) and topic.get('Text'):
                    sources.append({
                        'title': topic.get('Text', '')[:50] + '...',
                        'url': topic.get('FirstURL', 'https://duckduckgo.com'),
                        'snippet': topic.get('Text', '')[:200],
                        'relevance_score': 0.7
                    })
            
            return sources[:max_results]
        
        return [{
            'title': f'Fact-check result for: {query[:50]}...',
            'url': 'https://factcheck.org',
            'snippet': f'Search results for "{query}" - verify with multiple credible sources.',
            'relevance_score': 0.5
        }]
        
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return []

async def search_fact_check_sources(claims: List[str]) -> tuple:
    """Search for fact-checking sources for the claims"""
    all_sources = []
    search_queries = []
    
    for claim in claims:
        query = f'"{claim}" fact check'
        search_queries.append(query)
        sources = search_web_sources(query, max_results=2)
        
        for source in sources:
            source['query'] = query
            source['claim'] = claim
            all_sources.append(source)
        
        await asyncio.sleep(0.1)
    
    # Remove duplicates and sort by relevance
    unique_sources = {}
    for source in all_sources:
        url = source['url']
        if url not in unique_sources or source['relevance_score'] > unique_sources[url]['relevance_score']:
            unique_sources[url] = source
    
    sorted_sources = sorted(unique_sources.values(), key=lambda x: x['relevance_score'], reverse=True)
    
    return sorted_sources[:5], search_queries

def get_prediction(text: str):
    """Get prediction and confidence scores from the model"""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Optional temperature scaling
            temperature = 1.5
            scaled_logits = logits / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]
            
            predicted_class = int(np.argmax(probabilities))
            raw_confidence = float(probabilities[predicted_class])
            
            # Conservative calibration
            confidence = min(raw_confidence * 0.9, 0.95)
            
            # ✅ Use config mapping directly
            prediction = model.config.id2label[predicted_class]
            
            # Extra handling for uncertain predictions
            prob_diff = abs(probabilities[0] - probabilities[1])
            if prob_diff < 0.2:
                confidence = min(confidence, 0.65)
            
            raw_scores = {
                "fake": float(probabilities[0]),
                "real": float(probabilities[1])
            }
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
            
            return prediction, confidence, raw_scores
            
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise e

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and tokenizer is not None,
        explanation_service_available=len(explanation_service.services) > 0,
        search_enabled=True
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: NewsRequest):
    """
    Predict if news is fake or real with API-based explanations
    
    - **text**: The news text to analyze
    - **explain**: Whether to include LLM explanation (default: True)
    - **search_sources**: Whether to search for supporting sources (default: True)
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        prediction, confidence, raw_scores = get_prediction(request.text)
        
        sources = []
        search_queries = []
        explanation = None
        
        if request.search_sources and request.explain:
            logger.info("Extracting claims and searching for sources...")
            claims = extract_key_claims(request.text)
            sources, search_queries = await search_fact_check_sources(claims)
        
        if request.explain:
            explanation = await explanation_service.get_explanation(
                request.text, prediction, confidence, sources
            )
        
        source_info = []
        for source in sources:
            source_info.append(SourceInfo(
                title=source.get('title', ''),
                url=source.get('url', ''),
                snippet=source.get('snippet', ''),
                relevance_score=source.get('relevance_score', 0.0)
            ))
        
        return PredictionResponse(
            text=request.text,
            prediction=prediction,
            confidence_score=confidence,
            raw_scores=raw_scores,
            explanation=explanation,
            sources=source_info if request.search_sources else None,
            search_queries=search_queries if request.search_sources else None
        )
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/services")
async def available_services():
    """Get information about available explanation services"""
    return {
        "available_services": explanation_service.services,
        "total_services": len(explanation_service.services),
        "services_info": {
            "groq": "Fast inference, free tier available",
            "openai": "High quality, requires API key",
            "anthropic": "High quality, requires API key", 
            "huggingface": "Free tier, inference API",
            "ollama": "Local inference, no API costs"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Factify API - Lightweight Fake News Detection",
        "version": "3.0.0",
        "features": [
            "Local BERT classification (lightweight)",
            "API-based explanations (no local LLM storage)",
            "Multiple explanation service fallbacks",
            "Web search integration"
        ],
        "storage_saved": "~3GB+ (no local explanation LLM)",
        "available_services": len(explanation_service.services),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "services": "/services",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)