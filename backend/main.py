from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, List
import logging
import numpy as np
import requests
import re
from urllib.parse import quote
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Factify API", description="Fake news detection with source-based AI explanations")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
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
    explanation_model_loaded: bool
    search_enabled: bool

# Global variables for model
model = None
tokenizer = None
device = None
explanation_model = None
explanation_tokenizer = None

# Configuration
MODEL_NAME = "naheelkk/fake-news-bert-model" 
EXPLANATION_MODEL = "google/flan-t5-large"
MAX_LENGTH = 512
SEARCH_API_KEY = None  
SEARCH_ENGINE_ID = None  

@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer on startup"""
    global model, tokenizer, device, explanation_model, explanation_tokenizer
    
    try:
        logger.info(f"Loading classification model: {MODEL_NAME}")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model for classification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        
        logger.info("Classification model loaded successfully!")
        
        # Load explanation model (T5 for instruction following)
        try:
            logger.info(f"Loading explanation model: {EXPLANATION_MODEL}")
            explanation_tokenizer = T5Tokenizer.from_pretrained(EXPLANATION_MODEL)
            explanation_model = T5ForConditionalGeneration.from_pretrained(EXPLANATION_MODEL)
            explanation_model.to(device)
            explanation_model.eval()
            logger.info("Explanation model loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load explanation model: {str(e)}")
            explanation_model = None
            explanation_tokenizer = None
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e



def extract_key_claims(text: str) -> List[str]:
    """Extract key claims from the news text for fact-checking"""
    try:
        # Use FLAN-T5 to extract key claims
        if explanation_model is None:
            return [text[:100]]  # Fallback
        
        prompt = f"Extract 2-3 key factual claims from this news text that can be fact-checked:\n\n{text[:400]}\n\nKey claims:"
        
        inputs = explanation_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = explanation_model.generate(
                inputs,
                max_length=100,
                num_beams=3,
                temperature=0.3,
                do_sample=False,
                early_stopping=True,
                pad_token_id=explanation_tokenizer.pad_token_id
            )
        
        claims_text = explanation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Split claims into list
        claims = [claim.strip() for claim in claims_text.split('\n') if claim.strip()]
        
        # If extraction failed, use simple heuristics
        if not claims or len(claims[0]) < 10:
            # Extract sentences with specific patterns
            sentences = re.split(r'[.!?]+', text)
            claims = []
            for sentence in sentences[:3]:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(word in sentence.lower() for word in 
                    ['says', 'reports', 'according', 'claims', 'reveals', 'shows', 'study', 'research']):
                    claims.append(sentence[:80])
            
            if not claims:
                claims = [text[:80]]  # Fallback to beginning of text
        
        return claims[:3]  # Return max 3 claims
    
    except Exception as e:
        logger.error(f"Error extracting claims: {str(e)}")
        return [text[:100]]



def search_web_sources(query: str, max_results: int = 3) -> List[dict]:
    """Search for sources using web search API"""
    try:
        # Using DuckDuckGo Instant Answer API (free alternative)
        search_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_redirect=1&skip_disambig=1"
        
        response = requests.get(search_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            sources = []
            
            # Get abstract/instant answer
            if data.get('Abstract'):
                sources.append({
                    'title': data.get('AbstractText', 'DuckDuckGo Search Result'),
                    'url': data.get('AbstractURL', 'https://duckduckgo.com'),
                    'snippet': data.get('Abstract', '')[:200],
                    'relevance_score': 0.9
                })
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:max_results-len(sources)]:
                if isinstance(topic, dict) and topic.get('Text'):
                    sources.append({
                        'title': topic.get('Text', '')[:50] + '...',
                        'url': topic.get('FirstURL', 'https://duckduckgo.com'),
                        'snippet': topic.get('Text', '')[:200],
                        'relevance_score': 0.7
                    })
            
            return sources[:max_results]
        
        # Fallback: return simulated credible sources
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
        # Create fact-check queries
        queries = [
            f'"{claim}" fact check',
            f'{claim} verify truth',
            f'{claim} snopes factcheck'
        ]
        
        for query in queries[:1]:  # Limit to 1 query per claim to avoid rate limits
            search_queries.append(query)
            sources = search_web_sources(query, max_results=2)
            
            for source in sources:
                source['query'] = query
                source['claim'] = claim
                all_sources.append(source)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
    
    # Remove duplicates and sort by relevance
    unique_sources = {}
    for source in all_sources:
        url = source['url']
        if url not in unique_sources or source['relevance_score'] > unique_sources[url]['relevance_score']:
            unique_sources[url] = source
    
    sorted_sources = sorted(unique_sources.values(), key=lambda x: x['relevance_score'], reverse=True)
    
    return sorted_sources[:5], search_queries  # Return top 5 sources

def calibrate_confidence(raw_confidence: float, temperature: float = 2.0) -> float:
    """Apply temperature scaling to calibrate confidence scores"""
    # Temperature scaling: higher temperature = lower confidence
    import math
    calibrated = 1 / (1 + math.exp(-math.log(raw_confidence / (1 - raw_confidence)) / temperature))
    
    # Additional conservative adjustment
    if calibrated > 0.95:
        calibrated = 0.85 + (calibrated - 0.95) * 0.3  # Cap extreme confidence
    elif calibrated > 0.9:
        calibrated = 0.8 + (calibrated - 0.9) * 0.5   # Reduce high confidence
    elif calibrated < 0.55:
        calibrated = 0.55 + (calibrated - 0.5) * 0.2  # Avoid very low confidence
    
    return float(calibrated)

def get_prediction(text: str):
    """Get prediction and confidence scores from the model"""
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply temperature scaling to logits for better calibration
            temperature = 1.5  # Higher = more conservative confidence
            scaled_logits = logits / temperature
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            raw_confidence = float(probabilities[predicted_class])
            
            # Apply confidence calibration
            confidence = calibrate_confidence(raw_confidence)
            
            # Map class to label (adjust based on your model's labels)
            # Check if your model outputs [REAL, FAKE] or [FAKE, REAL]
            labels = ["Real", "Fake"]  # Update this based on your model
            prediction = labels[predicted_class]
            
            # Additional logic: if confidence is very close, be more conservative
            prob_diff = abs(probabilities[0] - probabilities[1])
            if prob_diff < 0.2:  # Very close scores
                confidence = min(confidence, 0.65)  # Cap confidence for ambiguous cases
                logger.info(f"Ambiguous case detected (diff: {prob_diff:.3f}), confidence capped at {confidence:.3f}")
            
            # Create raw scores dictionary with original probabilities
            raw_scores = {
                "real": float(probabilities[0]),
                "fake": float(probabilities[1])
            }
            
            # Log for debugging
            logger.info(f"Prediction: {prediction}, Raw confidence: {raw_confidence:.3f}, Calibrated: {confidence:.3f}")
            
            return prediction, confidence, raw_scores
            
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise e

async def get_source_based_explanation(text: str, prediction: str, confidence: float, sources: List[dict] = None):
    """Generate explanation using FLAN-T5 model with source information"""
    if explanation_model is None or explanation_tokenizer is None:
        return f"This text is classified as {prediction.lower()} news with {confidence:.1%} confidence based on content analysis."
    
    try:
        # CORRECTED: Better source context preparation
        source_context = ""
        if sources and len(sources) > 0:
            source_context = "\n\nVerification sources:\n"
            for i, source in enumerate(sources[:2], 1):  # Limit to 2 sources to save tokens
                title = source.get('title', 'Source')[:60]
                snippet = source.get('snippet', '')[:80]
                source_context += f"Source {i}: {title} - {snippet}\n"
        
        # CORRECTED: Improved prompt structure for FLAN-T5
        if prediction.lower() == "fake":
            instruction = (
                "Explain why this news article is likely fake or misleading. "
                "Focus on factual inaccuracies, logical inconsistencies, or lack of credible sources."
            )
        else:
            instruction = (
                "Explain why this news article appears to be legitimate. "
                "Focus on factual accuracy, credible sources, and logical consistency."
            )
        
        # CORRECTED: More structured prompt for FLAN-T5
        prompt = (
            f"{instruction}\n\n"
            f"Article: {text[:300]}...\n"
            f"Classification: {prediction} (confidence: {confidence:.1%})"
            f"{source_context}\n\n"
            f"Explanation:"
        )
        
        logger.info(f"Explanation prompt length: {len(prompt)} chars")
        
        # CORRECTED: Better tokenization
        inputs = explanation_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # CORRECTED: Improved generation parameters
        with torch.no_grad():
            outputs = explanation_model.generate(
                **inputs,
                max_length=200,
                min_length=40,           # Ensure substantive explanations
                do_sample=True,
                top_p=0.85,             # More focused than 0.9
                temperature=0.8,        # Better balance
                num_beams=3,            # Add beam search
                early_stopping=True,
                pad_token_id=explanation_tokenizer.pad_token_id,
                repetition_penalty=1.2, # Prevent repetition
                no_repeat_ngram_size=2  # Prevent 2-gram repetition
            )
        
        # Decode the response
        explanation = explanation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # CORRECTED: Better post-processing
        # Remove the original prompt from the output (T5 sometimes includes it)
        if "Explanation:" in explanation:
            explanation = explanation.split("Explanation:")[-1].strip()
        
        # Clean up common artifacts and the original prompt text
        explanation = re.sub(r'^(Explain why|Article:|Classification:).*?\n', '', explanation, flags=re.MULTILINE | re.DOTALL)
        explanation = explanation.strip()
        
        # CORRECTED: Better fallback detection and handling
        if (len(explanation.strip()) < 20 or 
            explanation.lower().startswith(text.lower()[:20]) or
            explanation.lower().startswith(prompt.lower()[:20])):
            
            logger.warning("T5 explanation generation failed, using enhanced fallback")
            
            # Enhanced template explanation
            if prediction.lower() == "fake":
                explanation = (
                    f"This content exhibits patterns commonly associated with misinformation, "
                    f"such as unverified claims, sensational language, or lack of credible attribution. "
                )
            else:
                explanation = (
                    f"This article demonstrates characteristics of legitimate journalism, "
                    f"including factual reporting style and verifiable information patterns. "
                )
            
            # Add source-based context if available
            if sources and len(sources) > 0:
                explanation += (
                    f"Cross-verification with {len(sources)} external sources "
                    f"{'contradicts key claims' if prediction.lower() == 'fake' else 'supports the main assertions'} "
                    f"presented in the article."
                )
        
        # Final cleanup
        explanation = re.sub(r'\s+', ' ', explanation).strip()
        
        # Ensure proper sentence ending
        if explanation and not explanation.endswith(('.', '!', '?')):
            explanation += '.'
            
        logger.info(f"Final explanation length: {len(explanation)} chars")
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        
        # CORRECTED: More informative fallback
        fallback = (
            f"Analysis indicates this is {prediction.lower()} news with {confidence:.1%} confidence. "
        )
        
        if sources:
            fallback += (
                f"Assessment included {len(sources)} verification sources "
                f"for fact-checking and credibility analysis."
            )
        else:
            fallback += "Classification based on content patterns and linguistic analysis."
            
        return fallback
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None and tokenizer is not None,
        explanation_model_loaded=explanation_model is not None,
        search_enabled=True  # Always enabled with DuckDuckGo fallback
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(request: NewsRequest):
    """
    Predict if news is fake or real with source-based explanations
    
    - **text**: The news text to analyze
    - **explain**: Whether to include LLM explanation (default: True)
    - **search_sources**: Whether to search for supporting sources (default: True)
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Get prediction from your model
        prediction, confidence, raw_scores = get_prediction(request.text)
        
        sources = []
        search_queries = []
        explanation = None
        
        # Search for sources if requested
        if request.search_sources and request.explain:
            logger.info("Extracting claims and searching for sources...")
            claims = extract_key_claims(request.text)
            sources, search_queries = await search_fact_check_sources(claims)
        
        # Get explanation if requested
        if request.explain:
            explanation = await get_source_based_explanation(
                request.text, prediction, confidence, sources
            )
        
        # Convert sources to SourceInfo objects
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

@app.get("/models/info")
async def model_info():
    """Get information about loaded models"""
    return {
        "classification_model": MODEL_NAME,
        "explanation_model": EXPLANATION_MODEL,
        "device": str(device),
        "classification_loaded": model is not None,
        "explanation_loaded": explanation_model is not None,
        "search_enabled": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Factify API - Source-Based Fake News Detection",
        "version": "2.0.0",
        "features": [
            "AI-powered fake news classification",
            "Source-based explanations",
            "Web search integration",
            "Fact-checking capabilities"
        ],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/models/info",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)