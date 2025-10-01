import streamlit as st
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Factify - AI Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
    }
    
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    
    .fake-news {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
        border-left: 5px solid #ff4757;
    }
    
    .real-news {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #2ed573;
    }
    
    .confidence-high { color: #2ed573; font-weight: bold; }
    .confidence-medium { color: #ffa502; font-weight: bold; }
    .confidence-low { color: #ff6b6b; font-weight: bold; }
    .confidence-very-low { color: #ee5a52; font-weight: bold; }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    
    .sample-news {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #17a2b8;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-news:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Helper functions
def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.85:
        return "confidence-high"
    elif confidence >= 0.75:
        return "confidence-medium" 
    elif confidence >= 0.65:
        return "confidence-low"
    else:
        return "confidence-very-low"

def create_confidence_gauge(confidence, prediction):
    """Create a confidence gauge chart"""
    color = "#2ed573" if prediction == "Real" else "#ff4757"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_scores_chart(raw_scores):
    """Create a bar chart for raw scores"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(raw_scores.keys()),
            y=list(raw_scores.values()),
            marker_color=['#2ed573', '#ff4757'],
            text=[f"{v:.1%}" for v in raw_scores.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Classification Scores",
        xaxis_title="Category",
        yaxis_title="Probability",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def predict_news(text, explain=True):
    """Make prediction request to the API"""
    try:
        payload = {"text": text, "explain": explain}
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=45)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection Error: {str(e)}"

# Sample news articles for testing
SAMPLE_NEWS = [
    {
        "title": "💬 Real News Sample - Technology",
        "text": "Researchers at Stanford University have developed a new machine learning algorithm that can predict protein structures with 90% accuracy. The study, published in the journal Science, could accelerate drug discovery processes. The research team spent three years developing the algorithm using existing protein databases.",
        "type": "real"
    },
    {
        "title": "🚨 Fake News Sample - Conspiracy", 
        "text": "SHOCKING REVELATION: Government scientists have been secretly controlling weather patterns for decades! Anonymous whistleblower reveals that hurricanes are artificially created using hidden technology. This bombshell discovery will change everything! Mainstream media won't report this truth!",
        "type": "fake"
    },
    {
        "title": "📈 Real News Sample - Business",
        "text": "Apple Inc. reported quarterly earnings that exceeded analyst expectations, with revenue reaching $89.5 billion in Q2 2024. The company's iPhone sales showed a 5% increase compared to the previous year. CEO Tim Cook attributed the growth to strong international demand and new product features.",
        "type": "real"
    },
    {
        "title": "🔬 Real News Sample - Health",
        "text": "A clinical trial published in The Lancet shows that a new diabetes medication reduced blood sugar levels by an average of 18% over six months. The study involved 2,400 patients across 15 countries. Researchers noted that further long-term studies are needed to assess potential side effects.",
        "type": "real"
    }
]

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Factify</h1>
        <p>AI-Powered Fake News Detection</p>
        <p><em>Analyze news articles with advanced machine learning</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API status
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ Backend API is not running. Please start the FastAPI server first.")
        st.code("python main.py", language="bash")
        st.stop()
    else:
        # Show API status in sidebar
        with st.sidebar:
            st.success("✅ API Connected")
            if health_data:
                st.json(health_data)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter News Article")
        
        # Text input
        news_text = st.text_area(
            "Paste your news article here:",
            placeholder="Enter the news article you want to analyze...",
            height=200,
            help="Paste any news article text to check if it's real or fake news."
        )
        
        # Options
        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            include_explanation = st.checkbox("Include AI Explanation", value=True)
        with col_opt2:
            show_charts = st.checkbox("Show Charts", value=True)
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
        
        # Sample news section
        st.subheader("💡 Try Sample Articles")
        
        for i, sample in enumerate(SAMPLE_NEWS):
            with st.expander(sample["title"]):
                st.write(sample["text"])
                if st.button(f"Analyze This Sample", key=f"sample_{i}"):
                    news_text = sample["text"]
                    st.rerun()
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        # Results container
        results_container = st.container()
        
        if analyze_button and news_text.strip():
            with st.spinner("🤖 Analyzing article..."):
                success, result = predict_news(news_text, include_explanation)
                
                if success:
                    prediction = result["prediction"]
                    confidence = result["confidence_score"]
                    raw_scores = result["raw_scores"]
                    explanation = result.get("explanation")
                    
                    # Prediction card
                    card_class = "fake-news" if prediction == "Fake" else "real-news"
                    confidence_class = get_confidence_class(confidence)
                    
                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h3>📊 Prediction: {prediction} News</h3>
                        <p class="{confidence_class}">Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Charts
                    if show_charts:
                        # Confidence gauge
                        fig_gauge = create_confidence_gauge(confidence, prediction)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Scores chart
                        fig_scores = create_scores_chart(raw_scores)
                        st.plotly_chart(fig_scores, use_container_width=True)
                    
                    # Explanation
                    if include_explanation and explanation:
                        st.subheader("🧠 AI Analysis")
                        st.info(explanation)
                    
                    # Raw scores
                    with st.expander("🔢 Raw Scores"):
                        st.json(raw_scores)
                    
                    # Analysis metadata
                    st.caption(f"Analysis completed at {datetime.now().strftime('%H:%M:%S')}")
                    
                else:
                    st.error(f"❌ Error: {result}")
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit, FastAPI, and Hugging Face Transformers</p>
        <p><small>This tool uses AI to detect potential fake news. Always verify information from multiple sources.</small></p>
    </div>
    """, unsafe_allow_html=True)

# History tracking (optional)
def show_history():
    """Show analysis history in sidebar"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if st.session_state.analysis_history:
        st.sidebar.subheader("📜 Recent Analyses")
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.sidebar.expander(f"{analysis['prediction']} - {analysis['confidence']:.1%}"):
                st.write(analysis['text'][:100] + "...")
                st.caption(analysis['timestamp'])

if __name__ == "__main__":
    main()