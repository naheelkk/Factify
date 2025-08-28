import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction(text, explain=True):
    """Test prediction endpoint"""
    data = {
        "text": text,
        "explain": explain
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Text: {result['text'][:100]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Raw Scores: {result['raw_scores']}")
        if result['explanation']:
            print(f"Explanation: {result['explanation']}")
        print("-" * 50)
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Test health
    test_health()
    
    # Test sample news articles
    test_cases = [
        {
            "text": "Breaking: Local scientists discover new method to purify water using advanced filtration technology. The research was published in the Journal of Environmental Science and shows promising results for developing countries.",
            "explain": True
        },
        {
            "text": "SHOCKING: Aliens have landed in Area 51 and the government is hiding it from us! Secret sources reveal that extraterrestrial beings are working with military officials. You won't believe what happens next!",
            "explain": True
        },
        {
            "text": "The stock market closed higher today with the S&P 500 gaining 1.2%. Technology stocks led the rally as investors showed confidence in the sector's growth prospects.",
            "explain": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        test_prediction(**test_case)
        print()