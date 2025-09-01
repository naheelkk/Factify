import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load your model
MODEL_PATH = "/content/drive/MyDrive/Data-Single/model/"  # Update this path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Check model configuration
print("Model config:")
print(f"Number of labels: {model.config.num_labels}")
if hasattr(model.config, 'id2label'):
    print(f"Label mapping: {model.config.id2label}")

# Test cases
test_cases = [
    "Scientists have discovered that vaccines contain microchips to control people's minds",  # Should be Fake
    "The World Health Organization announced new guidelines for COVID-19 vaccination schedules",  # Should be Real
]

def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = probabilities.cpu().numpy()[0]
        
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        # Based on training: 0 = Fake, 1 = Real
        labels = ["Fake", "Real"]
        prediction = labels[predicted_class]
        
        return prediction, confidence, probabilities

# Test predictions
for i, text in enumerate(test_cases):
    prediction, confidence, probs = predict_text(text)
    print(f"\nTest {i+1}: {text[:50]}...")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: Fake={probs[0]:.4f}, Real={probs[1]:.4f}")