import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDebugger:
    """Debug and fix the fake news detection model"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Print model configuration
            logger.info(f"Model config: {self.model.config}")
            logger.info(f"Number of labels: {self.model.config.num_labels}")
            
            if hasattr(self.model.config, 'id2label'):
                logger.info(f"ID to Label mapping: {self.model.config.id2label}")
            if hasattr(self.model.config, 'label2id'):
                logger.info(f"Label to ID mapping: {self.model.config.label2id}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def predict_single(self, text, max_length=256, debug=True):
        """Make a single prediction with detailed debugging"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if debug:
                logger.info(f"Input text: {text[:100]}...")
                logger.info(f"Tokenized length: {inputs['input_ids'].shape}")
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                if debug:
                    logger.info(f"Raw logits: {logits}")
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                probabilities = probabilities.cpu().numpy()[0]
                
                if debug:
                    logger.info(f"Probabilities: {probabilities}")
                
                # Get predicted class
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                
                # Map to labels
                if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                    label_mapping = self.model.config.id2label
                    prediction = label_mapping[predicted_class]
                else:
                    # Default mapping based on training data
                    labels = ["FAKE", "REAL"]  # or ["Fake", "Real"]
                    prediction = labels[predicted_class]
                
                if debug:
                    logger.info(f"Predicted class: {predicted_class}")
                    logger.info(f"Prediction: {prediction}")
                    logger.info(f"Confidence: {confidence:.4f}")
                
                return {
                    "prediction": prediction,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": {
                        "class_0": float(probabilities[0]),
                        "class_1": float(probabilities[1])
                    },
                    "raw_logits": logits.cpu().numpy()[0].tolist()
                }
                
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise e
    
    def test_known_samples(self):
        """Test with known fake and real samples"""
        
        # Test cases with expected labels
        test_cases = [
            {
                "text": "Scientists have discovered that vaccines contain microchips designed to control people's minds and track their movements through 5G networks",
                "expected": "FAKE",
                "category": "conspiracy"
            },
            {
                "text": "Breaking: Aliens have landed in Times Square and are demanding to speak to world leaders immediately",
                "expected": "FAKE", 
                "category": "sensational"
            },
            {
                "text": "Apple Inc. reported quarterly earnings that exceeded analyst expectations, with revenue reaching $94.8 billion in Q2 2024",
                "expected": "REAL",
                "category": "business"
            },
            {
                "text": "The World Health Organization announced new guidelines for COVID-19 vaccination schedules based on recent research findings",
                "expected": "REAL",
                "category": "health"
            },
            {
                "text": "Researchers at Stanford University published a study in Nature showing improved methods for renewable energy storage",
                "expected": "REAL",
                "category": "science"
            }
        ]
        
        results = []
        correct_predictions = 0
        
        logger.info("Testing model with known samples...")
        logger.info("="*60)
        
        for i, case in enumerate(test_cases, 1):
            result = self.predict_single(case["text"], debug=False)
            
            # Normalize predictions for comparison
            predicted = result["prediction"].upper()
            expected = case["expected"].upper()
            
            is_correct = predicted == expected
            if is_correct:
                correct_predictions += 1
            
            logger.info(f"Test {i}: {case['category'].upper()}")
            logger.info(f"Expected: {expected}")
            logger.info(f"Predicted: {predicted}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info(f"Correct: {'✓' if is_correct else '✗'}")
            logger.info(f"Probabilities: {result['probabilities']}")
            logger.info("-" * 40)
            
            results.append({
                **case,
                **result,
                "is_correct": is_correct
            })
        
        accuracy = correct_predictions / len(test_cases)
        logger.info(f"Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_cases)})")
        
        return results, accuracy
    
    def diagnose_label_mapping_issue(self):
        """Diagnose potential label mapping issues"""
        logger.info("Diagnosing label mapping issues...")
        logger.info("="*50)
        
        # Check if labels are swapped
        fake_sample = "This is obviously fake news with wild conspiracy theories"
        real_sample = "The stock market closed higher today with tech stocks leading gains"
        
        fake_result = self.predict_single(fake_sample, debug=False)
        real_result = self.predict_single(real_sample, debug=False)
        
        logger.info("DIAGNOSIS RESULTS:")
        logger.info(f"Fake sample predicted as: {fake_result['prediction']} (class {fake_result['predicted_class']})")
        logger.info(f"Real sample predicted as: {real_result['prediction']} (class {real_result['predicted_class']})")
        
        # Check if labels might be swapped
        if (fake_result['prediction'].upper() == 'REAL' and 
            real_result['prediction'].upper() == 'FAKE'):
            logger.warning("🚨 LABELS APPEAR TO BE SWAPPED! 🚨")
            logger.warning("The model is predicting opposite of expected results")
            return True
        
        return False
    
    def fix_label_mapping(self):
        """Fix label mapping in model config"""
        logger.info("Attempting to fix label mapping...")
        
        # Update the model's label mapping
        if hasattr(self.model.config, 'id2label'):
            original_mapping = self.model.config.id2label.copy()
            logger.info(f"Original mapping: {original_mapping}")
            
            # Swap the labels
            self.model.config.id2label = {
                0: "REAL" if original_mapping.get(0) == "FAKE" else "FAKE",
                1: "FAKE" if original_mapping.get(1) == "REAL" else "REAL"
            }
            
            # Also update label2id if it exists
            if hasattr(self.model.config, 'label2id'):
                self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}
            
            logger.info(f"New mapping: {self.model.config.id2label}")
            
            return True
        else:
            logger.warning("No id2label mapping found in config")
            # Create a new mapping
            self.model.config.id2label = {0: "FAKE", 1: "REAL"}
            self.model.config.label2id = {"FAKE": 0, "REAL": 1}
            logger.info("Created new label mapping")
            return True
    
    def save_corrected_model(self, save_path):
        """Save the model with corrected label mapping"""
        try:
            logger.info(f"Saving corrected model to: {save_path}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info("Model saved successfully!")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

def main():
    """Main debugging and fixing function"""
    
    # UPDATE THIS PATH TO YOUR MODEL LOCATION
    model_path = "naheelkk/fake-news-bert-model"  # or your HuggingFace model name
    
    debugger = ModelDebugger(model_path)
    
    # Step 1: Test current model performance
    logger.info("STEP 1: Testing current model performance")
    results, accuracy = debugger.test_known_samples()
    
    # Step 2: Diagnose label mapping issues
    logger.info("\nSTEP 2: Diagnosing label mapping issues")
    labels_swapped = debugger.diagnose_label_mapping_issue()
    
    # Step 3: Fix if needed
    if labels_swapped or accuracy < 0.6:
        logger.info("\nSTEP 3: Fixing label mapping")
        debugger.fix_label_mapping()
        
        # Test again after fix
        logger.info("\nSTEP 4: Testing after fix")
        results_after_fix, accuracy_after_fix = debugger.test_known_samples()
        
        if accuracy_after_fix > accuracy:
            logger.info(f"✅ Improvement detected! Accuracy: {accuracy:.2%} → {accuracy_after_fix:.2%}")
            
            # Save corrected model
            corrected_model_path = model_path.rstrip('/') + "_corrected/"
            if debugger.save_corrected_model(corrected_model_path):
                logger.info(f"✅ Corrected model saved to: {corrected_model_path}")
                logger.info("Update your API to use this corrected model!")
            
        else:
            logger.warning("No improvement after label fix. Issue might be deeper.")
    else:
        logger.info("✅ Model appears to be working correctly!")
    
    return debugger

if __name__ == "__main__":
    # Run the debugging
    debugger = main()
    
    # Test specific case from your screenshot
    print("\n" + "="*60)
    print("TESTING YOUR SPECIFIC CASE:")
    print("="*60)
    
    apple_text = """Apple Inc. reported quarterly earnings that exceeded analyst expectations, with revenue reaching $94.8 billion in Q2 2024. The company's iPhone sales showed a 5% increase compared to the previous year. CEO Tim Cook attributed the growth to strong international demand and new product features."""
    
    result = debugger.predict_single(apple_text)
    print(f"Your Apple news sample:")
    print(f"Predicted: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Should be: REAL")
    print(f"Correct: {'✅' if result['prediction'].upper() == 'REAL' else '❌'}")