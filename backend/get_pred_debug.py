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
            
            temperature = 1.5
            scaled_logits = logits / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]
            
            predicted_class = np.argmax(probabilities)
            raw_confidence = float(probabilities[predicted_class])
            
            # Simple confidence calibration
            confidence = min(raw_confidence * 0.9, 0.95)  # Conservative adjustment
            
            # changes for label flipping
            if hasattr(model.config, 'id2label') and model.config.id2label:
                labels = [model.config.id2label[0], model.config.id2label[1]]
            else:
                labels = ["Fake", "Real"]
            
            prediction = labels[predicted_class]
        
            # labels = model.config.id2label
            # prediction = labels[predicted_class]

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
