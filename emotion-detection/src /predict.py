import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load labels + thresholds
with open(f"{MODEL_PATH}/labels.json") as f:
    labels = json.load(f)

with open(f"{MODEL_PATH}/thresholds.json") as f:
    thresholds = json.load(f)


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    predictions = [
        labels[i] for i in range(len(labels))
        if probs[i] > thresholds[labels[i]]
    ]

    return predictions, probs


# Quick test
if __name__ == "__main__":
    text = "I feel completely alone and worthless."
    preds, probs = predict(text)

    print("Text:", text)
    print("Predictions:", preds)
