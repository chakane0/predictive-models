import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import whisper
import torch
import torch.nn.functional as F


# connect to HF
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# create finbert model
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def transcribe_audio(file_path):
    model = whisper.load_model("base")  # move outside if reused
    result = model.transcribe(file_path)
    return {
        "text": result["text"],
        "segments": result["segments"]
    }


AUDIO_FILE = "ExamplePipelines/goldmanstanley_q4_earnings_call.mp3"
output = transcribe_audio(AUDIO_FILE)
LABELS = ["positive", "negative", "neutral"]

def analyze_sentiment(output):
    results = []
    merged = merge_segments(output["segments"])
    for seg in merged:
        inputs = tokenizer(seg["text"], return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        label = LABELS[predicted.item()]
        score = confidence.item()
        if score < 0.70:
            label = "uncertain"
        results.append({
            "start": seg["start"], 
            "end": seg["end"], 
            "text": seg["text"], 
            "confidence": score,
            "sentiment": label
        })
    return results

def merge_segments(segments):
    merged = []
    current = {"text": "", "start": None, "end": None}

    for seg in segments:
        if current["start"] is None:
            current["start"] = seg["start"]
        current["text"] += " " + seg["text"].strip()
        current["end"] = seg["end"]

        if current["text"].strip().endswith((".", "?", "!")):
            merged.append(current)
            current = {"text": "", "start": None, "end": None}

    if current["text"].strip():
        merged.append(current)
    
    return merged

def calculate_weighted_sentiment_score(results):
    score = 0
    for r in results:
        if r['sentiment'] == 'positive':
            score += r['confidence']
        elif r['sentiment'] == 'negative':
            score -= r['confidence']
    
    overall_sentiment = score / len(results)
    print(f"\n\nOVERALL SENTIMENT SCORE: {overall_sentiment:.3f}")
    print(f"Range: -1 (very negative) to +1 (very positive)")
    

results = analyze_sentiment(output)
calculate_weighted_sentiment_score(results)


# for r in results:
#     print(f"[{r['start']:.1f}s - {r['end']:.1f}s] {r['sentiment']} ({r['confidence']:.0%}): {r['text']}")

# SCORE_MAP = {"positive": 1, "neutral": 0, "negative": -1, "uncertain": 0}
# COLOR_MAP = {"positive": "green", "neutral": "gray", "negative": "red", "uncertain": "lightblue"}


# times = [r["start"] for r in results]
# scores = [SCORE_MAP[r["sentiment"]] for r in results]
# colors = [COLOR_MAP[r["sentiment"]] for r in results]

# plt.figure(figsize=(14, 4))
# plt.bar(times, scores, color=colors, width=2)
# plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Sentiment")
# plt.title("Earnings Call Sentiment Over Time")
# plt.yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
# plt.tight_layout()
# plt.savefig("sentiment_chart.png")
# plt.show()
