import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import whisper
import torch

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


AUDIO_FILE = "ExampleModels/goldmanstanley_q4_earnings_call.mp3"
output = transcribe_audio(AUDIO_FILE)

LABELS = ["positive", "negative", "neutral"]

def analyze_sentiment(output):
    results = []
    for seg in output["segments"]:
        inputs = tokenizer(seg["text"], return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        label = LABELS[logits.argmax().item()]
        results.append({"start": seg["start"], "end": seg["end"], "text": seg["text"], "sentiment": label})
    return results

results = analyze_sentiment(output)
for r in results:
    print(f"[{r['start']:.1f}s - {r['end']:.1f}s] {r['sentiment']}: {r['text']}")

SCORE_MAP = {"positive": 1, "neutral": 0, "negative": -1}
COLOR_MAP = {"positive": "green", "neutral": "gray", "negative": "red"}

times = [r["start"] for r in results]
scores = [SCORE_MAP[r["sentiment"]] for r in results]
colors = [COLOR_MAP[r["sentiment"]] for r in results]

plt.figure(figsize=(14, 4))
plt.bar(times, scores, color=colors, width=2)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Time (seconds)")
plt.ylabel("Sentiment")
plt.title("Earnings Call Sentiment Over Time")
plt.yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
plt.tight_layout()
plt.savefig("sentiment_chart.png")
plt.show()
