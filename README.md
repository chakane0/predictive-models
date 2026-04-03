## Exploring usecases of predictive models

### Motivation
My initial motiviation involves a project i had done in 2018 when first learning about artifical intelligence. For a finance project, i wanted to use qualatative data (speech/text) to create a sentiment analyis chart to determne wether or not a certain earnings call went well. The aim was to reduce the time to listen or watch the entire call, reducing an hour listening to about 20 minutes of processing. Now that it is currently 2026, computers have gotten much faster and we a wide range of A.I models available to us, i want to attempt this idea again with these new tools.

##### huggingface.py
The first pipeline to input audio, use OpenAIs Whisper to get text, then using FinBert to output sentiment.

Whisper converts audio to pieces of text. Each piece is split into time-stamped segments based on natural speech pauses. The downside to using this is that it can be quite slow, maybe there is a much faster API. Its important to know Whisper segments on pauses, not sentences. This has been merged to create segments as sentences and not pauses for improved classification

FinBert takes each time stamped segements and classifies it as positive, negative, or neutral. The downside to this is that its trained on written financial news, natural conversation about financial news mey not be the best option.


#### Notes

###### Confidence Scoring
This is defined as a probability, expressed as a percentage which tells us how confident a model is. This can pertain to prediction, classification or data extraction. This is one of the key markers for reliability for ML model predictions. 

In the context of how were using this in `huggingface.py` we can see it as such. When we run a segment through FinBert, it outputs 3 raw numbers called logits which map to a negative, positive, or neutral number. Right now i just go with the highest number using `argmax()`.

For me to turn this into confidence scroes i need to apply a `softmax` function which converts the logits into probabilities which sum to 1.

logits = [-1.2, 2.4, 0.3] --> [neg, pos, neu]
softmax = [0.04, 0.88, 0.08] --> we can say the model is 88% confident about this segment


At this point we will have room to configure the threshold for positive and negative values. We can do something like, ignore anything below 70%. 

We'll have cateogires for confidence: (1) high confidence correct answers, (2) high confidence wrong answers, (3) low confidence correct answers, (4) low confidence wrong answers. This is a more complex problem to solve but it can be done through fine tuning the model