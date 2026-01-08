# TASK 3: NER & Sentiment with spaCy
import spacy

# Load English pipeline
nlp = spacy.load("en_core_web_sm")

# Sample Review Data
reviews = [
    "I bought a Samsung Galaxy S21 from Amazon and it works great!",
    "The battery life on this iPhone 13 is terrible, very disappointed."
]

print("\n--- Named Entity Recognition (NER) ---")
for text in reviews:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            print(f"Entity: {ent.text} | Label: {ent.label_}")

print("\n--- Rule-Based Sentiment Analysis ---")
# Simple rule-based logic (spaCy core doesn't do sentiment out-of-box)
positive_words = ["great", "good", "excellent", "love"]
negative_words = ["terrible", "bad", "disappointed", "poor"]

for text in reviews:
    doc = nlp(text.lower())
    score = 0
    for token in doc:
        if token.text in positive_words: score += 1
        if token.text in negative_words: score -= 1
    
    sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    print(f"Review: '{text}' -> Sentiment: {sentiment}")