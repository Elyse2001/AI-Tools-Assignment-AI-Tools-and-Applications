import spacy
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

reviews = [
    "I love my new Apple iPhone, the camera is amazing!",
    "The Samsung TV I bought is terrible, very bad sound quality."
]

for review in reviews:
    doc = nlp(review)
    print("\nReview:", review)

    # Named Entity Recognition
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

    # Simple sentiment analysis
    sentiment = TextBlob(review).sentiment.polarity
    print("Sentiment:", "Positive" if sentiment > 0 else "Negative")
