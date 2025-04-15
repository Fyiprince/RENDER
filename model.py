import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Dummy dataset (fake news vs real news)
data = {
    'text': ['Breaking: Election results rigged', 'New study reveals benefits of meditation', 
             'Rumor: Popular celebrity arrested', 'Research shows exercise improves health'],
    'label': [1, 0, 1, 0]  # 1 = fake news, 0 = real news
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split the data
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline with TF-IDF and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'fake_news_model.pkl')

print("Model trained and saved successfully!")
