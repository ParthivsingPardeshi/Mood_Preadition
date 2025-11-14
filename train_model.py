"""
Train emotion detection model using the provided datasets
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
import sys

# Add utils to path
sys.path.append('utils')
from preprocessing import TextPreprocessor

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

# Emotion label mapping
EMOTION_LABELS = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

print("="*60)
print("EMOTION DETECTION MODEL TRAINING")
print("="*60)

# Load datasets
print("\n1. Loading datasets...")
train_df = pd.read_csv('data/emotion_test.csv')
val_df = pd.read_csv('data/emotion_validation.csv')

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Display emotion distribution
print("\nEmotion distribution in training data:")
for label, emotion in EMOTION_LABELS.items():
    count = (train_df['label'] == label).sum()
    print(f"  {emotion}: {count} samples")

# Initialize preprocessor
print("\n2. Initializing text preprocessor...")
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    lemmatize=True,
    remove_punctuation=True,
    lowercase=True
)

# Preprocess text data
print("\n3. Preprocessing text data...")
print("   (This may take a few minutes...)")

train_df['clean_text'] = train_df['text'].apply(lambda x: preprocessor.clean_text(x))
val_df['clean_text'] = val_df['text'].apply(lambda x: preprocessor.clean_text(x))

print("   Preprocessing complete!")
print(f"   Sample cleaned text: {train_df['clean_text'].iloc[0][:100]}...")

# Prepare features
print("\n4. Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X_train = vectorizer.fit_transform(train_df['clean_text'])
X_val = vectorizer.transform(val_df['clean_text'])
y_train = train_df['label']
y_val = val_df['label']

print(f"   Feature matrix shape: {X_train.shape}")
print(f"   Number of features: {X_train.shape[1]}")

# Train model
print("\n5. Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0,
    solver='lbfgs',
    multi_class='multinomial'
)

model.fit(X_train, y_train)
print("   Model training complete!")

# Evaluate on training data
print("\n6. Evaluating model performance...")
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"\n   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Evaluate on validation data
val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Detailed classification report
print("\n7. Detailed Classification Report:")
print("="*60)
target_names = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]
print(classification_report(y_val, val_pred, target_names=target_names))

# Confusion matrix
print("\n8. Confusion Matrix:")
cm = confusion_matrix(y_val, val_pred)
print("\n   Predicted ->")
print("   Actual ↓")
print("        ", "  ".join([f"{e[:3]}" for e in target_names]))
for i, row in enumerate(cm):
    print(f"   {target_names[i][:8]:<8}", "  ".join([f"{val:>3}" for val in row]))

# Save model and components
print("\n9. Saving model and components...")

# Save model
with open('models/emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved to models/emotion_model.pkl")

# Save vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("   ✓ Vectorizer saved to models/vectorizer.pkl")

# Save preprocessor
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("   ✓ Preprocessor saved to models/preprocessor.pkl")

# Save label mapping
with open('models/label_mapping.pkl', 'wb') as f:
    pickle.dump(EMOTION_LABELS, f)
print("   ✓ Label mapping saved to models/label_mapping.pkl")

# Test with sample predictions
print("\n10. Testing with sample predictions:")
print("="*60)

test_samples = [
    "I am so happy and excited today!",
    "This is terrible, I hate it!",
    "I'm scared and worried about tomorrow",
    "I love you so much!",
    "Oh wow, I didn't expect that!",
    "I feel so sad and disappointed"
]

for sample in test_samples:
    clean_sample = preprocessor.clean_text(sample)
    sample_vec = vectorizer.transform([clean_sample])
    prediction = model.predict(sample_vec)[0]
    probabilities = model.predict_proba(sample_vec)[0]
    
    print(f"\nText: {sample}")
    print(f"Predicted Emotion: {EMOTION_LABELS[prediction]}")
    print(f"Confidence: {probabilities[prediction]*100:.2f}%")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print("\nYou can now run the Streamlit app with:")
print("  streamlit run app.py")
print("="*60)
