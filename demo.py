from datasets import load_dataset
import pandas as pd

# Download the full emotion dataset
print("Downloading emotion dataset...")
dataset = load_dataset("dair-ai/emotion")

# Access all splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")
print(f"Test samples: {len(test_data)}")

# Optional: Convert to pandas DataFrame for easier exploration
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(validation_data)
test_df = pd.DataFrame(test_data)

# Optional: Save to CSV files
train_df.to_csv("emotion_train.csv", index=False)
val_df.to_csv("emotion_validation.csv", index=False)
test_df.to_csv("emotion_test.csv", index=False)

print("Dataset downloaded and saved to CSV files!")

# Display emotion labels mapping
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

print("\nEmotion Labels Mapping:")
for label_id, emotion in emotion_labels.items():
    print(f"{label_id}: {emotion}")
