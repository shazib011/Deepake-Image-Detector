from datasets import load_dataset
from pathlib import Path
from PIL import Image
import io

# Load dataset from Hugging Face
dataset = load_dataset("saakshigupta/deepfake-detection-dataset-v3")

# Output base directory
BASE_DIR = Path("data/processed")
BASE_DIR.mkdir(parents=True, exist_ok=True)

def save_split(split_name, split_data):
    for i, sample in enumerate(split_data):
        label = sample["label"]  # 0 = real, 1 = fake
        img = sample["image"]

        if label == 0:
            out_dir = BASE_DIR / split_name / "real"
        else:
            out_dir = BASE_DIR / split_name / "fake"

        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = out_dir / f"{split_name}_{i}.jpg"
        img.save(img_path)

    print(f"{split_name} saved successfully")

# Save train data
save_split("train", dataset["train"])

# Save test data
save_split("test", dataset["test"])
