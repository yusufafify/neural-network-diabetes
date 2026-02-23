import os
import shutil
from dotenv import load_dotenv
import kagglehub

# Load environment variables from .env file
load_dotenv()

# Set Kaggle credentials from .env
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN", "")

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)

# Copy the CSV into the dataset/ folder for easy access
dataset_dir = os.path.dirname(os.path.abspath(__file__))
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join(dataset_dir, file)
    shutil.copy2(src, dst)
    print(f"Copied {file} -> {dst}")
