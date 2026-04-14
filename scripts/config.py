import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_RATIO = 0.5

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"

def get_data_path() -> Path:
    """Download the dataset via kagglehub if not present and return its path."""
    print("Fetching dataset via kagglehub...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    return Path(path) / "creditcard.csv"

COLORS = ["#2ecc71", "#e74c3c"]
LABELS = ["Normal (0)", "Fraud (1)"]
BAR_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
ROC_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
ROC_LINE_STYLES = ["-", "--", "-.", ":", "-"]

def setup_environment():
    """Initialize output directories and plotting configurations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    warnings.filterwarnings("ignore")
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.size"] = 11
