import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_RATIO = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "archive", "creditcard.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ["#2ecc71", "#e74c3c"]
LABELS = ["Normal (0)", "Fraud (1)"]
BAR_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
ROC_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
ROC_LINE_STYLES = ["-", "--", "-.", ":", "-"]

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11
