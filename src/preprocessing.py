
import nltk
import os

# Define a writable directory for NLTK data within the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk_data_dir = os.path.join(project_root, "nltk_data")

# Ensure the directory exists and download the necessary NLTK data to it.
# NLTK's downloader is smart and will not re-download if the data is already present.
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

# Add the custom path to NLTK's data path so it can find the downloaded data.
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Now that the data is guaranteed to be available, we can import the modules that use it.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """
    文本預處理
    
    Args:
        text (str): 輸入文本
        
    Returns:
        str: 處理後的文本
    """
    # 轉換為小寫
    text = text.lower()
    
    # 標記化和移除停用詞
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
    
    # 重新組合文本
    return ' '.join(tokens)
