
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define a writable directory for NLTK data within the project
# This is crucial for environments like Streamlit Cloud where the default path might not be writable.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk_data_dir = os.path.join(project_root, "nltk_data")

# Add the custom path to NLTK's data path
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

def download_nltk_data():
    """
    Downloads the necessary NLTK data to a local, project-specific directory.
    """
    # Ensure the target directory exists
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    try:
        # Check if the data is available in the custom path
        nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)
    try:
        # Check if the data is available in the custom path
        nltk.data.find('corpora/stopwords', paths=[nltk_data_dir])
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_dir)

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

# Call the download function once when this module is imported.
download_nltk_data()
