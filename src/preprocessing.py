
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def download_nltk_data():
    """
    Downloads the necessary NLTK data (punkt, stopwords) if not already downloaded.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')

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
