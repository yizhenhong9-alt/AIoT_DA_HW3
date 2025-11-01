"""
垃圾郵件分類模型訓練腳本
"""

import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import nltk
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

# 下載必要的 NLTK 數據
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SpamClassifier:
    def __init__(self):
        """初始化分類器"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        self.model = SVC(
            kernel='linear',
            probability=True
        )
        
    def preprocess_text(self, text):
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
    
    def fit(self, X, y):
        """
        訓練模型
        
        Args:
            X (list): 文本列表
            y (list): 標籤列表
        """
        # 預處理文本
        X_clean = [self.preprocess_text(text) for text in X]
        
        # 特徵提取
        X_features = self.vectorizer.fit_transform(X_clean)
        
        # 訓練模型
        self.model.fit(X_features, y)
        
    def predict(self, X):
        """
        預測新數據
        
        Args:
            X (list): 待預測的文本列表
            
        Returns:
            np.array: 預測結果
        """
        X_clean = [self.preprocess_text(text) for text in X]
        X_features = self.vectorizer.transform(X_clean)
        return self.model.predict(X_features)
    
    def predict_proba(self, X):
        """
        預測機率
        
        Args:
            X (list): 待預測的文本列表
            
        Returns:
            np.array: 預測機率
        """
        X_clean = [self.preprocess_text(text) for text in X]
        X_features = self.vectorizer.transform(X_clean)
        return self.model.predict_proba(X_features)

def analyze_threshold_metrics(y_test, y_prob):
    """
    Analyze model performance across different threshold values
    """
    thresholds = np.arange(0.1, 1.0, 0.1)
    metrics = []
    
    for threshold in thresholds:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
        tp = np.sum((y_test == 1) & (y_pred == 1))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tn = np.sum((y_test == 0) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': (tp + tn) / len(y_test)
        })
    
    return pd.DataFrame(metrics)

def create_visualization(df, y_test, y_pred, y_prob, model, classifier, project_root, test_size=0.2, random_state=42):
    """
    Create and save visualization charts
    """
    # Create visualization directory
    vis_dir = os.path.join(project_root, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label')
    plt.title('Email Distribution')
    plt.xlabel('Email Type')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Ham', 'Spam'])
    plt.savefig(os.path.join(vis_dir, 'distribution.png'))
    plt.close()
    
    # NOTE: Word clouds removed per UI requirements. We produce token frequency CSV instead.
    
    # 3. Top Tokens Analysis
    # 3. Top Tokens Analysis: compute token frequencies per class and save CSV for app to consume
    feature_names = classifier.vectorizer.get_feature_names_out()

    # Preprocess messages using classifier preprocessing to match vectorizer
    preprocessed = df['message'].apply(classifier.preprocess_text)

    X_all = classifier.vectorizer.transform(preprocessed)

    # boolean masks for classes
    ham_mask = df['label'] == 'ham'
    spam_mask = df['label'] == 'spam'

    X_ham = classifier.vectorizer.transform(preprocessed[ham_mask])
    X_spam = classifier.vectorizer.transform(preprocessed[spam_mask])

    ham_counts = np.asarray(X_ham.sum(axis=0)).ravel()
    spam_counts = np.asarray(X_spam.sum(axis=0)).ravel()

    toks = []
    for token, hcount, scount in zip(feature_names, ham_counts, spam_counts):
        toks.append({'token': token, 'ham_count': int(hcount), 'spam_count': int(scount), 'total_count': int(hcount + scount)})

    toks_df = pd.DataFrame(toks).sort_values('total_count', ascending=False)
    toks_df.to_csv(os.path.join(vis_dir, 'top_tokens.csv'), index=False)
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(vis_dir, 'roc_curve.png'))
    plt.close()
    
    # 5. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Ham', 'Spam'])
    plt.yticks([0.5, 1.5], ['Ham', 'Spam'])
    plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
    plt.close()

    # 6. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
    avg_precision = average_precision_score(y_test, y_prob[:, 1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'Average Precision = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'precision_recall_curve.png'))
    plt.close()

    # 7. Threshold Analysis
    metrics_df = analyze_threshold_metrics(y_test, y_prob)
    
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', marker='o')
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', marker='o')
    plt.plot(metrics_df['threshold'], metrics_df['f1_score'], label='F1 Score', marker='o')
    plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy', marker='o')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Model Metrics vs Decision Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'threshold_analysis.png'))
    plt.close()
    
    # Save metrics to CSV
    metrics_df.round(3).to_csv(os.path.join(vis_dir, 'threshold_metrics.csv'), index=False)
    
    # Save model configuration
    config = {
        'test_size': test_size,
        'random_seed': random_state,
        'current_threshold': 0.5,
        'model_performance': {
            'auc_score': roc_auc,
            'avg_precision': avg_precision,
            'accuracy': (y_test == y_pred).mean(),
            'confusion_matrix': {
                'true_positives': int(cm[1][1]),
                'false_positives': int(cm[0][1]),
                'true_negatives': int(cm[0][0]),
                'false_negatives': int(cm[1][0])
            }
        }
    }
    
    with open(os.path.join(project_root, 'models', 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def main():
    # 取得專案根目錄的絕對路徑
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    # 建立模型保存目錄
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # 載入數據
    url = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    df = pd.read_csv(url, names=['label', 'message'])
    
    # 將標籤轉換為數值
    df['label_num'] = (df['label'] == 'spam').astype(int)
    
    # Parse CLI args (allow overriding from app)
    parser = argparse.ArgumentParser(description='Train spam classifier and generate visualizations')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'],
        df['label_num'],
        test_size=args.test_size,
        random_state=args.random_seed
    )
    
    # 訓練模型
    classifier = SpamClassifier()
    classifier.fit(X_train, y_train)
    
    # 評估模型
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)
    
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))
    
    # 創建視覺化
    create_visualization(df, y_test, y_pred, y_prob, classifier.model, classifier, project_root, test_size=args.test_size, random_state=args.random_seed)
    print("\n視覺化圖表已保存到 visualizations/ 目錄")
    
    # 保存模型和向量化器
    model_path = os.path.join(models_dir, 'model.pkl')
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    joblib.dump(classifier.model, model_path)
    joblib.dump(classifier.vectorizer, vectorizer_path)
    print(f"\n模型已保存到：{models_dir}")

if __name__ == "__main__":
    main()