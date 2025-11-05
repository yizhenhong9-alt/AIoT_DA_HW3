# Spam Email Detection — AIoT_Data_HW3

此專案為一個簡易的垃圾郵件（SMS）分類系統，包含訓練腳本與 Streamlit 互動式分析介面。主要功能包括：

- 訓練 SVM 分類器（使用 TF-IDF 向量化）
- 生成模型效能視覺化（ROC、Precision-Recall、混淆矩陣、標籤分布）
- Threshold Analysis（門檻指標表格）動態檢視
- Top-N tokens 分析（可在 UI 調整 Top-N，並支援 spam/ham/total 檢視）
- Streamlit 前端可動態調整 Test Size、Random Seed、Decision Threshold 與 Top-N（會即時重新計算評估指標與曲線，無須再次訓練）

---

## Tech Stack
### 核心技術
- Python 3.10+
- pandas, numpy：數據處理和分析
- scikit-learn：
  - TfidfVectorizer：文本特徵提取
  - SVM、LogisticRegression：分類模型
- nltk：文本清理和停用詞移除
- matplotlib, seaborn：數據視覺化
- wordcloud：關鍵字頻率視覺化
- joblib：模型持久化

### 部署工具
- Streamlit：網頁介面開發
- GitHub：版本控制
- Streamlit Cloud：應用部署

## Project Conventions

### 專案結構
```
src/
 ├── app.py                 # Streamlit 應用程式
 ├── train_model.py         # 模型訓練腳本
 ├── preprocessing.py       # 文本預處理模組
dataset/
 ├── sms_spam_no_header.csv # 原始數據
models/
 ├── model.pkl              # 訓練模型
 ├── vectorizer.pkl         # TF-IDF 向量化器
requirements.txt            # 依賴套件清單
```

### Architecture Patterns
- 模組化設計：將功能分離為獨立模組
- 數據預處理流程：
  1. 文本清理（移除標點、轉小寫）
  2. 停用詞移除
  3. TF-IDF 向量化
- 模型訓練流程：
  1. 數據載入和分割
  2. 特徵提取
  3. 模型訓練和評估
  4. 結果視覺化

### Testing Strategy
- 使用 pytest 進行單元測試
- 模型評估指標：
  - 準確率 (Accuracy)
  - 精確率 (Precision)
  - 召回率 (Recall)
  - F1 分數
- 使用混淆矩陣和 ROC 曲線進行視覺化評估

## Domain Context
### 垃圾郵件分類
- 二元分類問題：spam vs. ham
- 文本特徵工程關鍵點：
  - 停用詞處理
  - 詞頻-逆文檔頻率 (TF-IDF)
  - N-gram 特徵
- 模型選擇考量：
  - SVM：高維空間中的優秀表現
  - 邏輯迴歸：可解釋性和效率

## Important Constraints
- Python 版本要求：3.10+
- 記憶體使用限制：注意大型數據集處理
- 模型大小：部署時的限制
- API 響應時間：需在 3 秒內

## External Dependencies
### 數據來源
- 數據集 URL：https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- 格式：CSV，包含 'label' 和 'message' 欄位

### 部署相關
- Streamlit Cloud 部署：https://7114056010spamemail.streamlit.app/
- NLTK 數據包：需要額外下載
- scikit-learn：模型和向量化器版本相容性
