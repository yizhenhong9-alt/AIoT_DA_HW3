# Project Context

## Purpose
Spam Email Classification WebApp 是一個完整的垃圾郵件檢測系統，使用機器學習技術進行分類。本專案擴展了 "Hands-On Artificial Intelligence for Cybersecurity" 第三章的內容，加入了完整的預處理流程、模型訓練、視覺化分析和 Streamlit 網頁應用。

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
 ├── app.py           # Streamlit 應用程式
 ├── train_model.py   # 模型訓練腳本
 ├── preprocessing.py # 文本預處理模組
 ├── utils.py        # 通用工具函數
data/
 ├── raw/            # 原始數據
 ├── processed/      # 處理後數據
models/
 ├── model.pkl       # 訓練模型
 ├── vectorizer.pkl  # TF-IDF 向量化器
requirements.txt     # 依賴套件清單
```

### Code Style
- 遵循 PEP8 規範
- 所有函數必須包含清晰的 docstrings
- 採用模組化和可重用的程式碼結構
- 變數命名使用 snake_case
- 類別命名使用 PascalCase

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

### Git Workflow
- 主分支：main
- 開發分支：develop
- 功能分支：feature/*
- 修復分支：hotfix/*
- 提交訊息格式：<type>(<scope>): <message>
  - type: feat, fix, docs, style, refactor
  - scope: model, api, ui, data, etc.

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
