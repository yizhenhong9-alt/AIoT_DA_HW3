# Spam Email Detection — AIoT_Data_HW3

此專案為一個簡易的垃圾郵件（SMS）分類系統，包含訓練腳本與 Streamlit 互動式分析介面。主要功能包括：

- 訓練 SVM 分類器（使用 TF-IDF 向量化）
- 生成模型效能視覺化（ROC、Precision-Recall、混淆矩陣、標籤分布）
- Threshold Analysis（門檻指標表格）動態檢視
- Top-N tokens 分析（可在 UI 調整 Top-N，並支援 spam/ham/total 檢視）
- Streamlit 前端可動態調整 Test Size、Random Seed、Decision Threshold 與 Top-N（會即時重新計算評估指標與曲線，無須再次訓練）

---

## 目錄

- `src/train_model.py` - 模型訓練與視覺化建立（會輸出到 `visualizations/` 並儲存模型到 `models/`）
- `src/app.py` - Streamlit 前端應用
- `dataset/` - 建議放置資料集 (`sms_spam_no_header.csv`)（已加入範例檔）
- `visualizations/` - 訓練腳本會產生的圖表與 CSV（如 `threshold_metrics.csv`, `top_tokens.csv`）
- `models/` - 訓練後模型（`model.pkl`）與向量化器（`vectorizer.pkl`）與 `config.json`

---

## 快速開始（Windows / cmd.exe）

1. 建議先建立虛擬環境（可選）：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. 安裝相依套件（在專案根目錄執行）：

```cmd
pip install -r requirements.txt
```

如果專案尚未提供 `requirements.txt`，可手動安裝以下套件：

```cmd
pip install streamlit scikit-learn pandas numpy nltk joblib matplotlib seaborn altair pillow
```

3. （首次）下載 NLTK 必要資源：

在 Python 中執行或於第一次運行時會自動下載：

```py
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. （選擇）離線資料集：

專案已包含 `dataset/sms_spam_no_header.csv`。若要重新下載：

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv" -OutFile "dataset\sms_spam_no_header.csv" -UseBasicParsing
```

5. 訓練模型並生成視覺化（會輸出至 `visualizations/` 並儲存模型在 `models/`）：

```cmd
python src\train_model.py --test_size 0.2 --random_seed 42
```

可透過 `--test_size` 與 `--random_seed` 參數調整資料切分行為。

6. 啟動 Streamlit 應用：

```cmd
streamlit run src\app.py
```

之後開啟終端顯示的本機網址（通常是 http://localhost:8501）。

---

## Streamlit 介面說明

- Prediction：輸入郵件內容並點選「分析」獲得垃圾郵件機率預測（使用保存的模型）
- Model Analysis：
  - Data Distribution：顯示資料標籤分布（由訓練時輸出圖片）
  - Top Tokens Analysis：從 `visualizations/top_tokens.csv` 讀取，支援選擇 spam/ham/total 與 Top-N 數量，可動態調整並以互動圖表顯示
  - Threshold Analysis：顯示完整門檻指標表格，並可用 slider 選取單一 threshold 以表格形式檢視
  - Model Performance Curves：動態根據目前介面的 Test Size / Random Seed / Threshold 計算並繪出 ROC 與 Precision-Recall 曲線（Altair）
  - Confusion Matrix：載入 `visualizations/confusion_matrix.png`（若存在）以提供視覺化檢視

重要：在 UI 調整 `Test Size`、`Random Seed` 或 `Decision Threshold` 時，系統會在前端重新以載入的模型與本地資料重新分割並即時計算指標（此為「重新計算評估」而非完整 re-train）。

---

## 檔案產出與用途

- `models/model.pkl`：訓練後的分類器（joblib）
- `models/vectorizer.pkl`：TF-IDF 向量化器
- `models/config.json`：儲存訓練參數與模型指標（test_size, random_seed, auc, etc.）
- `visualizations/threshold_metrics.csv`：原始 threshold analysis CSV（訓練時生成）
- `visualizations/top_tokens.csv`：token 頻率資料，供 UI 生成 Top-N 視覺化
- `visualizations/*.png`：若訓練時有成功執行，包含 distribution、roc_curve、precision_recall_curve、confusion_matrix 等

---

## 常見問題 & 排錯

- Streamlit 報錯無法載入圖片（MediaFileStorageError）：請確認 `visualizations/` 中對應檔案存在，若先前有更新視覺化生成邏輯，建議刪除舊檔後重新執行 `python src\train_model.py` 來重新產生。
- NLTK 無法下載：檢查網路或手動在 Python 中執行 `nltk.download('punkt')` 與 `nltk.download('stopwords')`。
- 若要離線使用資料集，請確保 `dataset/sms_spam_no_header.csv` 已存在（本專案已將其加入 `dataset/`）。

---

## 開發與擴充建議

- 將 `requirements.txt` 與 `venv` 一併管理，便於重現環境
- 若模型訓練耗時過久，考慮把訓練改成背景任務（Celery、RQ 或類似方案），並在 UI 顯示完成通知
- 可增加單元測試來覆蓋預處理、向量化與閾值計算邏輯

---

