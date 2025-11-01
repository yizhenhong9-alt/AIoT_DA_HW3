"""
垃圾郵件分類 Streamlit 網頁應用
"""

import streamlit as st
import joblib
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import altair as alt
from PIL import Image
from preprocessing import preprocess_text

def load_models():
    """
    載入訓練好的模型和向量化器
    """
    try:
        # 取得專案根目錄的絕對路徑
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, 'models')
        
        # 檢查模型文件是否存在
        model_path = os.path.join(models_dir, 'model.pkl')
        vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("模型文件不存在")
        
        # 載入模型和向量化器
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        raise Exception(f"載入模型時發生錯誤：{str(e)}")

def predict_spam(text, model, vectorizer):
    """
    預測文本是否為垃圾郵件
    
    Args:
        text (str): 輸入文本
        model: 訓練好的模型
        vectorizer: TF-IDF 向量化器
        
    Returns:
        float: 垃圾郵件的機率
    """
    # 預處理文本
    text_clean = preprocess_text(text)
    
    # 特徵提取
    features = vectorizer.transform([text_clean])
    
    # 預測
    proba = model.predict_proba(features)[0]
    return proba[1]  # 返回是垃圾郵件的機率

def load_config():
    """Load model configuration"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'models', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    except Exception:
        return None

@st.cache_data
def load_full_dataset():
    """
    Loads the local dataset and caches it.
    """
    # Construct path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, 'dataset', 'sms_spam_no_header.csv')
    df = pd.read_csv(dataset_path, names=['label', 'message'])
    df['label_num'] = (df['label'] == 'spam').astype(int)
    return df

def main():
    st.title("Spam Email Detection System 🚫✉️")
    st.write("This system uses machine learning to identify spam emails. Enter email content below for analysis.")
    
    # Load configuration
    config = load_config()
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model parameters
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.01,
        help="Probability threshold for classifying an email as spam"
    )

    # Allow user to adjust test size and random seed and top-N tokens
    default_test_size = config['test_size'] if config and 'test_size' in config else 0.2
    default_seed = config['random_seed'] if config and 'random_seed' in config else 42

    test_size = st.sidebar.slider("Test Size", min_value=0.05, max_value=0.5, value=float(default_test_size), step=0.01, format="%.2f")
    random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=int(default_seed))
    top_n = st.sidebar.number_input("Top-N Tokens", min_value=5, max_value=100, value=15)
    
    if config:
        st.sidebar.markdown("### Model Settings")
        st.sidebar.write(f"Test Size: {config['test_size']}")
        st.sidebar.write(f"Random Seed: {config['random_seed']}")
        
        st.sidebar.markdown("### Current Performance")
        st.sidebar.write(f"AUC Score: {config['model_performance']['auc_score']:.3f}")
        st.sidebar.write(f"Avg Precision: {config['model_performance']['avg_precision']:.3f}")
    
    # 添加頁籤
    tab1, tab2 = st.tabs(["Prediction", "Model Analysis"])
    
    with tab1:
        # 載入模型
        try:
            model, vectorizer = load_models()
        except Exception as e:
            # 更詳細的錯誤診斷，幫助在遠端（例如 Streamlit Cloud）排查問題
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, 'models')
            info_lines = []
            info_lines.append(f"載入模型時發生例外: {e}")
            try:
                if os.path.exists(models_dir):
                    info_lines.append(f"models 資料夾位於: {models_dir}")
                    for fname in sorted(os.listdir(models_dir)):
                        fpath = os.path.join(models_dir, fname)
                        try:
                            size = os.path.getsize(fpath)
                        except Exception:
                            size = 'NA'
                        info_lines.append(f" - {fname} (size={size})")
                else:
                    info_lines.append("models 資料夾不存在於 repo 根目錄")
            except Exception as _:
                info_lines.append("無法列出 models 內容（權限或其他錯誤）")

            st.error("錯誤：無法載入模型。請確保已經運行過 train_model.py 訓練模型，或已將 models/ 資料夾推送至遠端 repo。")
            with st.expander("載入模型診斷資訊"):
                for line in info_lines:
                    st.write(line)

            st.stop()
        
        # 文本輸入
        text = st.text_area(
            "請輸入郵件內容：",
            height=200,
            placeholder="在此輸入郵件內容..."
        )
        
        if st.button("分析"):
            if not text.strip():
                st.warning("請輸入郵件內容")
            else:
                # 進行預測
                spam_prob = predict_spam(text, model, vectorizer)
                
                # 顯示結果
                st.subheader("分析結果")
                
                # 使用進度條顯示預測機率
                st.progress(spam_prob)
                
                # 顯示預測結果
                if spam_prob > 0.5:
                    st.error(f"⚠️ 這可能是垃圾郵件 (信心度: {spam_prob:.1%})")
                else:
                    st.success(f"✅ 這可能是正常郵件 (信心度: {1-spam_prob:.1%})")
                
                # 顯示詳細信息
                with st.expander("查看詳細分析"):
                    st.write(f"垃圾郵件機率：{spam_prob:.2%}")
                    st.write(f"正常郵件機率：{1-spam_prob:.2%}")
                    
                    # 顯示預處理後的文本
                    st.write("預處理後的文本：")
                    st.code(preprocess_text(text))

    with tab2:
        st.header("Model Analysis")

        # Show data distribution
        st.subheader("1. Data Distribution")
        def _show_image_safe(relpath, caption=None):
            p = os.path.join(os.getcwd(), relpath)
            if os.path.exists(p):
                try:
                    img = Image.open(p)
                    st.image(img, caption=caption)
                except Exception as e:
                    st.error(f"Failed to open image {relpath}: {e}")
            else:
                st.error(f"Cannot find image: {relpath}")

        _show_image_safe("visualizations/distribution.png")

        # Load dataset and compute dynamic metrics (without retraining the model)
        # We'll re-split using the selected test_size and random_seed, then evaluate the loaded model on that test set.
        try:
            full_df = load_full_dataset()

            # split according to sidebar inputs
            X_train, X_test, y_train, y_test = train_test_split(
                full_df['message'], full_df['label_num'], test_size=float(test_size), random_state=int(random_seed)
            )

            # preprocess and vectorize using loaded vectorizer
            X_test_pre = X_test.apply(preprocess_text)
            X_test_features = vectorizer.transform(X_test_pre)

            y_prob = model.predict_proba(X_test_features)
            y_pred = (y_prob[:, 1] >= threshold).astype(int)

            # compute metrics for dynamic display
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob[:, 1])
            avg_precision = average_precision_score(y_test, y_prob[:, 1])
            cm = confusion_matrix(y_test, y_pred)

            # threshold metrics table
            thresholds = np.round(np.arange(0.0, 1.01, 0.01), 2)
            rows = []
            for th in thresholds:
                yp = (y_prob[:, 1] >= th).astype(int)
                tp = np.sum((y_test == 1) & (yp == 1))
                fp = np.sum((y_test == 0) & (yp == 1))
                fn = np.sum((y_test == 1) & (yp == 0))
                tn = np.sum((y_test == 0) & (yp == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / len(y_test)
                rows.append({'threshold': th, 'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': accuracy})

            threshold_metrics_df = pd.DataFrame(rows)

        except Exception as e:
            st.error(f"Failed to compute dynamic metrics: {e}")
            full_df = None
            threshold_metrics_df = None

    # Top Tokens (interactive)
        st.subheader("2. Top Tokens Analysis (interactive)")
        try:
            toks_path = os.path.join('visualizations', 'top_tokens.csv')
            toks_df = pd.read_csv(toks_path)
            # allow selecting which class to view
            token_view = st.selectbox('Token view', options=['spam', 'ham', 'total'], index=0)
            if token_view == 'spam':
                plot_df = toks_df[['token', 'spam_count']].rename(columns={'spam_count': 'count'})
            elif token_view == 'ham':
                plot_df = toks_df[['token', 'ham_count']].rename(columns={'ham_count': 'count'})
            else:
                plot_df = toks_df[['token', 'total_count']].rename(columns={'total_count': 'count'})

            plot_df = plot_df.nlargest(int(top_n), 'count')[['token', 'count']]

            chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('count:Q'),
                y=alt.Y('token:N', sort='-x'),
                color=alt.Color('token:N', legend=None)
            ).properties(width=700, height=30 * min(len(plot_df), 30))
            st.altair_chart(chart, width='stretch')
            with st.expander('Top tokens data'):
                st.dataframe(plot_df.reset_index(drop=True))
        except Exception as e:
            st.error(f"Cannot load top tokens data: {e}")
        
        # Show threshold analysis (table) — dynamic using current sidebar Test Size/Seed/Threshold
        st.subheader("3. Threshold Analysis (table)")
        if threshold_metrics_df is not None:
            st.dataframe(threshold_metrics_df.style.format({
                'threshold': '{:.2f}',
                'precision': '{:.3f}',
                'recall': '{:.3f}',
                'f1_score': '{:.3f}',
                'accuracy': '{:.3f}'
            }))

            sel_thresh = st.slider(
                'Select Threshold to view row',
                min_value=float(threshold_metrics_df.threshold.min()),
                max_value=float(threshold_metrics_df.threshold.max()),
                value=float(threshold),
                step=0.01
            )
            row = threshold_metrics_df.iloc[(threshold_metrics_df['threshold'] - sel_thresh).abs().argsort()[:1]]
            st.markdown('**Metrics at selected threshold**')
            st.table(row.reset_index(drop=True).round(3))
        else:
            st.error('Threshold metrics unavailable')
        
        # Show ROC and PR curves
        st.subheader("4. Model Performance Curves")
        col1, col2 = st.columns(2)
        with col1:
            # Draw dynamic ROC curve using computed fpr/tpr
            try:
                roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
                roc_chart = alt.Chart(roc_df).mark_line().encode(x='fpr', y='tpr')
                st.altair_chart(roc_chart.properties(height=300, width=400), width='stretch')
                st.write(f'AUC = {roc_auc:.3f}')
            except Exception:
                st.error("Cannot compute ROC curve")
        with col2:
            try:
                pr_df = pd.DataFrame({'recall': recall_vals, 'precision': precision_vals})
                pr_chart = alt.Chart(pr_df).mark_line().encode(x='recall', y='precision')
                st.altair_chart(pr_chart.properties(height=300, width=400), width='stretch')
                st.write(f'Average precision = {avg_precision:.3f}')
            except Exception:
                st.error("Cannot compute Precision-Recall curve")

        # Show confusion matrix
        st.subheader("5. Confusion Matrix")
        try:
            st.image("visualizations/confusion_matrix.png")
        except Exception:
            st.error("Cannot load confusion matrix")

    # 添加說明
    with st.sidebar:
        st.subheader("關於")
        st.write("""
        此應用使用機器學習來識別垃圾郵件。
        
        使用的技術：
        - 支持向量機 (SVM)
        - TF-IDF 文本向量化
        - NLTK 文本預處理
        """)
        
        st.subheader("使用說明")
        st.write("""
        1. 在文本框中輸入郵件內容
        2. 點擊「分析」按鈕
        3. 查看預測結果和詳細分析
        """)

if __name__ == "__main__":
    main()