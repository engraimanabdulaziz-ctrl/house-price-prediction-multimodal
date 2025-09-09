# 🏡 Task 3 – Multimodal House Price Prediction

## 📌 Objective
The goal of this task is to build a **multimodal machine learning model** that predicts house prices by combining:
- **Tabular data** (number of bedrooms, bathrooms, square footage, city, etc.)
- **Image data** (photos of the house)

This demonstrates how structured features and visual features can be fused to improve predictive modeling.

---

## 📂 Dataset
We used the **SoCal House Prices and Images dataset** (Kaggle).  
The dataset contains:
- A CSV file (`socal2.csv`) with 15,474 rows, including:
  - `bed`, `bath`, `sqft` (numeric features)
  - `citi` (categorical feature)
  - `price` (target variable)
- An image folder (`socal_pics`) containing 15,474 house photos.

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Dropped unused columns (`street`, `image_id`).
   - Standardized numeric features (`bed`, `bath`, `sqft`, `n_citi`).
   - One-hot encoded categorical feature (`citi`).

2. **Image Feature Extraction**
   - Images resized to `128×128`.
   - Pre-trained **ResNet18** used as a feature extractor (last FC layer removed).
   - Extracted a 512-dimensional feature vector per image.

3. **Feature Fusion**
   - Concatenated tabular features (419-dim) with image features (512-dim).
   - Created a combined feature space of ~931 dimensions.

4. **Model Training**
   - Trained a **Random Forest Regressor** with 200 trees.
   - Used an 80/20 train-test split.

5. **Evaluation Metrics**
   - **Mean Absolute Error (MAE)**
   - **Root Mean Squared Error (RMSE)**

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **MAE** | 52,300 |
| **RMSE** | 77,800 |

*(Values are approximate and based on experimental runs. Due to random image-row pairing, errors are higher than with true mappings.)*

---

## 🚀 Deliverables
- Jupyter/Colab notebook implementing the full pipeline.
- Trained model saved as:
  - `house_price_model.joblib`
  - `tabular_preprocessor.joblib`
- This README file.

---

## 📌 Key Takeaways
- Multimodal ML combines structured + unstructured data for better predictions.
- Pre-trained CNNs (like ResNet18) are powerful feature extractors for images.
- Proper dataset-image mapping would further improve results and reduce error.
