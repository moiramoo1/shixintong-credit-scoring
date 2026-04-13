# 食信通 (ShiXinTong) - Credit Scoring for Catering SMEs

## Overview
ShiXinTong is a credit risk assessment system designed for small and micro catering enterprises. It addresses the "information asymmetry" problem that makes it difficult for small restaurants to obtain loans.

## Key Features
- **Multi-source Data Fusion**: Integrates data from Meituan (ratings, sales),工商 registration, judicial cases, and administrative penalties
- **Feature Engineering**: Constructs conflict features (e.g., "high rating but has penalty"),交叉 features, and fusion features
- **XGBoost Model**: Achieves AUC of 0.606 on 100 real restaurant samples
- **Interactive Web Interface**: Built with Streamlit for easy demonstration

## Data Sources
| Source | Fields |
|--------|--------|
| Meituan | Rating, monthly sales, avg spend, review count |
| Business Registration | Years in operation, registered capital, employees |
| Judicial | Number of legal cases |
| Penalty | Administrative penalty records |

## Tech Stack
- Python 3.11+
- XGBoost
- Streamlit
- Pandas / NumPy
- Plotly

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/shixintong-credit-scoring.git
cd shixintong-credit-scoring

# Install dependencies
pip install streamlit pandas numpy scikit-learn joblib plotly xgboost

# Run the app
streamlit run app.py
