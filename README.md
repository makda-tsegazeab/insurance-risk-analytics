Insurance Risk Analytics & Predictive Modeling
📋 Project Overview
This project analyzes historical car insurance data for AlphaCare Insurance Solutions (ACIS) to identify low-risk customer segments and build predictive models for premium optimization. The analysis helps develop targeted marketing strategies and risk-based pricing models.

🎯 Business Objective
Discover "low-risk" customer segments where premiums could be reduced, creating opportunities to attract new clients while maintaining profitability.

📊 Dataset
Source: Historical insurance claims data (February 2014 - August 2015)

Format: Pipe-delimited text file (machineLearningRating_v3.txt)

Size: Large dataset requiring careful memory management

Location: data/raw/machineLearningRating_v3.txt

🏗️ Project Structure
text
insurance-risk-analytics/
├── data/
│   ├── raw/                    # Original data file
│   ├── sample/                 # Sample data for development
│   └── metadata.txt            # Data documentation
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/                    # Python scripts for analysis
├── reports/                    # Generated reports
├── .dvc/                       # Data version control
├── .dvcignore                  # Files to ignore in DVC
├── requirements.txt            # Python dependencies
└── README.md                   # This file
📋 Tasks Completed
Task 1: Exploratory Data Analysis (EDA) & Project Setup
Objective: Develop foundational understanding of data quality and initial risk patterns.

Deliverables:

✅ Created GitHub repository with proper branching strategy

✅ Implemented comprehensive EDA with statistical analysis

✅ Analyzed Loss Ratio (TotalClaims/TotalPremium) across provinces, vehicle types, and gender

✅ Identified distributions and outliers in key financial variables

✅ Created 3 insightful visualizations highlighting key patterns

✅ Examined temporal trends over the 18-month period

✅ Identified high/low-risk vehicle makes/models

Key Files:

explore_data_structure.py - Initial data exploration

EDA notebooks and visualizations

Task 2: Data Version Control (DVC) Setup
Objective: Establish reproducible and auditable data pipeline using DVC.

Deliverables:

✅ Installed and configured DVC for data versioning

✅ Set up local remote storage for data tracking

✅ Implemented .dvcignore to handle large data files

✅ Created sample dataset for development

✅ Version-controlled data pipeline for reproducibility

✅ Merged Task 1 into main branch via Pull Request

Key Files:

.dvc/ - DVC configuration

.dvcignore - Large file exclusion rules

data/metadata.txt - Data documentation

data/sample/ - Sample data for development

Task 3: A/B Hypothesis Testing
Objective: Statistically validate key hypotheses about risk drivers for segmentation strategy.

Hypotheses Tested:

✅ Province Risk Differences: Are there significant risk variations across provinces?

✅ Zip Code Risk Differences: Do risk profiles differ by postal codes?

✅ Zip Code Margin Differences: Are there profit margin variations by location?

✅ Gender Risk Differences: Is there significant risk difference between women and men?

Methodology:

Risk quantified by Claim Frequency and Claim Severity

Statistical tests: Chi-square, ANOVA, t-tests

Significance level: α = 0.05

Deliverables:

✅ Hypothesis testing scripts with statistical validation

✅ Business interpretation of statistical results

✅ Recommendations for risk-adjusted premium strategies

✅ Comprehensive hypothesis testing report

Key Files:

hypothesis_testing.py - Complete hypothesis testing implementation

hypothesis_test_results.csv - Statistical test results

hypothesis_testing_report.md - Business recommendations

Task 4: Predictive Modeling
Objective: Build and evaluate predictive models for dynamic, risk-based pricing system.

Modeling Approaches:

Claim Severity Prediction: Regression models predicting claim amounts

Claim Probability Prediction: Classification models predicting claim likelihood

Premium Optimization: Models to predict optimal premium values

Models Implemented:

Linear Regression

Decision Trees

Random Forests

Logistic Regression

Gradient Boosting (where available)

Deliverables:

✅ Comprehensive data preparation pipeline

✅ Multiple ML models with performance comparison

✅ Feature importance analysis using SHAP/XAI

✅ Model interpretability and business insights

✅ Premium optimization recommendations

✅ Risk-based pricing framework

Key Files:

data_preparation.py - Data preprocessing pipeline

predictive_modeling.py or predictive_modeling_simple.py - Model training

predictive_modeling_report.md - Comprehensive modeling report

Feature importance visualizations

🛠️ Technical Stack
Programming: Python 3.8+

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost (optional)

Visualization: Matplotlib, Seaborn

Statistical Analysis: SciPy, Statsmodels

Version Control: Git, DVC

Model Interpretability: SHAP (where available)

📈 Key Insights
Risk Drivers Identified:
Geographic Variations: Significant risk differences across provinces and zip codes

Vehicle Characteristics: Make, model, and age strongly correlate with claim risk

Policy Factors: Cover type and sum insured impact both frequency and severity

Demographic Factors: Gender and other client attributes show varying risk profiles

Business Recommendations:
Risk-Based Pricing: Implement tiered premiums based on predicted risk scores

Targeted Marketing: Focus on low-risk segments with competitive pricing

Dynamic Pricing: Adjust premiums based on real-time risk assessment

Portfolio Optimization: Balance high-risk and low-risk policies for profitability

🚀 Getting Started
Installation
bash
# Clone repository
git clone https://github.com/makda-tsegazeab/insurance-risk-analytics.git
cd insurance-risk-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install optional packages
pip install xgboost shap  # For advanced modeling
Running the Analysis
Task 1: EDA
bash
python explore_data_structure.py
Task 3: Hypothesis Testing
bash
python hypothesis_testing.py
python generate_report.py
Task 4: Predictive Modeling
bash
# Simple version (no XGBoost required)
python predictive_modeling_simple.py

# Full version (requires XGBoost)
python predictive_modeling.py
📁 File Descriptions
Core Scripts:
explore_data_structure.py - Initial data exploration and format detection

hypothesis_testing.py - Complete A/B testing implementation

generate_report.py - Report generation for hypothesis testing

data_preparation.py - Data preprocessing for modeling

predictive_modeling.py / predictive_modeling_simple.py - Model training and evaluation

Generated Reports:
hypothesis_testing_report.md - Statistical findings and business implications

predictive_modeling_report.md - Model performance and recommendations

minimal_model_report.md - Quick analysis results

Configuration:
requirements.txt - Python dependencies

.dvcignore - Data version control rules

.gitignore - Git exclusion rules

📊 Results Summary
Hypothesis Testing Results:
Province Risk: Significant differences found (p < 0.05)

Gender Risk: Mixed results requiring further investigation

Location-based Pricing: Recommended for high-risk areas

Segment-specific Strategies: Required for optimal pricing

Modeling Performance:
Claim Prediction Accuracy: XX% (varies by model)

Premium Prediction R²: XX% (varies by model)

Key Predictive Features: Vehicle value, location, cover type

Model Interpretability: High with feature importance analysis

🔮 Future Work
Real-time Risk Scoring: Implement API for instant risk assessment

Deep Learning Models: Explore neural networks for complex patterns

Customer Lifetime Value: Predict long-term customer profitability

Fraud Detection: Implement anomaly detection for suspicious claims

Dynamic Pricing Engine: Real-time premium adjustment system
