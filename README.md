# 🚖 Predicting Uber Ride Cancellations

A fast, interpretable machine learning pipeline to predict ride cancellations using **XGBoost** and other classifiers.  
The project focuses on handling **class imbalance (~7% cancellations)** while ensuring model interpretability and strong performance.

## 📌 Introduction: Why This Matters
Every cancelled ride means lost revenue, wasted time, and a sign of potential customer churn.  
By predicting ride cancellations **before they happen**, ride-hailing platforms can:
- 🚐 Improve customer experience  
- 📍 Optimize driver dispatch  
- 🛠️ Reduce operational inefficiencies  

This project builds a predictive model using only **pre-ride booking metadata** — ensuring no data leakage and full interpretability.

## 🎯 Problem Statement
Can we predict if a customer will cancel an Uber ride **before it begins**, using only booking details?  

- **Task:** Binary Classification  
- **Target Variable:** `target_customer_cancelled` (`1 = cancelled`, `0 = not cancelled`)  
- ⚠️ **Note:** Labels are highly imbalanced (~7% cancellations)

## 📊 Dataset
- **Name:** Uber Ride Analytics 2025
- **Size:** ~150,000 bookings  
- **Target:** Derived from booking status → `target_customer_cancelled`  
- **Features:** Time of day, vehicle type, pickup/drop locations, user/driver ratings, etc.  
- **Quality:** Clean, well-structured, minimal missing values  

## 🛠️ Project Workflow
1. 🧹 Data Cleaning  
2. 🔍 Exploratory Data Analysis (EDA)  
3. ⚙️ Feature Engineering  
4. ⚖️ Handle Class Imbalance (undersampling + class weights)  
5. 🤖 Train Models: Logistic Regression, Random Forest, XGBoost  
6. 📈 Evaluate using ROC-AUC & F1 Score  
7. 💡 Highlight insights & business recommendations  

## 📦 Tech Stack
- **Python**  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  

## 📌 Key Results
- ✅ Built an interpretable, imbalance-aware prediction pipeline  
- ✅ Improved cancellation prediction by leveraging class weights & feature engineering  
- ✅ Achieved strong performance (ROC-AUC & F1 balanced for imbalanced data)  

## 🚀 Future Enhancements
- Incorporate real-time features (traffic, surge pricing)  
- Deploy as an API for real-time ride cancellation prediction  
- Add SHAP-based interpretability for model explanations  


## 📜 License
This project is licensed under the MIT License.
