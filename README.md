# Rock-vs-Mine-prediction
This project aims to classify sonar signals as either rocks or mines using various statistical and machine learning techniques. The dataset, sourced from the UCI Machine Learning Repository, contains 60 numeric features representing sonar signals bouncing off different surfaces. The target is a binary classification — Rock or Mine.

📌 Project Objectives
Understand the data through exploratory data analysis (EDA)

Apply Shapiro-Wilk test for normality and select appropriate parametric and non-parametric tests

Use Principal Component Analysis (PCA) to reduce dimensionality

Perform feature selection and multicollinearity checks using VIF

Train and evaluate Random Forest, Support Vector Machine (SVM), and Logistic Regression

Improve model accuracy using hyperparameter tuning and cross-validation

Conduct quantile regression analysis on principal components for deeper statistical insights

Visualize results using multimodal data visualizations

🧰 Tools and Technologies Used
Programming Language: Python

Libraries: NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Statsmodels, Scipy

Environment: Google Colab

Models: Random Forest, SVM, Logistic Regression

Statistical Tests: Shapiro-Wilk, Welch’s t-test, Mann-Whitney U test

Techniques: PCA, VIF Analysis, Quantile Regression, Cross-validation

📈 Key Features
Data Cleaning and Preprocessing: Handling missing values, checking data types, and normality

Dimensionality Reduction: PCA to identify the most informative features

Model Building: Multiple classifiers evaluated with and without tuning

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

Statistical Testing: In-depth analysis of distribution, mean differences, and test suitability

Quantile Regression: For analyzing the behavior of principal components across different quantiles

🧪 Statistical Insights
The Shapiro-Wilk test showed that the data and some principal components are not normally distributed.

As a result, both parametric (Welch’s t-test) and non-parametric (Mann-Whitney U test) were applied.

Quantile regression revealed additional characteristics of the data that aren’t captured by mean-based models.

🚀 Model Performance
Random Forest Classifier: Achieved the highest accuracy of ~87% after hyperparameter tuning.

Logistic Regression: Achieved a decent performance with 76% accuracy, serving as a strong baseline model.

Support Vector Machine (SVM): Also performed well after tuning, offering competitive results.

Cross-validation ensured robust evaluation of model generalization and helped prevent overfitting.

🌱 Future Work
Apply deep learning models such as CNNs for signal classification.

Incorporate time-series signal processing techniques to capture signal patterns.

Explore ensemble learning methods like Gradient Boosting or XGBoost.

Expand the analysis to real-time sonar signal classification applications.

📎 Appendix
VIF analysis results

PCA component loading plots

Summary of statistical tests

SHAP/feature importance visualizations

Cross-validation scores


