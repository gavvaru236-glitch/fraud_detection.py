# Credit Card Fraud Detection using Random Forest

An end-to-end supervised machine learning pipeline implemented in Python to detect fraudulent credit card transactions using historical, highly imbalanced transaction patterns.

## Project Workflow
1. **Exploratory Data Analysis**: Analyzed class proportions confirming severe operational imbalance (only ~0.17% fraud transactions). 
2. **Feature Analysis**: Constructed a comprehensive correlation heatmap matrix isolating inter-variable dependencies.
3. **Data Splitting**: Partitioned the 284,807 transaction entries into an 80/20 train-test split.
4. **Model Architecture**: Deployed a robust ensemble **Random Forest Classifier** to extract hidden patterns from anonymized principal component features.
5. **Multi-Metric Evaluation**: Evaluated performance on precision, sensitivity (recall), F1-Score, and the Matthews Correlation Coefficient.

## Performance Metrics Achieved
* **Accuracy**: 99.96%
* **Precision**: 0.9873 (Minimal false-alarm operational overhead)
* **Recall**: 0.7959 (High sensitivity to capturing true anomalies)
* **F1-Score**: 0.8814
* **Matthews Correlation Coefficient (MCC)**: 0.8863

## Stack Used
* **Language**: Python 3
* **Libraries**: Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib
