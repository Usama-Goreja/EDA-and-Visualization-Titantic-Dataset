
# 🚢 Titanic Survival Prediction - Machine Learning Project

This project applies machine learning to the classic Titanic dataset to predict passenger survival. Using data preprocessing, exploratory data analysis, and Logistic Regression, we build a model that determines the likelihood of a passenger surviving the Titanic disaster based on specific features.

---

## 📌 Project Objective

The primary goal is to:

- Analyze the Titanic dataset to uncover survival patterns.
- Clean and preprocess the data for modeling.
- Train a machine learning model (Logistic Regression) to predict survival.
- Evaluate the model's accuracy and performance using standard metrics.

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── titanic_analysis.ipynb      # Jupyter notebook with full analysis
│
├── src/
│   ├── preprocessing.py            # Data cleaning and encoding
│   ├── eda.py                      # Exploratory data analysis
│   ├── model.py                    # Model training and evaluation
│   └── predict.py                  # Custom prediction function
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🧠 Technologies Used

- **Python 3.8+**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 🧹 Data Preprocessing

- Imputed missing values in `Age` and `Embarked`
- Dropped the `Cabin` column due to high missing rates
- Removed outliers in `Fare` and `Age` using IQR method
- Encoded categorical variables using Label Encoding

---

## 📊 Exploratory Data Analysis

Key insights discovered:
- Overall survival rate: ~38%
- Women had significantly higher survival chances
- Passengers in first and second class had better odds of survival
- Positive correlation found between survival and features like sex, class, and fare

Visualizations included:
- Bar plots
- Histograms
- Correlation heatmaps

---

## 🤖 Machine Learning Model

- Model Used: **Logistic Regression**
- Features: Sex, Age, Pclass, Fare, Embarked, etc.
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

**Model Accuracy:** ~80%

---

## 📦 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Usama-Goreja/titanic-survival-prediction.git
cd titanic-survival-prediction
```



3. Launch Jupyter Notebook:
```bash
jupyter notebook notebooks/titanic_analysis.ipynb
```

---

## 🔮 Future Enhancements

- Apply advanced feature engineering techniques
- Test alternative ML models (Random Forest, XGBoost)
- Perform hyperparameter optimization
- Include additional external data sources

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

https://www.linkedin.com/in/usamaiqbal2000/
