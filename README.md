# Psychological State Detection using Machine Learning

 **Live App:**
   https://psychological-state-detection-ml-3bgur9jmb7vhes2pgewzuq.streamlit.app/

   Upload your own dataset and get instant predictions using the deployed ML model.

---

---

##  Project Overview

This project uses **Machine Learning** to predict a person's **psychological state** based on physiological and behavioral data.

The model is trained on structured data and deployed using **Streamlit**, allowing users to upload a CSV file and instantly get predictions.

---

##  Features

* 📊 Upload CSV data for prediction
* 🤖 ML models (Random Forest, Logistic Regression, Gradient Boosting)
* 🧠 Predict psychological states
* 📥 Download prediction results
* 🌐 Fully deployed web application

---

## Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Streamlit**
* **Joblib**

---

##  Project Structure

```
psychological-state-detection-ml/
│
├── streamlit_app.py # Streamlit web app
├── final_tuned_model.pkl # Trained ML model
├── best_model.pkl # Saved best model
├── label_encoder.pkl # Label encoder
├── feature_columns.pkl # Feature alignment file
│
├── psychological_state_dataset.csv # Dataset
├── psychological_state_detection.ipynb # Jupyter notebook
│
├── notebooks/ # Additional notebooks
├── requirements.txt # Dependencies
├── .gitignore
└── README.md
```

---

##  How It Works

1. User uploads a CSV file
2. Data is preprocessed (encoding + alignment)
3. Model predicts psychological state
4. Results are displayed and downloadable

---

##  Machine Learning Workflow

* Data Cleaning & Preprocessing
* Feature Engineering (One-Hot Encoding)
* Train-Test Split
* Model Training & Evaluation
* Model Selection
* Deployment with Streamlit

---

##  Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 0.21     |
| Random Forest       | 0.28     |
| Gradient Boosting   | 0.24     |

---

##  How to Run Locally

```bash
git clone https://github.com/your-username/psychological-state-detection-ml.git
cd psychological-state-detection-ml
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

##  Input Format

Upload a CSV file with relevant physiological features.
The app automatically processes and aligns the data.

---

##  Key Learnings

* Handling feature mismatch in ML deployment
* Model serialization using Joblib
* Building end-to-end ML pipelines
* Deploying ML apps with Streamlit
* Debugging real-world ML issues

---

## Future Improvements

*  Improve model accuracy
*  Add data visualization
*  Add form-based input (no CSV needed)
*  Use deep learning models

---

##  Author

**Janu Dhanya Vinoth Singh**



Developed by \*\*Janu Dhanya Vinoth Singh\*\* as part of Machine Learning Internship Project.



