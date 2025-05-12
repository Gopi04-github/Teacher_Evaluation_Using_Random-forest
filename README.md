# 📊 Teacher Evaluation System Based on Student Understanding

This project is a Machine Learning-based system that evaluates and predicts the best subject area (Theory, Logical, or Analytical) for students based on their scores. It helps indirectly assess teacher effectiveness by analyzing student performance patterns across different sections.

## 🔍 Project Description

- A mock dataset is generated with 5 sections (A–E), each having 10 students.
- Students are evaluated on three key skills: **Theory**, **Logical**, and **Analytical**.
- The model predicts each student’s strongest subject based on their scores.
- Visualizations are provided to analyze:
  - Distribution of subject strengths
  - Section-wise average performance
  - Top-performing sections for each subject

## 🧠 Machine Learning Model

- **Model Used**: Random Forest Classifier
- **Target Variable**: Predicted best subject (`Result`)
- **Features**:
  - Theory_Mark
  - Logical_Mark
  - Analytical_Mark

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

## 📈 Visualizations

- Bar chart for strongest subject distribution
- Section-wise average score chart
- Top sections by subject performance
