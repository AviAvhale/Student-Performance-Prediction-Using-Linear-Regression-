# Student-Performance-Prediction-Using-Linear-Regression-# Student Performance Prediction Using Linear Regression

A machine learning project built for **Data Warehousing & Mining (DWM) — CIAP Phase 1** assignment. Predicts a student's final grade (G3) using Linear Regression based on past grades, study habits, and attendance data.

---

## 📌 Project Overview

| Field        | Details                                      |
|--------------|----------------------------------------------|
| Subject      | Data Warehousing & Mining (DWM)              |
| Assignment   | Individual Assignment — CIAP Phase 1         |
| Algorithm    | Linear Regression (scikit-learn)             |
| Dataset      | Student Performance Dataset (student-mat.csv)|
| Source       | UCI Machine Learning Repository              |
| Target       | G3 — Final Grade (scale: 0–20)              |

---

## 🧠 Features Used

| Feature      | Description                        |
|--------------|------------------------------------|
| `G1`         | First period grade (0–20)          |
| `G2`         | Second period grade (0–20)         |
| `studytime`  | Weekly study time (1–4 scale)      |
| `absences`   | Number of school absences          |
| `failures`   | Number of past class failures      |

**Target variable:** `G3` — Final grade

---

## 📊 Results

| Metric                  | Value  |
|-------------------------|--------|
| R² Score                | 0.7822 |
| Mean Absolute Error     | 1.3394 |
| Root Mean Squared Error | 2.1134 |

> Actual values will vary slightly; run the script to see live results.

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── student_prediction.py       # Main ML script (all 9 steps)
├── plots/
│   └── all_plots_combined.png  # Auto-generated visualization dashboard
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Run the script**
```bash
python student_prediction.py
```

The script will:
- Auto-download the dataset from UCI/GitHub
- Train a Linear Regression model
- Print all results in the terminal
- Save a combined visualization panel to `plots/all_plots_combined.png`

---

## 📈 Visualizations Generated

The script generates a **2×3 dashboard** with the following plots:

1. Distribution of G3 (Final Grade)
2. Correlation Heatmap
3. G1 vs G2 vs G3 scatter comparison
4. Actual vs Predicted G3
5. Feature Coefficients bar chart
6. Model Summary text box

---

## 🔑 Key Finding

**G2 (Second Period Grade)** is by far the strongest predictor of G3 with a coefficient of **0.980**, nearly 7x higher than G1 (0.144). `failures` has a strong negative impact (-0.456), meaning even one past failure significantly pulls the final grade down.

---

## 🛠️ Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---

## 👨‍💻 Author

**Avinash** — TY Computer Engineering, Sem 6  
Shah & Anchor Kutchhi Engineering College, Mumbai
