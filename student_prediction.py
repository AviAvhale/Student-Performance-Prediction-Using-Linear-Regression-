# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Project Title : Student Performance Prediction Using Linear Regression      ║
# ║  Subject       : Data Warehousing & Mining (DWM) — CIAP Phase 1              ║
# ║  Assignment    : Individual Assignment                                       ║
# ║  Dataset       : Student Performance Dataset (student-mat.csv)               ║
# ║  Source        : UCI Machine Learning Repository / GitHub                    ║
# ║  Separator     : Semicolon (;)                                               ║
# ║  Target        : G3 (Final Grade, scale 0–20)                                ║
# ║  Algorithm     : Linear Regression (scikit-learn)                            ║
# ║  Libraries     : pandas, numpy, matplotlib, seaborn, scikit-learn            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Import Libraries ─────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend first for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
import os
import sys

# Fix Unicode output on Windows (CP1252 terminals can't print emoji)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ── Helper: Print a bordered section header ──────────────────────────────────
def print_header(title):
    """Print a professionally bordered section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title):
    """Print a sub-section header."""
    print(f"\n── {title} {'─' * max(1, 60 - len(title))}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load & Explore Data
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 1: Load & Explore Dataset")

DATASET_URL = "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv"

try:
    df = pd.read_csv(DATASET_URL, sep=";")
    print(f"\n  ✅ Dataset loaded successfully from URL.")
except Exception as e:
    print(f"\n  ❌ Failed to load dataset from URL: {e}")
    print("  💡 Tip: Check your internet connection or download the file manually.")
    print(f"  📎 URL: {DATASET_URL}")
    sys.exit(1)

# Print shape
print_subheader("Dataset Shape")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")

# Print first 5 rows
print_subheader("First 5 Rows")
print(df.head().to_string(index=False))

# Print column names
print_subheader("Column Names")
cols_per_line = 6
all_cols = list(df.columns)
for i in range(0, len(all_cols), cols_per_line):
    chunk = all_cols[i : i + cols_per_line]
    print("  " + ", ".join(chunk))

# Basic statistics
print_subheader("Basic Statistics (Numerical Columns)")
print(df.describe().round(2).to_string())

# Missing values
print_subheader("Missing Values per Column")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅ No missing values found in any column!")
else:
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} missing")

print(f"\n  ✅ Step 1 Complete — Dataset loaded and explored.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Feature Selection
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 2: Feature Selection")

FEATURES = ["G1", "G2", "studytime", "absences", "failures"]
TARGET = "G3"

X = df[FEATURES]
y = df[TARGET]

print(f"\n  Selected Features : {FEATURES}")
print(f"  Target Variable   : {TARGET}")

print_subheader("Sample Feature Values (First 5 Rows)")
sample_df = df[FEATURES + [TARGET]].head()
print(sample_df.to_string(index=False))

print(f"\n  ✅ Step 2 Complete — Features and target selected.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Prepare Visualization Data (plots created after training)
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 3: Data Visualizations (will be generated after training)")

# Create output directory for plots
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Pre-compute correlation matrix for later use
corr_cols = FEATURES + [TARGET]
corr_matrix = df[corr_cols].corr()

print(f"\n  ℹ️  All 5 plots will be generated as ONE combined panel after model training.")
print(f"  ✅ Step 3 — Visualization data prepared.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Train-Test Split
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 4: Train-Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n  Split Ratio       : 80% Train / 20% Test")
print(f"  Random State      : 42")
print(f"  Training Samples  : {X_train.shape[0]}")
print(f"  Testing Samples   : {X_test.shape[0]}")
print(f"  Total Samples     : {X_train.shape[0] + X_test.shape[0]}")

print(f"\n  ✅ Step 4 Complete — Data split into training and testing sets.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Train Linear Regression Model
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 5: Train Linear Regression Model")

model = LinearRegression()
model.fit(X_train, y_train)

intercept = model.intercept_
coefficients = model.coef_

# Print the model equation
print_subheader("Model Equation")
equation_parts = [f"{intercept:.4f}"]
for feat, coef in zip(FEATURES, coefficients):
    sign = "+" if coef >= 0 else "-"
    equation_parts.append(f" {sign} {abs(coef):.4f} × {feat}")
equation = "G3 = " + "".join(equation_parts)
print(f"  {equation}")

# Print coefficients table
print_subheader("Feature Coefficients")
print(f"  {'Feature':<15} {'Coefficient':>15} {'Impact':>10}")
print(f"  {'─' * 15} {'─' * 15} {'─' * 10}")
for feat, coef in zip(FEATURES, coefficients):
    impact = "Positive" if coef > 0 else "Negative"
    print(f"  {feat:<15} {coef:>15.4f} {impact:>10}")
print(f"\n  Intercept: {intercept:.4f}")

print(f"\n  ✅ Step 5 Complete — Linear Regression model trained.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Predictions
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 6: Predictions on Test Set")

y_pred = model.predict(X_test)

# Print comparison table — first 10 rows
print_subheader("Prediction Comparison (First 10 Samples)")
print(f"  {'#':<5} {'Actual G3':>12} {'Predicted G3':>14} {'Error':>10}")
print(f"  {'─' * 5} {'─' * 12} {'─' * 14} {'─' * 10}")
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = actual - predicted
    print(f"  {i + 1:<5} {actual:>12} {predicted:>14.2f} {error:>10.2f}")

print(f"\n  ✅ Step 6 Complete — Predictions generated for test set.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 (continued): ALL 5 Visualizations — Single Combined Panel
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 3 (continued): All Visualizations — Combined Panel")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Student Performance Prediction — Complete Visualization Dashboard",
             fontsize=16, fontweight="bold", y=1.02)

# ── Subplot 1: Correlation Heatmap ───────────────────────────────────────────
ax1 = axes[0, 0]
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax1,
)
ax1.set_title("1. Correlation Heatmap", fontsize=11, fontweight="bold")

# ── Subplot 2: Histogram of G3 Distribution ──────────────────────────────────
ax2 = axes[0, 1]
sns.histplot(df["G3"], bins=20, kde=True, color="steelblue", edgecolor="black", ax=ax2)
ax2.axvline(df["G3"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {df['G3'].mean():.2f}")
ax2.set_xlabel("G3 (Final Grade)", fontsize=10)
ax2.set_ylabel("Frequency", fontsize=10)
ax2.set_title("2. Distribution of G3 (Final Grade)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)

# ── Subplot 3: Scatter Plot — G2 vs G3 ───────────────────────────────────────
ax3 = axes[0, 2]
sns.scatterplot(x=df["G2"], y=df["G3"], alpha=0.6, edgecolor="black", s=30, color="dodgerblue", ax=ax3)
z = np.polyfit(df["G2"], df["G3"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["G2"].min(), df["G2"].max(), 100)
ax3.plot(x_line, p(x_line), color="red", linewidth=2, linestyle="--", label="Trend Line")
ax3.set_xlabel("G2 (Second Period Grade)", fontsize=10)
ax3.set_ylabel("G3 (Final Grade)", fontsize=10)
ax3.set_title("3. Scatter Plot — G2 vs G3", fontsize=11, fontweight="bold")
ax3.legend(fontsize=8)

# ── Subplot 4: Actual vs Predicted G3 ────────────────────────────────────────
ax4 = axes[1, 0]
ax4.scatter(y_test, y_pred, alpha=0.6, edgecolor="black", s=30, color="dodgerblue", label="Predictions")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax4.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2, linestyle="--", label="Perfect Prediction")
ax4.set_xlabel("Actual G3", fontsize=10)
ax4.set_ylabel("Predicted G3", fontsize=10)
ax4.set_title("4. Actual vs Predicted G3", fontsize=11, fontweight="bold")
ax4.legend(fontsize=8)

# ── Subplot 5: Feature Coefficients Bar Chart ────────────────────────────────
ax5 = axes[1, 1]
colors = ["green" if c >= 0 else "red" for c in coefficients]
bars = ax5.barh(FEATURES, coefficients, color=colors, edgecolor="black", height=0.5)
for bar, coef in zip(bars, coefficients):
    x_pos = coef + 0.02 if coef >= 0 else coef - 0.02
    ha = "left" if coef >= 0 else "right"
    ax5.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{coef:.3f}",
             va="center", ha=ha, fontsize=9, fontweight="bold")
ax5.set_xlabel("Coefficient Value", fontsize=10)
ax5.set_ylabel("Feature", fontsize=10)
ax5.set_title("5. Feature Coefficients", fontsize=11, fontweight="bold")
ax5.axvline(x=0, color="black", linewidth=0.8, linestyle="-")

# ── Subplot 6: Summary Text Box ─────────────────────────────────────────────
ax6 = axes[1, 2]
ax6.axis("off")
summary_text = (
    f"MODEL SUMMARY\n"
    f"{'─' * 30}\n"
    f"Algorithm   : Linear Regression\n"
    f"Features    : {', '.join(FEATURES)}\n"
    f"Target      : G3 (Final Grade)\n"
    f"Train/Test  : 80% / 20%\n"
    f"{'─' * 30}\n"
    f"R² Score    : {r2_score(y_test, y_pred):.4f}\n"
    f"MAE         : {mean_absolute_error(y_test, y_pred):.4f}\n"
    f"RMSE        : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\n"
    f"{'─' * 30}\n"
    f"Key Finding:\n"
    f"  'G2' is the strongest\n"
    f"  predictor of G3"
)
ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
         fontsize=11, fontfamily="monospace", verticalalignment="center",
         horizontalalignment="center",
         bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", edgecolor="black", linewidth=1.5))

plt.tight_layout()
combined_path = os.path.join(PLOT_DIR, "all_plots_combined.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  ✅ All 5 plots saved as ONE combined panel: {combined_path}")
print(f"  ✅ Step 3 Complete — All visualizations generated.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 7: Model Evaluation")

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print_subheader("Evaluation Metrics")
print(f"  {'Metric':<25} {'Value':>10}")
print(f"  {'─' * 25} {'─' * 10}")
print(f"  {'R² Score':<25} {r2:>10.4f}")
print(f"  {'Mean Absolute Error':<25} {mae:>10.4f}")
print(f"  {'Root Mean Squared Error':<25} {rmse:>10.4f}")

print_subheader("Interpretation")
print(f"  📊 R² Score ({r2:.4f}):")
print(f"     → The model explains {r2 * 100:.2f}% of the variance in G3 (final grade).")
print(f"     → Closer to 1.0 is better; 1.0 means perfect prediction.\n")

print(f"  📏 MAE ({mae:.4f}):")
print(f"     → On average, predictions are off by {mae:.2f} grade points.")
print(f"     → Lower is better; 0 means perfect accuracy.\n")

print(f"  📐 RMSE ({rmse:.4f}):")
print(f"     → Penalizes larger errors more heavily than MAE.")
print(f"     → Lower is better; useful for detecting outlier predictions.")

# Model rating
print_subheader("Model Rating")
if r2 >= 0.85:
    rating = "🌟 Excellent"
    emoji = "🎯"
elif r2 >= 0.70:
    rating = "👍 Good"
    emoji = "✅"
else:
    rating = "⚠️  Needs Improvement"
    emoji = "🔧"

print(f"  {emoji} Rating: {rating} (R² = {r2:.4f})")

print(f"\n  ✅ Step 7 Complete — Model evaluated.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Predict for a New Student
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 8: Predict for a New Student")

# Hardcoded sample student
new_student = {
    "G1": 12,
    "G2": 13,
    "studytime": 2,
    "absences": 3,
    "failures": 0,
}

print_subheader("New Student Profile")
for key, value in new_student.items():
    print(f"  {key:<15} : {value}")

# Create DataFrame for prediction
new_student_df = pd.DataFrame([new_student])
raw_prediction = model.predict(new_student_df)[0]

# Clamp the result between 0 and 20
clamped_prediction = np.clip(raw_prediction, 0, 20)

print_subheader("Prediction Result")
print(f"  Raw Predicted G3    : {raw_prediction:.2f}")
print(f"  Clamped G3 (0–20)  : {clamped_prediction:.2f}")

# Interpret the grade
if clamped_prediction >= 16:
    grade_label = "Excellent"
elif clamped_prediction >= 14:
    grade_label = "Good"
elif clamped_prediction >= 12:
    grade_label = "Satisfactory"
elif clamped_prediction >= 10:
    grade_label = "Sufficient"
else:
    grade_label = "Fail"

print(f"  Grade Interpretation: {grade_label}")

print(f"\n  ✅ Step 8 Complete — New student prediction done.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Final Summary Block
# ══════════════════════════════════════════════════════════════════════════════
print_header("STEP 9: Final Summary")

# Find the feature with highest absolute coefficient
max_coef_idx = np.argmax(np.abs(coefficients))
max_coef_feature = FEATURES[max_coef_idx]
max_coef_value = coefficients[max_coef_idx]

summary_width = 70
print("\n" + "╔" + "═" * (summary_width - 2) + "╗")
print("║" + " STUDENT PERFORMANCE PREDICTION — FINAL SUMMARY ".center(summary_width - 2) + "║")
print("╠" + "═" * (summary_width - 2) + "╣")
print("║" + f"  Dataset Size       : {df.shape[0]} students, {df.shape[1]} features".ljust(summary_width - 2) + "║")
print("║" + f"  Algorithm          : Linear Regression (OLS)".ljust(summary_width - 2) + "║")
print("║" + f"  Features Used      : {', '.join(FEATURES)}".ljust(summary_width - 2) + "║")
print("║" + f"  Target Variable    : G3 (Final Grade, 0–20)".ljust(summary_width - 2) + "║")
print("║" + f"  Train/Test Split   : 80% / 20% (random_state=42)".ljust(summary_width - 2) + "║")
print("╠" + "═" * (summary_width - 2) + "╣")
print("║" + f"  R² Score           : {r2:.4f}".ljust(summary_width - 2) + "║")
print("║" + f"  MAE                : {mae:.4f}".ljust(summary_width - 2) + "║")
print("║" + f"  RMSE               : {rmse:.4f}".ljust(summary_width - 2) + "║")
print("║" + f"  Model Rating       : {rating}".ljust(summary_width - 2) + "║")
print("╠" + "═" * (summary_width - 2) + "╣")
print("║" + f"  🔑 Key Finding:".ljust(summary_width - 2) + "║")
print("║" + f"  '{max_coef_feature}' has the highest coefficient ({max_coef_value:.4f}),".ljust(summary_width - 2) + "║")
print("║" + f"  making it the strongest predictor of G3.".ljust(summary_width - 2) + "║")
print("╚" + "═" * (summary_width - 2) + "╝")

print(f"\n  ✅ Step 9 Complete — Summary generated.")
print(f"\n  📁 All plots saved in: ./{PLOT_DIR}/")
print(f"  📊 Combined panel: {combined_path}")

# ── Show the combined plot (switch to interactive backend) ────────────────────
try:
    matplotlib.use("TkAgg")
    import importlib
    importlib.reload(plt)
    print("\n" + "=" * 70)
    print("  Displaying combined plot panel... Close the window when done.")
    print("=" * 70)

    img = plt.imread(combined_path)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Student Performance Prediction — All Visualizations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
except Exception:
    print(f"\n  ℹ️  Interactive display not available. Combined panel saved at: {combined_path}")

print("\n" + "=" * 70)
print("  ✅ ALL STEPS COMPLETE — Assignment Ready for Submission!")
print("=" * 70 + "\n")
