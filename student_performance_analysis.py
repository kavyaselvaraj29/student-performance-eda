# Exploratory Data Analysis and Deep Learning-Based Student Performance Analysis
# Dataset: UCI Student Performance (Mathematics)
# Run in Google Colab. This notebook downloads the dataset, performs EDA,
# creates visualizations, trains an MLP classifier, and evaluates it.

!pip -q install ucimlrepo tensorflow scikit-learn seaborn

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sns.set_theme()
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# 1. Acquire data from the UCI Machine Learning Repository
dataset = fetch_ucirepo(id=320)
X_all = dataset.data.features.copy()
y_all = dataset.data.targets.copy()
df = pd.concat([X_all, y_all], axis=1)

# Select the mathematics dataset and standardize column names.
# UCI returns the full combined dataset for this repository ID in some versions.
# If a subject column exists, retain Mathematics; otherwise use the provided records.
if "subject" in df.columns:
    df = df[df["subject"].astype(str).str.lower().str.contains("math")].copy()

print("Dataset shape:", df.shape)
print("Number of features/fields:", df.shape[1])
display(df.head())
display(df.info())
display(df.describe(include="all").T)
print("Missing values by column:\n", df.isna().sum().sort_values(ascending=False).head(15))
print("Duplicate rows:", df.duplicated().sum())

# If the UCI API returns features and target separately with G3 target name,
# normalize target naming and ensure G3 exists.
if "G3" not in df.columns:
    target_candidates = [c for c in df.columns if str(c).lower() in ("g3", "final grade", "final_grade")]
    if target_candidates:
        df = df.rename(columns={target_candidates[0]: "G3"})
if "G3" not in df.columns:
    raise ValueError("Final grade column G3 was not found. Check the UCI dataset schema.")

# 2. Cleaning and transformation
df = df.drop_duplicates().copy()
# This dataset is documented by UCI as having no missing values. The imputer is retained
# in the model pipeline to make the workflow robust to missing values.
df["G3"] = pd.to_numeric(df["G3"], errors="coerce")
df = df.dropna(subset=["G3"]).copy()

# Derived binary target: pass if final grade >= 10 on the original 0–20 scale.
# G1 and G2 are excluded to avoid using earlier grades that are highly correlated with G3
# and to make the prediction more useful from background/behavioral information.
df["pass"] = (df["G3"] >= 10).astype(int)
drop_cols = ["G3", "pass", "G1", "G2"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["pass"]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split into train, validation and test sets: 70%, 15%, 15%.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1765,
    random_state=RANDOM_STATE, stratify=y_train_full
)
X_train_p = preprocessor.fit_transform(X_train).astype("float32")
X_val_p = preprocessor.transform(X_val).astype("float32")
X_test_p = preprocessor.transform(X_test).astype("float32")

print(f"Train: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print("Encoded input features:", X_train_p.shape[1])
print("Pass distribution:\n", y.value_counts(normalize=True).rename("proportion"))

# 3. EDA visuals (all are generated from the selected dataset)
plt.figure(figsize=(8,5))
sns.histplot(df["G3"], bins=11, kde=True)
plt.title("Distribution of Final Mathematics Grades (G3)")
plt.xlabel("Final grade (0–20)"); plt.ylabel("Number of students")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Final Grade by Weekly Study-Time Category")
plt.xlabel("Study time category (1=lowest, 4=highest)"); plt.ylabel("Final grade")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="absences", y="G3", alpha=.7)
plt.title("School Absences vs Final Grade")
plt.xlabel("Number of absences"); plt.ylabel("Final grade")
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,7))
numeric_corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(numeric_corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout(); plt.show()

plt.figure(figsize=(7,5))
sns.countplot(data=df, x="pass")
plt.title("Pass/Fail Class Distribution")
plt.xlabel("Class (0=Fail, 1=Pass)"); plt.ylabel("Number of students")
plt.tight_layout(); plt.show()

# Additional discrete/probability distribution plot
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="studytime")
plt.title("Frequency Distribution of Weekly Study-Time Categories")
plt.xlabel("Study-time category"); plt.ylabel("Number of students")
plt.tight_layout(); plt.show()

# 4. Statistical summaries and relationships
print("Final grade summary:\n", df["G3"].describe())
print("\nStudy-time category vs mean final grade:\n", df.groupby("studytime", observed=False)["G3"].agg(["count","mean","median","std"]))
print("\nPass rate by study time:\n", df.groupby("studytime", observed=False)["pass"].mean())
print("\nCorrelation of numerical features with G3:\n", numeric_corr["G3"].sort_values(ascending=False))

# 5. Deep Learning MLP model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_p.shape[1],)),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dropout(0.15),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history = model.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=150, batch_size=32, callbacks=[early_stop], verbose=0
)
print("Training stopped after", len(history.history["loss"]), "epochs")

# 6. Evaluation on held-out test set
test_prob = model.predict(X_test_p, verbose=0).ravel()
test_pred = (test_prob >= 0.5).astype(int)
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 4))
print(classification_report(y_test, test_pred, target_names=["Fail", "Pass"], zero_division=0))

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy loss"); plt.legend()
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout(); plt.show()

cm = confusion_matrix(y_test, test_pred, labels=[0,1])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted fail","Predicted pass"],
            yticklabels=["Actual fail","Actual pass"])
plt.title("Confusion Matrix")
plt.tight_layout(); plt.show()

fpr, tpr, thresholds = roc_curve(y_test, test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"MLP (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1], linestyle="--", label="Random baseline")
plt.title("ROC Curve")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.legend(); plt.tight_layout(); plt.show()
print("ROC AUC:", round(roc_auc, 4))

# 7. Basic hyperparameter experiment: compare hidden-layer widths.
# Keep the same preprocessed split; this is a small exploratory comparison.
tuning_results = []
for width in [32, 64, 96]:
    trial = Sequential([
        Dense(width, activation="relu", input_shape=(X_train_p.shape[1],)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    trial.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    trial.fit(X_train_p, y_train, validation_data=(X_val_p, y_val),
              epochs=80, batch_size=32,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)], verbose=0)
    val_loss, val_acc = trial.evaluate(X_val_p, y_val, verbose=0)
    tuning_results.append({"hidden_units": width, "validation_loss": val_loss,
                           "validation_accuracy": val_acc})
tuning_df = pd.DataFrame(tuning_results)
display(tuning_df)
sns.barplot(data=tuning_df, x="hidden_units", y="validation_accuracy")
plt.title("Basic Hyperparameter Comparison")
plt.xlabel("First hidden-layer units"); plt.ylabel("Validation accuracy")
plt.tight_layout(); plt.show()

# 8. Export dataset used and model summary
df.to_csv("student_performance_cleaned.csv", index=False)
model.save("student_performance_mlp.keras")
print("Saved student_performance_cleaned.csv and student_performance_mlp.keras")
