import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv("data/loan_default_prediction_dataset.csv")

# print(df.head())

# print(df.shape)

# print(df.info())

# print(df.describe())

# print(df.isna().sum())

# print(df.duplicated().sum())

print("\nLoan Default Count: ")
print(
    df["Loan_Default"].value_counts()
)

print("\nLoan Default Percentage: ")
print(
    (
        df["Loan_Default"].value_counts(
            normalize=True
        ) * 100
    ).round(2)
)

# sns.countplot(
#     x= "Loan_Default",
#     data=df,
#     edgecolor="gray",
#     palette="Set3"
# )

# plt.title("Loan Default Distribution")
# plt.xlabel("Loan Default")
# plt.ylabel("Count")
# plt.savefig(
#     "images/loan-default-distribution.png",
#     dpi=350
# )
# plt.show()

# df.hist(
#     figsize=(12, 8),
#     bins=30,
#     color="royalblue",
#     edgecolor="black",
#     grid=False
# )

# plt.tight_layout()
# plt.show()

corr = df.corr(
    numeric_only=True
)

print("\nCorrelation with Target: ")
print(
    corr["Loan_Default"].sort_values(
        ascending=False
    )
)

# sns.heatmap(
#     corr,
#     cmap="YlGnBu"
# )

# plt.title("Correlation Heatmap")
# plt.tight_layout()
# plt.savefig(
#     "images/correlation-heatmap.png",
#     dpi=350
# )
# plt.show()

# cat_cols = df.select_dtypes(
#     include="object"
# ).columns.tolist()

# for col in cat_cols:
    
#     sns.countplot(
#         x=col,
#         hue="Loan_Default",
#         data=df,
#         palette="Set3",
#         edgecolor="gray"
#     )
    
#     plt.title(f"{col} vs Loan Default")
#     plt.show()

# imp_num_cols = [
#     "Credit_Score",
#     "Annual_Income",
#     "Monthly_Income",
#     "Loan_Amount",
#     "Debt",
#     "Interest_Rate",
#     "Debt_to_Income",
#     "Loan_to_Income",
#     "Savings",
#     "Assets"
# ]

# for col in imp_num_cols:
    
#     sns.boxplot(
#         x="Loan_Default",
#         y=col,
#         data=df
#     )
    
#     plt.title(f"{col} vs Loan Default")   
#     plt.show()
    

X = df.drop(
    columns=["Customer_ID", "Loan_Default"]
)
y = df["Loan_Default"]

print("\nX Shape: ", X.shape)
print("Y Shape: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTraining Data: ")
print("X_train Shape: ", X_train.shape)
print("y_train Shape: ", y_train.shape)

print("\nTesting Data: ")
print("X_test Shape: ", X_test.shape)
print("y_test Shape: ", y_test.shape)

print("\nTraining Target Distribution: ")
print(
    y_train.value_counts(
        normalize=True
    ).mul(100).round(2)
)

print("\nTesting Target Distribution: ")
print(
    y_test.value_counts(
        normalize=True
    ).mul(100).round(2)
)


cat_features = X.select_dtypes(
    include="object"
).columns.tolist()

num_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


cat_pipe = Pipeline([
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("Encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_pipe = Pipeline([
    ("Imputer", SimpleImputer(strategy="median")),
    ("Scaler", StandardScaler())
])


preprocessor = ColumnTransformer([
    ("cat", cat_pipe, cat_features),
    ("num", num_pipe, num_features)
])


lr_pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

lr_param_grid = {
    "model__class_weight": [
        None,
        "balanced"
    ],
    "model__C": [
        0.01,
        0.1,
        0.5,
        1,
        10
    ]
}

lr_grid = GridSearchCV(
    estimator=lr_pipe,
    param_grid=lr_param_grid,
    cv=5,
    n_jobs=1,
    scoring="f1"
)


rf_pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    ))
])


print("\nTraining Logistic Regression.....")

lr_grid.fit(
    X_train,
    y_train
)

lr_pipe = lr_grid.best_estimator_

print("\nBest Parameters: ")
print(lr_grid.best_params_)

print("\nBest F1 CV Score: ")
print(round(lr_grid.best_score_, 3))

print("\nLogistic Regression Training Complete")

print("\nTraining Random Forest.....")

rf_pipe.fit(
    X_train,
    y_train
)

print("\nRandom Forest Training Complete")


lr_pred = lr_pipe.predict(X_test)
rf_pred = rf_pipe.predict(X_test)

lr_prob = lr_pipe.predict_proba(X_test)[:,1]
rf_prob = rf_pipe.predict_proba(X_test)[:,1]


def evaluate_model(name, y_pred, y_prob):
    
    print(f"\n{name}: ")
    print("-" * 40)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    
evaluate_model(
    "Logistic Regression",
    lr_pred,
    lr_prob
)

evaluate_model(
    "Random Forest",
    rf_pred,
    rf_prob
)


lr_cm = confusion_matrix(
    y_test,
    lr_pred
)
print("\nLogistic Regression Confusion Matrix: ")
print(lr_cm)

# ConfusionMatrixDisplay(
#     confusion_matrix=lr_cm,
#     display_labels=["No Default", "Default"]
# ).plot()

# plt.title("Logistic Regression - Confusion Matrix")
# plt.tight_layout()
# plt.savefig(
#     "images/logistic-regression-confusion-matrix.png",
#     dpi=350
# )
# plt.show()

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)
print("\nRandom Forest Confusion Matrix: ")
print(rf_cm)

# ConfusionMatrixDisplay(
#     confusion_matrix=rf_cm,
#     display_labels=["No Default", "Default"]
# ).plot()

# plt.title("Random Forest - Confusion Matrix")
# plt.tight_layout()
# plt.savefig(
#     "images/random-forest-confusion-matrix.png",
#     dpi=350
# )
# plt.show()


print("\nLogistic Regression Classification Report: ")
print(
    classification_report(
        y_test,
        lr_pred,
        target_names=["No Default", "Default"]
    )
)

print("\nRandom Forest Classification Report: ")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=["No Default", "Default"] 
    )
)


lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    lr_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)

lr_auc = roc_auc_score(
    y_test,
    lr_prob
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

# plt.plot(
#     lr_fpr,
#     lr_tpr,
#     label = f"Logistic Regression (AUC = {lr_auc:.3f})"
# )    

# plt.plot(
#     rf_fpr,
#     rf_tpr,
#     label = f"Random Forest (AUC = {rf_auc:.3f})"
# )

# plt.plot(
#     [0,1],
#     [0,1],
#     linestyle="dashed",
#     label="Random Classifier",
#     color="gray"
# )

# plt.title("ROC Curve")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positve Rate")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(
#     "images/roc-curve.png",
#     dpi=350
# )
# plt.show()


lr_feature_names = lr_pipe.named_steps[
    "preprocessing"
].get_feature_names_out()

lr_coef = lr_pipe.named_steps[
    "model"
].coef_[0]

lr_coef_df = pd.DataFrame(
    { 
        "Features": lr_feature_names,
        "Coefficients": lr_coef
    }
)

lr_coef_df["Absolute_Coefficients"] = (
    lr_coef_df["Coefficients"].abs()
)

lr_coef_df = lr_coef_df.sort_values(
    by="Absolute_Coefficients",
    ascending=False
)

print("\nTop 20 Feature Coefficients: ")
print(
    lr_coef_df[[
        "Features",
        "Coefficients"
    ]].head(20).set_index("Features")
)


rf_feature_names = rf_pipe.named_steps[
    "preprocessing"
].get_feature_names_out()

rf_imp = rf_pipe.named_steps[
    "model"
].feature_importances_

rf_imp_df = pd.DataFrame(
    {
        "Features": rf_feature_names,
        "Importances": rf_imp
    }
)

rf_imp_df = rf_imp_df.sort_values(
    by="Importances",
    ascending=False
)

print("\nTop 20 Feature Importances: ")
print(
    rf_imp_df.head(20).set_index("Features")
)


# top_feature_imp = rf_imp_df.head(20)

# plt.barh(
#     top_feature_imp["Features"][::-1],
#     top_feature_imp["Importances"][::-1],
#     color="skyblue",
#     edgecolor="gray"
# )

# plt.title("Top 20 Features - Random Forest")
# plt.xlabel("Feature Importances")
# plt.ylabel("Features")
# plt.tight_layout()
# plt.savefig(
#     "images/random-forest-feature-importances.png",
#     dpi=350
# )
# plt.show()


thresholds = np.arange(
    0.05,
    0.95,
    0.05
)

results = []

for threshold in thresholds:
    
    lr_pred_threshold = (
        lr_prob >= threshold
    ).astype(int)
    
    precision = precision_score(
        y_test,
        lr_pred_threshold
    )
    
    recall = recall_score(
        y_test,
        lr_pred_threshold
    )
    
    f1 = f1_score(
        y_test,
        lr_pred_threshold
    )
    
    results.append(
        [
            threshold,
            precision,
            recall,
            f1
        ]
    )
    
threshold_df = pd.DataFrame(
    results,
    columns=[
        "Threshold",
        "Precision",
        "Recall",
        "F1"
    ]
)   
    
print("\nModel Evaluation on Different Thresholds: ")
print(threshold_df)

best_threshold = threshold_df.sort_values(
    by="F1",
    ascending=False
).iloc[0]
    
print("\nBest Threshold: ")
print(best_threshold)


# plt.plot(
#     threshold_df["Threshold"],
#     threshold_df["Precision"],
#     marker="o",
#     label="Precision"
# )
    
# plt.plot(
#     threshold_df["Threshold"],
#     threshold_df["Recall"],
#     marker="o",
#     label="Recall"
# )

# plt.plot(
#     threshold_df["Threshold"],
#     threshold_df["F1"],
#     marker="o",
#     label="F1"
# )

# plt.title("Precision Recall Trade-off")
# plt.xlabel("Threshold")
# plt.ylabel("Score")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(
#     "images/threshold-tuning.png",
#     dpi=350
# )
# plt.show()


optimal_lr_pred = (
    lr_prob >= best_threshold["Threshold"]
).astype(int)

optimal_lr_cm = confusion_matrix(
    y_test,
    optimal_lr_pred
)

print("\nConfusion Matrix After Threshold Tuning: ")
print(optimal_lr_cm)

print("\nClassification Report After Threshold Tuning: ")
print(
    classification_report(
        y_test,
        optimal_lr_pred,
        target_names=[
            "No Default",
            "Default"
        ]
    )
)






