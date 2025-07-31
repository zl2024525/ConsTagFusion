import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_score,
    f1_score,
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import os
from scipy import stats


plt.rcParams["axes.unicode_minus"] = False

PICTURES_DIR = ""
TABLES_DIR = ""
os.makedirs(PICTURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def load_data(file_path):
    """Load credit risk data in Excel format"""
    try:
        df = pd.read_excel(file_path)
        print(
            f"Data loaded successfully, total {df.shape[0]} rows, {df.shape[1]} columns"
        )
        return df
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None

# Data preprocessing (1 = overdue (positive sample), 0 = not overdue (negative sample))
def preprocess_data(df):
    """Preprocess credit risk data, including feature encoding and binning"""
    processed_df = df.copy()

    # 1. Process Age feature
    age_bins = [18, 29, 40, 51, 62, 74]
    age_labels = [1, 2, 3, 4, 5]
    processed_df["Age_Group"] = pd.cut(
        processed_df["Age"], bins=age_bins, labels=age_labels, right=False
    )

    # 2. Process Gender feature
    processed_df["Gender"] = (processed_df["Gender"] == "Female").astype(int)

    # 3. Process City feature
    city_mapping = {}

    city_level_mapping = {
        "First-tier cities": 1,
        "New first-tier cities": 2,
        "Second-tier cities": 3,
        "Third-tier cities": 4,
        "Fourth-tier cities": 5,
        "Fifth-tier cities": 6,
    }

    city_to_level = {}
    for level, cities in city_mapping.items():
        for city in cities:
            city_to_level[city] = city_level_mapping[level]

    processed_df["City_Level"] = processed_df["City"].map(city_to_level)

    # 4. Process Education feature
    education_mapping = {
        "Uncertain education": 0,
        "Doctorate": 7,
        "Master's": 6,
        "Bachelor's": 5,
        "Associate's": 4,
        "High school": 3,
        "Junior high school": 2,
        "Primary school": 1,
    }
    processed_df["Education"] = processed_df["Education"].map(
        education_mapping
    )

    # 5. Process Income feature
    salary_bins = [2142, 4421, 6700, 8979, 11258, 13535]
    salary_labels = [1, 2, 3, 4, 5]
    processed_df["Salary"] = pd.cut(
        processed_df["Salary"],
        bins=salary_bins,
        labels=salary_labels,
        right=False,
    )

    # 6. Process Housing feature
    housing_mapping = {"Uncertain home ownership": 0, "Owns property": 1}
    processed_df["Housing_Flag"] = processed_df["Housing_flag"].map(housing_mapping)

    # 7. Process Wealth feature
    capital_bins = [8, 26, 44, 62, 80, 99]
    capital_labels = [1, 2, 3, 4, 5]  # Corresponding to low to high assets
    processed_df["Capital_Group"] = pd.cut(
        processed_df["Capital_score"],
        bins=capital_bins,
        labels=capital_labels,
        right=False,
    )

    # 8. Process CreditLimit feature
    credit_bins = [0, 10000, 50000, 100000, float("inf")]
    credit_labels = [1, 2, 3, 4]  # Corresponding to ordinary to diamond cards
    processed_df["Credit_Card_Level"] = pd.cut(
        processed_df["Credit_limit_original"],
        bins=credit_bins,
        labels=credit_labels,
        right=False,
    )

    # 9. Process target variable Default
    processed_df["Default_Flag"] = processed_df["Default_flag_6_months"]

    # Convert binned features to numeric type
    for col in ["Age_Group", "Salary_Group", "Capital_Group", "Credit_Card_Level"]:
        processed_df[col] = processed_df[col].astype(float)

    return processed_df

# Feature engineering
def feature_engineering(df):
    """Feature engineering: select original features only"""
    selected_features = [
        "Age_Group",
        "Gender",
        "City_Level",
        "Education_Level",
        "Salary_Group",
        "Housing_Flag",
        "Capital_Group",
        "Credit_Card_Level",
        "Credit_limit_usage",
    ]
    target = "Default_Flag"
    return df[selected_features], df[target]


def split_data(X, y, random_state=42):
    """Split data into train(70%)/val(10%)/test(20%) sets with stratified sampling"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.125,
        random_state=random_state,
        stratify=y_train_val,
    )

    def count_samples(y, name):
        pos = sum(y == 1)  # Positive samples: overdue
        neg = len(y) - pos  # Negative samples: not overdue
        return {
            "name": name,
            "total": len(y),
            "positive": pos,
            "negative": neg,
            "positive_ratio": pos / len(y),
        }

    # Statistics for each dataset
    train_stats = count_samples(y_train, "Training set")
    val_stats = count_samples(y_val, "Validation set")
    test_stats = count_samples(y_test, "Test set")


# Model construction
def build_models(pos_weight):
    """Build multiple machine learning models"""
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            solver="liblinear",
            class_weight={1: pos_weight, 0: 1},
            max_iter=1000,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=4,
            min_samples_split=5,
            class_weight={1: pos_weight, 0: 1},
            n_jobs=-1,
        ),
        "Gradient Boosting Tree": GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,
            subsample=0.8,
            learning_rate=0.05,
            max_depth=5,
        ),
        "XGBoost": xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            reg_lambda=5,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
        ),
        "CatBoost": cb.CatBoostClassifier(
            random_state=42,
            iterations=100,
            learning_rate=0.05,
            depth=3,
            l2_leaf_reg=5,
            scale_pos_weight=pos_weight,
            verbose=0,
            eval_metric="AUC",
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=42, n_estimators=100, learning_rate=0.1
        ),
        "SVM": SVC(
            random_state=42,
            probability=True,
            class_weight={1: pos_weight, 0: 1},
            kernel="rbf",
            gamma="scale",
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform", n_jobs=-1),
    }
    return models


# Model training and evaluation
def train_and_evaluate(models, X_train, X_val, y_train, y_val):
    results = {}

    for name, model in models.items():

        # Train the model
        if name == "CatBoost":
            model.fit(
                X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10
            )
        else:
            model.fit(X_train, y_train)

        y_train_pred_proba = model.predict_proba(X_train)[:, 1]

        if name == "Gradient Boosting Tree":
            y_train_pred = (y_train_pred_proba >= 0.3).astype(int)
        else:
            y_train_pred = model.predict(X_train)

        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        train_recall = recall_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_pred_proba)
        train_ks = max(train_tpr - train_fpr)

        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        if name == "Gradient Boosting Tree":
            y_val_pred = (y_val_pred_proba >= 0.3).astype(int)
        else:
            y_val_pred = model.predict(X_val)

        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        val_recall = recall_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_fpr, val_tpr, _ = roc_curve(y_val, y_val_pred_proba)
        val_ks = max(val_tpr - val_fpr)

        results[name] = {
            "model": model,
            "train_auc": train_auc,
            "train_recall": train_recall,
            "train_precision": train_precision,
            "train_f1": train_f1,
            "train_ks": train_ks,
            "val_auc": val_auc,
            "val_recall": val_recall,
            "val_precision": val_precision,
            "val_f1": val_f1,
            "val_ks": val_ks,
            "y_val_pred_proba": y_val_pred_proba,
        }

    return results


# Model cross-validation
def cross_validate_models(models, X, y, cv=5):
    """Perform cross-validation for models"""
    print("\nStarting cross-validation...")
    cv_results = {}

    for name, model in models.items():
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        if name == "SVM":

            scoring = "roc_auc"
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        else:
            scoring = "roc_auc"
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

        cv_results[name] = {
            "mean_auc": cv_scores.mean(),
            "std_auc": cv_scores.std(),
            "auc_scores": cv_scores,
        }

        print(
            f"{name} cross-validation results: AUC mean = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}"
        )
        print(f"Fold scores: {[round(score, 4) for score in cv_scores]}")

    return cv_results


def plot_learning_curves(models, X_train, y_train, cv=5):
    """Plot learning curves for all models to check overfitting"""
    for name, model in models.items():
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)


def analyze_feature_importance(models, X_train, feature_names):
    """Analyze and visualize feature importance"""
    print("\nFeature importance analysis (predicting default):")

    for name, model in models.items():
        plt.figure(figsize=(10, 6))

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        elif name == "LightGBM":
            importance = model.feature_importances_
        elif name == "CatBoost":
            importance = model.get_feature_importance()
        else:
            print(f"{name} does not support feature importance analysis")
            continue

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values("Importance", ascending=False)

        sns.barplot(x="Importance", y="Feature", data=importance_df)

        print(f"\n{name} feature importance:")
        print(importance_df)

    # SHAP analysis for XGBoost
    if "XGBoost" in models:
        feature_name_mapping = {
            "Age_Group": "Age_Group",
            "Gender": "Gender",
            "City_Level": "City_Level",
            "Education_Level": "Education_Level",
            "Salary_Group": "Salary_Group",
            "Housing_Flag": "Housing_Flag",
            "Capital_Group": "Capital_Group",
            "Credit_Card_Level": "Credit_Card_Level",
            "Credit_limit_usage": "Credit_limit_usage",
        }
        mapped_features = [feature_name_mapping.get(f, f) for f in feature_names]

        explainer = shap.TreeExplainer(models["XGBoost"])


# Test set evaluation
def evaluate_all_models_on_test(models, X_test, y_test, cv_results, val_results):
    """Evaluate all models on test set with threshold adjustment for GBT"""
    test_results = {}
    print("\n" + "=" * 50)
    print("Model Evaluation Results Summary (Train/Validation/Test Set)")
    print("=" * 50)

    for name, model in models.items():
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        if name == "Gradient Boosting Tree":
            y_test_pred = (y_test_pred_proba >= 0.3).astype(int)
        else:
            y_test_pred = model.predict(X_test)

        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_recall = recall_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_pred_proba)
        test_ks = max(test_tpr - test_fpr)

        # Overfitting test
        cv_scores = cv_results[name]["auc_scores"]
        t_stat, p_value = stats.ttest_1samp(cv_scores, test_auc)

        test_results[name] = {
            "auc": test_auc,
            "recall": test_recall,
            "precision": test_precision,
            "f1": test_f1,
            "ks": test_ks,
            "t_stat": t_stat,
            "p_value": p_value,
            "cv_mean": cv_scores.mean(),
        }

        print(f"\n{name} Model Evaluation Results:")
        print("-" * 50)
        print(
            f"Training Set: AUC={val_results[name]['train_auc']:.4f}, Default Recall={val_results[name]['train_recall']:.4f}, Default Precision={val_results[name]['train_precision']:.4f}, F1={val_results[name]['train_f1']:.4f}, KS={val_results[name]['train_ks']:.4f}"
        )
        print(
            f"Validation Set: AUC={val_results[name]['val_auc']:.4f}, Default Recall={val_results[name]['val_recall']:.4f}, Default Precision={val_results[name]['val_precision']:.4f}, F1={val_results[name]['val_f1']:.4f}, KS={val_results[name]['val_ks']:.4f}"
        )
        print(
            f"Test Set: AUC={test_auc:.4f}, Default Recall={test_recall:.4f}, Default Precision={test_precision:.4f}, F1={test_f1:.4f}, KS={test_ks:.4f}"
        )
        print(
            f"Overfitting Test: t-statistic={t_stat:.4f}, p-value={p_value:.4f}, Difference is {'' if p_value < 0.05 else 'not '}significant"
        )

    return test_results


def main():
    file_path = ""
    df = load_data(file_path)

    if df is None:
        print("Data loading failed, program exits.")
        return

    processed_df = preprocess_data(df)

    X, y = feature_engineering(processed_df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_stats,
        val_stats,
        test_stats,
    ) = split_data(X, y)


    positive_count = train_stats["positive"]
    negative_count = train_stats["negative"]
    pos_weight = negative_count / positive_count
    print(
        f"\nClass distribution: Positive(default)={positive_count}, Negative(non-default)={negative_count}, Weight ratio={pos_weight:.4f}"
    )

    models = build_models(pos_weight)
    cv_results = cross_validate_models(models, X, y)
    val_results = train_and_evaluate(models, X_train, X_val, y_train, y_val)
    trained_models = {name: val_results[name]["model"] for name in val_results}
    analyze_feature_importance(trained_models, X_train, X.columns.tolist())
    test_results = evaluate_all_models_on_test(
        trained_models, X_test, y_test, cv_results, val_results
    )

    result_df = pd.DataFrame(
        {
            "Model": list(val_results.keys()),
            "Train AUC": [val_results[name]["train_auc"] for name in val_results],
            "Train Default Recall": [
                val_results[name]["train_recall"] for name in val_results
            ],
            "Train Default Precision": [
                val_results[name]["train_precision"] for name in val_results
            ],
            "Train F1": [val_results[name]["train_f1"] for name in val_results],
            "Train KS": [val_results[name]["train_ks"] for name in val_results],
            "Validation AUC": [val_results[name]["val_auc"] for name in val_results],
            "Validation Default Recall": [
                val_results[name]["val_recall"] for name in val_results
            ],
            "Validation Default Precision": [
                val_results[name]["val_precision"] for name in val_results
            ],
            "Validation F1": [val_results[name]["val_f1"] for name in val_results],
            "Validation KS": [val_results[name]["val_ks"] for name in val_results],
            "CV AUC Mean": [cv_results[name]["mean_auc"] for name in val_results],
            "Test AUC": [test_results[name]["auc"] for name in val_results],
            "Test Default Recall": [
                test_results[name]["recall"] for name in val_results
            ],
            "Test Default Precision": [
                test_results[name]["precision"] for name in val_results
            ],
            "Test F1": [test_results[name]["f1"] for name in val_results],
            "Test KS": [test_results[name]["ks"] for name in val_results],
            "Overfitting Test p-value": [
                test_results[name]["p_value"] for name in val_results
            ],
        }
    )

    table_path = os.path.join(TABLES_DIR, "")
    result_df.to_csv(table_path, index=False)


if __name__ == "__main__":
    main()
