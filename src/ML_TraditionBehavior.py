import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from scipy import stats
from collections import defaultdict

plt.rcParams["axes.unicode_minus"] = False

PICTURES_DIR = ""
TABLES_DIR = ""
os.makedirs(PICTURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully, total {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None


# Calculate tag importance (chi-square test for screening important tags)
def calculate_tag_importance(train_df):
    if 'Consumption_Tags' not in train_df.columns:
        return {}

    def parse_tags(tag_str):
        if pd.isna(tag_str) or tag_str.strip() == "":
            return {}
        tags = tag_str.split(';')
        tag_dict = {}
        for tag in tags:
            if ':' in tag:
                key, val = tag.split(':', 1)
                try:
                    tag_dict[key.strip()] = int(val.strip())
                except:
                    continue
        return tag_dict

    # Parse tags and count occurrences in default/non-default samples
    tag_dicts = train_df['Consumption_Tags'].apply(parse_tags)
    default_flags = train_df['Default_flag_6_months'].values
    total_pos = sum(default_flags)
    total_neg = len(default_flags) - total_pos

    tag_samples = defaultdict(lambda: [0, 0])
    for i, tag_dict in enumerate(tag_dicts):
        is_default = default_flags[i] == 1
        for tag in tag_dict:
            if is_default:
                tag_samples[tag][0] += 1
            else:
                tag_samples[tag][1] += 1

    tag_importance = {}
    print(f"\n===== Chi-square Test Detailed Results =====")
    print(
        f"{'Tag Name':<20} | Occurrences in Default Samples | Occurrences in Non-default Samples | Chi-square Value (Chi2)")
    print("-" * 110)

    for tag, (pos_count, neg_count) in tag_samples.items():
        if pos_count + neg_count < 5:
            continue

        # Construct observation matrix (2x2 contingency table)
        observed = np.array([
            [pos_count, neg_count],  # First row: Samples with this tag (default/non-default)
            [total_pos - pos_count, total_neg - neg_count]  # Second row: Samples without this tag (default/non-default)
        ])

        # Calculate expected matrix
        expected = np.outer(np.sum(observed, axis=1), np.sum(observed, axis=0)) / np.sum(observed)
        expected[expected == 0] = 1e-10  # Avoid division by zero

        # Calculate chi-square value (with continuity correction)
        chi2 = np.sum((np.abs(observed - expected) - 0.5) ** 2 / expected)
        tag_importance[tag] = chi2

        # Print statistics for current tag (limit length to avoid spamming)
        if len(tag_importance) <= 20 or chi2 > 10:  # Print first 20 and tags with large chi-square values
            print(f"{tag[:18]:<20} | {pos_count:<24} | {neg_count:<27} | {chi2:.4f}")

    # Normalize chi-square values
    if tag_importance:
        max_chi2 = max(tag_importance.values())
        for tag in tag_importance:
            tag_importance[tag] /= max_chi2

        # Print filtered high-risk tags (sorted by chi-square value)
        sorted_tags = sorted(tag_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\n===== Filtering Results =====")
        print(f"Total {len(tag_importance)} valid tags identified (after excluding low-frequency tags)")
        print(f"Top 10 High-Risk Tags (Normalized Chi-square Value):")
        for i, (tag, imp) in enumerate(sorted_tags[:10]):
            print(f"{i + 1}. {tag:<18} | Normalized Importance: {imp:.4f}")

    return tag_importance


# Process ConsumptionTags field
def process_consumption_tags(df, tag_importance):
    if 'Consumption_Tags' not in df.columns:
        return df

    def parse_tags(tag_str):
        if pd.isna(tag_str) or tag_str.strip() == "":
            return {}
        tags = tag_str.split(';')
        tag_dict = {}
        for tag in tags:
            if ':' in tag:
                key, val = tag.split(':', 1)
                try:
                    tag_dict[key.strip()] = int(val.strip())
                except:
                    continue
        return tag_dict

    df['tag_dict'] = df['Consumption_Tags'].apply(parse_tags)

    df['tag_count'] = df['tag_dict'].apply(lambda x: len(x))
    df['tag_total_freq'] = df['tag_dict'].apply(lambda x: sum(x.values()) if x else 0)

    # Split important tags into individual features (based on chi-square test results)
    top_tags = sorted(tag_importance.items(), key=lambda x: x[1], reverse=True)[:15]  # Take top 15 important tags
    top_tags = [(tag, imp) for tag, imp in top_tags if imp > 0]

    for tag, imp in top_tags:
        # Generate weighted tag features (weight is the importance from chi-square test)
        df[f'tag_{tag}_weighted'] = df['tag_dict'].apply(
            lambda x: x.get(tag, 0) * imp)  # Tag frequency * importance weight

    # Generate total tag risk score (weighted sum of all tags, representing overall risk)
    def risk_score(tag_dict):
        return sum(tag_dict.get(tag, 0) * imp for tag, imp in top_tags)

    df['tag_risk_score'] = df['tag_dict'].apply(risk_score)

    return df.drop(columns=['tag_dict', 'Consumption_Tags'])


# Data preprocessing
def preprocess_data(df, tag_importance):
    processed_df = df.copy()

    age_bins = [18, 29, 40, 51, 62, 74]
    age_labels = [1, 2, 3, 4, 5]
    processed_df['Age_Group'] = pd.cut(
        processed_df['Age'], bins=age_bins, labels=age_labels, right=False
    ).astype(float).fillna(3.0)  # Fill missing values

    # Gender encoding (Female=1, Male=0)
    processed_df['Gender'] = (processed_df['Gender'] == 'Female').astype(int)

    # City level mapping
    city_mapping = {}

    city_level_mapping = {"First-tier City": 1, "New First-tier City": 2, "Second-tier City": 3, "Third-tier City": 4,
                          "Fourth-tier City": 5, "Fifth-tier City": 6}
    city_to_level = {city: city_level_mapping[level] for level, cities in city_mapping.items() for city in cities}
    processed_df['City_Level'] = processed_df['City'].map(city_to_level).fillna(3.0).astype(float)

    education_mapping = {'Uncertain Education': 0, 'Doctorate': 7, 'Master': 6, 'Bachelor': 5, 'Associate': 4,
                         'High School': 3, 'Junior High': 2, 'Primary School': 1}
    processed_df['Education_Level'] = processed_df['Education_level'].map(education_mapping).fillna(3.0).astype(float)

    salary_bins = [2142, 4421, 6700, 8979, 11258, 13535]
    salary_labels = [1, 2, 3, 4, 5]
    processed_df['Salary_Group'] = pd.cut(
        processed_df['Salary_level'], bins=salary_bins, labels=salary_labels, right=False
    ).astype(float).fillna(3.0)

    housing_mapping = {'Uncertain Homeownership': 0, 'Homeowner': 1}
    processed_df['Housing_Flag'] = processed_df['Housing_flag'].map(housing_mapping).fillna(0.0).astype(float)

    capital_bins = [8, 26, 44, 62, 80, 99]
    capital_labels = [1, 2, 3, 4, 5]
    processed_df['Capital_Group'] = pd.cut(
        processed_df['Capital_score'], bins=capital_bins, labels=capital_labels, right=False
    ).astype(float).fillna(3.0)

    credit_bins = [0, 10000, 50000, 100000, float('inf')]
    credit_labels = [1, 2, 3, 4]
    processed_df['Credit_Card_Level'] = pd.cut(
        processed_df['Credit_limit_original'], bins=credit_bins, labels=credit_labels, right=False
    ).astype(float).fillna(2.0)

    processed_df['Default_Flag'] = processed_df['Default_flag_6_months'].fillna(0).astype(int)

    processed_df = process_consumption_tags(processed_df, tag_importance)

    return processed_df


# Feature engineering (distinguish between base features and tag features)
def feature_engineering(df):
    # Base features
    base_features = [
        'Age_Group', 'Gender', 'City_Level', 'Education_Level',
        'Salary_Group', 'Housing_Flag', 'Capital_Group',
        'Credit_Card_Level', 'Credit_limit_usage'
    ]
    base_features = [f for f in base_features if f in df.columns]

    # Tag features (all split sub-tag features)
    tag_features = [col for col in df.columns if col.startswith('tag_')]

    # All features
    all_features = base_features + tag_features
    target = 'Default_Flag'

    for col in all_features:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    print(f"Base features: {base_features}")
    print(f"Tag sub-features (will be aggregated as Consumption_Tags overall): {tag_features}")
    return df[all_features], df[target], base_features, tag_features  # Return feature groups


def split_data(X, y, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=random_state, stratify=y_train_val
    )

    def count_samples(y, name):
        pos = sum(y == 1)
        neg = len(y) - pos
        return {
            'name': name, 'total': len(y), 'positive': pos,
            'negative': neg, 'positive_ratio': pos / len(y)
        }

    train_stats = count_samples(y_train, "Training set")
    val_stats = count_samples(y_val, "Validation set")
    test_stats = count_samples(y_test, "Test set")

    print("\nData splitting details:")
    for stats in [train_stats, val_stats, test_stats]:
        print(
            f"{stats['name']}: Total samples={stats['total']}, Default samples={stats['positive']}, Non-default samples={stats['negative']}, Default ratio={stats['positive_ratio']:.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test, train_stats, val_stats, test_stats


# Build XGBoost model
def build_xgb_model(pos_weight):
    return xgb.XGBClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        reg_lambda=8,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=20
    )


# Train and evaluate model
def train_and_evaluate_xgb(model, X_train, X_val, y_train, y_val):
    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # Training set evaluation
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    train_recall = recall_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    # Validation set evaluation
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_recall = recall_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    print(
        f"Training set: AUC={train_auc:.4f}, Recall={train_recall:.4f}, Precision={train_precision:.4f}, F1={train_f1:.4f}")
    print(f"Validation set: AUC={val_auc:.4f}, Recall={val_recall:.4f}, Precision={val_precision:.4f}, F1={val_f1:.4f}")

    return {
        'model': model, 'train_auc': train_auc, 'train_recall': train_recall,
        'train_precision': train_precision, 'train_f1': train_f1,
        'val_auc': val_auc, 'val_recall': val_recall,
        'val_precision': val_precision, 'val_f1': val_f1
    }


# Cross-validation
def cross_validate_xgb(model, X, y, cv=5):
    print("\nXGBoost cross-validation...")
    cv_model = xgb.XGBClassifier(
        random_state=model.random_state,
        n_estimators=model.n_estimators,
        learning_rate=model.learning_rate,
        max_depth=model.max_depth,
        reg_lambda=model.reg_lambda,
        subsample=model.subsample,
        colsample_bytree=model.colsample_bytree,
        scale_pos_weight=model.scale_pos_weight,
        eval_metric=model.eval_metric
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        cv_model, X, y,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1
    )

    print(f"Cross-validation results: Mean AUC={cv_scores.mean():.4f}, Std Dev={cv_scores.std():.4f}")
    print(f"Scores per fold: {[round(score, 4) for score in cv_scores]}")
    return {'mean_auc': cv_scores.mean(), 'std_auc': cv_scores.std(), 'auc_scores': cv_scores}


# Analyze feature importance
def analyze_feature_importance(model, feature_names, base_features, tag_features):
    print("\nFeature importance analysis:")

    # Get importance of individual features
    importance = model.feature_importances_
    single_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).set_index('Feature')

    base_importance = single_importance.loc[base_features].reset_index()

    tag_total_importance = single_importance.loc[tag_features]['Importance'].sum()
    tag_overall = pd.DataFrame({
        'Feature': ['Consumption_Tags'],
        'Importance': [tag_total_importance]
    })

    overall_importance = pd.concat([base_importance, tag_overall], ignore_index=True)
    overall_importance = overall_importance.sort_values('Importance', ascending=False)

    # Print overall importance ranking
    print("Overall feature importance ranking:")
    print(overall_importance)

    # (Optional) Print detailed importance of tag sub-features (for granular analysis)
    tag_single_importance = single_importance.loc[tag_features].reset_index()
    tag_single_importance = tag_single_importance.sort_values('Importance', ascending=False)
    print("\nFine-grained importance of tag sub-features:")
    print(tag_single_importance.head(10))  # Only show top 10 sub-tags

    return overall_importance


# Test set evaluation
def evaluate_on_test(model, X_test, y_test):
    print("\nTest set evaluation...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"Test set: AUC={test_auc:.4f}, Recall={test_recall:.4f}, Precision={test_precision:.4f}, F1={test_f1:.4f}")
    return {
        'auc': test_auc, 'recall': test_recall,
        'precision': test_precision, 'f1': test_f1
    }


# Main function
def main():
    file_path = ""
    df = load_data(file_path)
    if df is None:
        return

    temp_train, _ = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Default_flag_6_months']
    )
    tag_importance = calculate_tag_importance(temp_train)
    print(f"Identified {len(tag_importance)} high-risk tags (based on chi-square test)")

    processed_df = preprocess_data(df, tag_importance)

    # Feature engineering (distinguish between base features and tag features)
    X, y, base_features, tag_features = feature_engineering(processed_df)

    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = split_data(X, y)

    # Calculate class weights (to handle imbalance)
    pos_count = sum(y_train == 1)
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / pos_count
    print(f"Class weights: Positive sample weight={pos_weight:.4f}")

    xgb_model = build_xgb_model(pos_weight)
    cv_results = cross_validate_xgb(xgb_model, X, y)
    val_results = train_and_evaluate_xgb(xgb_model, X_train, X_val, y_train, y_val)

    analyze_feature_importance(xgb_model, X.columns, base_features, tag_features)

    test_results = evaluate_on_test(xgb_model, X_test, y_test)
    # Save results
    result_df = pd.DataFrame({
        'Model': ['XGBoost'],
        'Training Set AUC': [val_results['train_auc']],
        'Validation Set AUC': [val_results['val_auc']],
        'Test Set AUC': [test_results['auc']],
        'Cross-validation AUC Mean': [cv_results['mean_auc']]
    })
    result_df.to_excel(os.path.join(TABLES_DIR, ''), index=False)
    print("\nResults saved to:", os.path.join(TABLES_DIR, ''))


if __name__ == "__main__":
    main()