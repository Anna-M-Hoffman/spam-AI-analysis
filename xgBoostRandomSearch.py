import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- 1. Load and Prepare Data -----------------
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]  # keep only label and text

# Encode labels: ham -> 0, spam -> 1
le = LabelEncoder()
y = le.fit_transform(df['v1'])

# Vectorize email text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['v2'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Label distribution in training set:", pd.Series(y_train).value_counts())

# ----------------- 2. XGBoost Setup -----------------
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
)

# RandomizedSearchCV parameter grid
param_dist = {
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ----------------- 3. Randomized Search -----------------
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Fit model
random_search.fit(X_train, y_train)

# ----------------- 4. Evaluation -----------------
print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

y_pred = random_search.predict(X_test)
y_probs = random_search.predict_proba(X_test)[:, 1]  # probability of spam

print(classification_report(y_test, y_pred, target_names=le.classes_))

# ----------------- Confusion Matrix Summary -----------------
cm = confusion_matrix(y_test, y_pred)

cm_summary = pd.DataFrame({
    'Type': ['True Ham', 'False Ham', 'False Spam', 'True Spam'],
    'Count': [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})

colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']  # one color per bar

plt.figure(figsize=(6,4))
bars = plt.bar(cm_summary['Type'], cm_summary['Count'], color=colors)
plt.title('Confusion Matrix Summary')
plt.ylabel('Count')
plt.show()


# ----------------- Top 20 Important Words -----------------
importances = random_search.best_estimator_.feature_importances_
feature_names = vectorizer.get_feature_names_out()
feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
top20 = feat_imp_df.sort_values(by='importance', ascending=False).head(20)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=top20, hue='feature', dodge=False, palette='viridis')
plt.title('Top 20 Important Words for Spam Detection')
plt.xlabel('Importance')
plt.ylabel('Word')
plt.legend([],[], frameon=False)  # remove legend since each bar is unique
plt.tight_layout()
plt.show()


# ----------------- Precision-Recall Curve -----------------
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
avg_precision = average_precision_score(y_test, y_probs)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='b', label=f'AP = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
