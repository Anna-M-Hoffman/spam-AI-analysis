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
# v1 column is ham/spam and v2 column is text slice
df = df[['v1', 'v2']]  # keep only ham/spam and text

# Encode labels: ham -> 0, spam -> 1
le = LabelEncoder()
y = le.fit_transform(df['v1'])  # LabelEncoder transforms categorical labels to numerical form

# Vectorize email text using TF-IDF
# TF or Term Frequency measures how often a word appears in a document.
# IDF or Inverse Document Frequency reduces common words weight across data while increasing the weight of rare words.
vectorizer = TfidfVectorizer(max_features=5000)  # max_features limits vocabulary to 5000 most important words
X = vectorizer.fit_transform(df['v2'])  # convert text into numeric features

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,    # 20% for testing, 80% for training
    stratify=y,             # stratify keeps balanced percentages of ham and spam
    random_state=42         # consistent seed
)

print("Training samples:", X_train.shape[0])    # Prints number of training samples
print("Test samples:", X_test.shape[0])         # Prints number of testing samples


# ----------------- 2. XGBoost Setup -----------------
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # binary classification
    eval_metric='logloss',        # evaluation metric for optimization
    random_state=42               # for reproducibility
)

# RandomizedSearchCV parameter grid
param_dist = {
    'max_depth': [3, 4, 5],                 # tree depth
    'n_estimators': [100, 200, 300],        # number of trees
    'learning_rate': [0.01, 0.05, 0.1],     # step size shrinkage of each tree contribution for a more accurate model
    'subsample': [0.8, 1.0],                # fraction of samples per tree
    'colsample_bytree': [0.8, 1.0]          # fraction of features (vectored words) per tree
}

# Stratified K-Fold CV (cross-validation) - how well unseen data is predicted
cv = StratifiedKFold(n_splits=5,       # Data divided into three folds, testing multiple splits
                     shuffle=True,     # Each fold has a good mix of classes (ham and spam)
                     random_state=42)  # reproducibility

# ----------------- 3. Randomized Search (for hyperparameter tuning) -----------------
random_search = RandomizedSearchCV(
    estimator=xgb_model,             # model to tune
    param_distributions=param_dist,  # hyperparameter grid
    n_iter=20,                       # number of random combinations to try
    scoring='accuracy',              # evaluation metric - the best hyperparameters are the most accurate
    cv=cv,                           # cross-validation strategy - k fold subsets for prediction accuracy of unseen data
    verbose=1,                       # does not print progress (2 needed to print progress)
    n_jobs=-1,                       # use all CPU cores
    random_state=42                  # reproducibility
)

# Fit model
random_search.fit(X_train, y_train)

# ----------------- 4. Evaluation -----------------
print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

y_pred = random_search.predict(X_test)               # outputs the predicted class labels (0, 1) for ham/spam
y_probs = random_search.predict_proba(X_test)[:, 1]  # takes the probability of spam for each email

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

# Print only precision, recall, f1-score, support per class
for label in le.classes_:
    metrics = report[label]
    print(f"{label}:")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall:    {metrics['recall']:.2f}")
    print(f"  F1-score:  {metrics['f1-score']:.2f}")
    print(f"  Support:   {int(metrics['support'])}")
    print()

# ----------------- Confusion Matrix Summary -----------------
cm = confusion_matrix(y_test, y_pred)  # [[TP, FN], [FP, TN]]

cm_summary = pd.DataFrame({
    'Type': ['True Ham', 'False Ham', 'False Spam', 'True Spam'],  # descriptive labels
    'Count': [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})

# Different color for each bar
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']  # blue, orange, red, green

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
sns.barplot(
    x='importance', y='feature',
    data=top20,
    hue='feature',   # ensures unique color per bar
    dodge=False,
    palette='viridis'
)
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

