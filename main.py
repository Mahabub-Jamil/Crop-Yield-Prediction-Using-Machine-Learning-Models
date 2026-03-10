import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('D:/AI Project/Capstone Project/Final/dataset/Augmented_Crop_Yield_Data.csv')

# Basic exploration
print(df.head())
print(df.tail())
print("Shape:", df.shape)
print("Duplicated:", df.duplicated().sum())
print("Null values:\n", df.isnull().sum())
df.info()
print(df.describe())

# Unique crops
print("Unique Crops:", df['Crop'].unique())
print("Crop Counts:\n", df['Crop'].value_counts())

# Visualize Soil Type distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='Soil Type', data=df, palette='Set2')
plt.title("Distribution of Soil Types")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crop summary using only numeric columns
numeric_cols = df.select_dtypes(include='number').columns.tolist()
crop_summary = pd.pivot_table(df[numeric_cols + ['Crop']], index='Crop', aggfunc='mean').reset_index()

# Boxplots for each numeric feature
df1 = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Moisture']]
for col in df1.columns:
    plt.figure(figsize=(12, 5))
    sns.boxplot(x=df1[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# Barplot of each feature per crop
for col in df1.columns:
    plt.figure(figsize=(14, 6))
    sns.barplot(x='Crop', y=col, data=crop_summary, palette='hls')
    plt.xticks(rotation=90)
    plt.title(f'{col} levels across crops')
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df1.corr(), annot=True, cmap='Wistia')
plt.title('Correlation between Features', fontsize=15)
plt.tight_layout()
plt.show()

# Encode categorical labels
from sklearn.preprocessing import LabelEncoder
le_crop = LabelEncoder()
le_soil = LabelEncoder()
df['Crop'] = le_crop.fit_transform(df['Crop'])
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

# Drop Yield if present
if 'Yield' in df.columns:
    df = df.drop('Yield', axis=1)

# Features and label
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Moisture', 'Soil Type']
X = df[features]
y = df['Crop']

# Train-test split
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Print train and test sample counts
print(f"Training dataset samples: {X_train.shape[0]}")
print(f"Test dataset samples: {X_test.shape[0]}")

# Pie chart for train-test split
sizes = [X_train.shape[0], X_test.shape[0]]
labels = ['Training Set', 'Test Set']
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=(0.05, 0.05))
plt.title('Train-Test Dataset Split')
plt.show()

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                              confusion_matrix, classification_report, roc_curve, auc, 
                              roc_auc_score)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

models = {
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=10, criterion='entropy'),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=0),
    "SVM": SVC(kernel='linear', random_state=0, probability=True),  # Added probability=True for ROC
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0),
    "PAC": PassiveAggressiveClassifier(max_iter=1000, random_state=0),
    "Ridge": RidgeClassifier(),
    "SGD": SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=0),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=0),
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=0)
}

crop_names = le_crop.classes_
n_classes = len(crop_names)

# Binarize the output for ROC curve (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
algorithm_labels = []
best_model_name = None
best_model_score = 0.0
best_model = None

print("\n" + "="*80)
print("MODEL PERFORMANCE EVALUATION")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"MODEL: {name}")
    print('='*80)
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    mean_cv_score = scores.mean()
    std_dev = scores.std()
    
    print(f"\nCross-Validation Results:")
    print(f"  Fold Accuracies: {scores}")
    print(f"  Mean Accuracy:   {mean_cv_score:.4f}")
    print(f"  Std Deviation:   {std_dev:.4f}")
    
    # Train on full training set and predict on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    accuracy_scores.append(mean_cv_score)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)
    algorithm_labels.append(name)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=crop_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=crop_names, yticklabels=crop_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # ROC Curve (only for models with predict_proba)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 10))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 
                       'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'navy',
                       'teal', 'maroon', 'lime', 'aqua', 'fuchsia', 'silver'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{crop_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {name}')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        
        print(f"\nROC AUC Score (Macro-Average): {roc_auc_macro:.4f}")
    
    # Track best model
    if mean_cv_score > best_model_score:
        best_model_score = mean_cv_score
        best_model_name = name
        best_model = model

# Comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
sns.barplot(x=algorithm_labels, y=accuracy_scores, palette='Set3', ax=axes[0, 0])
axes[0, 0].set_xlabel('Algorithm', fontsize=12)
axes[0, 0].set_ylabel('Mean CV Accuracy', fontsize=12)
axes[0, 0].set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

# Precision
sns.barplot(x=algorithm_labels, y=precision_scores, palette='Set2', ax=axes[0, 1])
axes[0, 1].set_xlabel('Algorithm', fontsize=12)
axes[0, 1].set_ylabel('Precision (Weighted)', fontsize=12)
axes[0, 1].set_title('Model Comparison - Precision', fontsize=14, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

# Recall
sns.barplot(x=algorithm_labels, y=recall_scores, palette='Set1', ax=axes[1, 0])
axes[1, 0].set_xlabel('Algorithm', fontsize=12)
axes[1, 0].set_ylabel('Recall (Weighted)', fontsize=12)
axes[1, 0].set_title('Model Comparison - Recall', fontsize=14, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

# F1-Score
sns.barplot(x=algorithm_labels, y=f1_scores, palette='viridis', ax=axes[1, 1])
axes[1, 1].set_xlabel('Algorithm', fontsize=12)
axes[1, 1].set_ylabel('F1-Score (Weighted)', fontsize=12)
axes[1, 1].set_title('Model Comparison - F1-Score', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Overall metrics comparison in one plot
metrics_df = pd.DataFrame({
    'Algorithm': algorithm_labels,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores
})

plt.figure(figsize=(16, 8))
x = np.arange(len(algorithm_labels))
width = 0.2

plt.bar(x - 1.5*width, accuracy_scores, width, label='Accuracy', color='skyblue')
plt.bar(x - 0.5*width, precision_scores, width, label='Precision', color='lightgreen')
plt.bar(x + 0.5*width, recall_scores, width, label='Recall', color='salmon')
plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='gold')

plt.xlabel('Algorithm', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Comprehensive Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, algorithm_labels, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Save the best model and label encoders
print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"Cross-Validation Accuracy: {best_model_score:.4f}")
print('='*80)

joblib.dump(best_model, 'best_crop_model.pkl')
joblib.dump(le_crop, 'label_encoder_crop.pkl')
joblib.dump(le_soil, 'label_encoder_soil.pkl')

print("\nModels saved successfully!")
print("  - best_crop_model.pkl")
print("  - label_encoder_crop.pkl")
print("  - label_encoder_soil.pkl")