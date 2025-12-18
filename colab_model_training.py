# 🚀 Smart Irrigation System - Multi-Model Comparison (Google Colab)
# Run this in Google Colab

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, StandardScaler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1️⃣ Upload dataset file
print("📁 Please upload your data_core.csv file:")
from google.colab import files
uploaded = files.upload()

# 2️⃣ Start Spark session
spark = SparkSession.builder.appName("Smart_Irrigation_System").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 3️⃣ Load dataset
data = spark.read.csv("data_core.csv", header=True, inferSchema=True)
print("✅ Dataset Loaded")
data.show(5)
print(f"Total Records: {data.count()}")

# 3️⃣ Handle missing values using Imputer
numeric_cols = ["Temparature", "Humidity", "Nitrogen", "Potassium", "Phosphorous"]
imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols).setStrategy("median")
data = imputer.fit(data).transform(data)

# 4️⃣ Encode categorical columns
categorical_cols = ["Soil Type", "Crop Type", "Fertilizer Name"]
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name.replace(" ","_") + "_Indexed", handleInvalid="keep")
    data = indexer.fit(data).transform(data)

# 5️⃣ Feature engineering (Moisture is target column)
feature_cols = ["Temparature", "Humidity", "Nitrogen", "Potassium", "Phosphorous",
                "Soil_Type_Indexed", "Crop_Type_Indexed", "Fertilizer_Name_Indexed"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# ✅ Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
data = scaler.fit(data).transform(data)

# 6️⃣ Create classification labels for Logistic Regression & SVM
# Convert continuous moisture values to categories: Low, Medium, High
data = data.withColumn("Moisture_Category", 
                       when(col("Moisture") < 30, 0.0)
                       .when((col("Moisture") >= 30) & (col("Moisture") < 60), 1.0)
                       .otherwise(2.0))

# 7️⃣ Split data
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

print(f"\n📊 Training Set: {train_data.count()} records")
print(f"📊 Test Set: {test_data.count()} records")

# ========================================
# 🔴 MODEL 1: Linear Regression
# ========================================
print("\n" + "="*60)
print("🔴 Training Linear Regression Model...")
print("="*60)

lr = LinearRegression(featuresCol="scaledFeatures", labelCol="Moisture",
                      regParam=0.1, elasticNetParam=0.3, maxIter=100)
lr_model = lr.fit(train_data)

lr_predictions = lr_model.transform(test_data)

# Evaluation
evaluator_reg = RegressionEvaluator(labelCol="Moisture", predictionCol="prediction")
lr_rmse = evaluator_reg.evaluate(lr_predictions, {evaluator_reg.metricName: "rmse"})
lr_mae = evaluator_reg.evaluate(lr_predictions, {evaluator_reg.metricName: "mae"})
lr_r2 = evaluator_reg.evaluate(lr_predictions, {evaluator_reg.metricName: "r2"})

print(f"\n📊 Linear Regression Performance:")
print(f"   RMSE = {lr_rmse:.4f}")
print(f"   MAE = {lr_mae:.4f}")
print(f"   R² = {lr_r2:.4f}")
print(f"   Accuracy (R²) = {lr_r2 * 100:.2f}%")

# ========================================
# 🟢 MODEL 2: Logistic Regression (Classification)
# ========================================
print("\n" + "="*60)
print("🟢 Training Logistic Regression Model (Classification)...")
print("="*60)

logistic = LogisticRegression(featuresCol="scaledFeatures", labelCol="Moisture_Category",
                               maxIter=100, regParam=0.1, elasticNetParam=0.3)
logistic_model = logistic.fit(train_data)

logistic_predictions = logistic_model.transform(test_data)

# Evaluation
evaluator_class = MulticlassClassificationEvaluator(labelCol="Moisture_Category", 
                                                     predictionCol="prediction")
logistic_accuracy = evaluator_class.evaluate(logistic_predictions, {evaluator_class.metricName: "accuracy"})
logistic_f1 = evaluator_class.evaluate(logistic_predictions, {evaluator_class.metricName: "f1"})
logistic_precision = evaluator_class.evaluate(logistic_predictions, {evaluator_class.metricName: "weightedPrecision"})
logistic_recall = evaluator_class.evaluate(logistic_predictions, {evaluator_class.metricName: "weightedRecall"})

print(f"\n📊 Logistic Regression Performance:")
print(f"   Accuracy = {logistic_accuracy * 100:.2f}%")
print(f"   F1 Score = {logistic_f1:.4f}")
print(f"   Precision = {logistic_precision:.4f}")
print(f"   Recall = {logistic_recall:.4f}")

# ========================================
# 🔵 MODEL 3: Decision Tree Classifier (Replaces SVM for multi-class)
# ========================================
print("\n" + "="*60)
print("🔵 Training Decision Tree Classifier (Multi-class)...")
print("="*60)

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol="scaledFeatures", labelCol="Moisture_Category",
                            maxDepth=10, maxBins=32)
dt_model = dt.fit(train_data)

dt_predictions = dt_model.transform(test_data)

# Evaluation
dt_accuracy = evaluator_class.evaluate(dt_predictions, {evaluator_class.metricName: "accuracy"})
dt_f1 = evaluator_class.evaluate(dt_predictions, {evaluator_class.metricName: "f1"})
dt_precision = evaluator_class.evaluate(dt_predictions, {evaluator_class.metricName: "weightedPrecision"})
dt_recall = evaluator_class.evaluate(dt_predictions, {evaluator_class.metricName: "weightedRecall"})

print(f"\n📊 Decision Tree Performance:")
print(f"   Accuracy = {dt_accuracy * 100:.2f}%")
print(f"   F1 Score = {dt_f1:.4f}")
print(f"   Precision = {dt_precision:.4f}")
print(f"   Recall = {dt_recall:.4f}")

# ========================================
# 📊 MODEL COMPARISON
# ========================================
print("\n" + "="*60)
print("📊 MODEL COMPARISON SUMMARY")
print("="*60)

comparison_data = {
    'Model': ['Linear Regression', 'Logistic Regression', 'Decision Tree'],
    'Accuracy': [lr_r2 * 100, logistic_accuracy * 100, dt_accuracy * 100],
    'F1_Score': ['-', logistic_f1, dt_f1],
    'RMSE': [lr_rmse, '-', '-']
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Find best model
best_classification_model = 'Logistic Regression' if logistic_accuracy > dt_accuracy else 'Decision Tree'
print(f"\n🏆 Best Classification Model: {best_classification_model}")
print(f"🏆 Best Regression Model: Linear Regression (R² = {lr_r2 * 100:.2f}%)")

# ========================================
# 💾 SAVE PREDICTIONS
# ========================================
print("\n" + "="*60)
print("💾 Saving Predictions to CSV...")
print("="*60)

# Save Linear Regression predictions
lr_predictions_pd = lr_predictions.select(
    "Temparature", "Humidity", "Nitrogen", "Potassium", "Phosphorous",
    "Soil Type", "Crop Type", "Fertilizer Name", "Moisture", "prediction"
).toPandas()
lr_predictions_pd.to_csv("predictions.csv", index=False)
print("✅ predictions.csv saved (Linear Regression)")

# Save all model comparisons
all_predictions = lr_predictions.select(
    "Temparature", "Humidity", "Nitrogen", "Potassium", "Phosphorous",
    "Soil Type", "Crop Type", "Fertilizer Name", "Moisture"
).toPandas()

all_predictions['LR_Prediction'] = lr_predictions_pd['prediction']
all_predictions['Logistic_Category'] = logistic_predictions.select("prediction").toPandas()['prediction']
all_predictions['DecisionTree_Category'] = dt_predictions.select("prediction").toPandas()['prediction']

# Convert category predictions back to moisture ranges for visualization
all_predictions['Logistic_Prediction'] = all_predictions['Logistic_Category'].map({
    0.0: 'Low (<30)', 1.0: 'Medium (30-60)', 2.0: 'High (>60)'
})
all_predictions['DecisionTree_Prediction'] = all_predictions['DecisionTree_Category'].map({
    0.0: 'Low (<30)', 1.0: 'Medium (30-60)', 2.0: 'High (>60)'
})

all_predictions.to_csv("all_model_predictions.csv", index=False)
print("✅ all_model_predictions.csv saved (All Models)")

# Save model metrics
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Logistic Regression', 'Decision Tree'],
    'Type': ['Regression', 'Classification', 'Classification'],
    'Accuracy_R2': [lr_r2 * 100, logistic_accuracy * 100, dt_accuracy * 100],
    'RMSE': [lr_rmse, np.nan, np.nan],
    'MAE': [lr_mae, np.nan, np.nan],
    'F1_Score': [np.nan, logistic_f1, dt_f1],
    'Precision': [np.nan, logistic_precision, dt_precision],
    'Recall': [np.nan, logistic_recall, dt_recall]
})
metrics_df.to_csv("model_metrics.csv", index=False)
print("✅ model_metrics.csv saved")

# ========================================
# 📈 VISUALIZATIONS
# ========================================
print("\n" + "="*60)
print("📈 Creating Visualizations...")
print("="*60)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Model Accuracy Comparison
ax1 = axes[0, 0]
models = ['Linear Reg\n(R²)', 'Logistic Reg\n(Accuracy)', 'Decision Tree\n(Accuracy)']
accuracies = [lr_r2 * 100, logistic_accuracy * 100, dt_accuracy * 100]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

# 2. Linear Regression: Actual vs Predicted
ax2 = axes[0, 1]
sample_lr = lr_predictions_pd.sample(min(100, len(lr_predictions_pd)))
ax2.scatter(sample_lr['Moisture'], sample_lr['prediction'], alpha=0.6, color='#FF6B6B')
ax2.plot([sample_lr['Moisture'].min(), sample_lr['Moisture'].max()], 
         [sample_lr['Moisture'].min(), sample_lr['Moisture'].max()], 
         'k--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Moisture', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted Moisture', fontsize=12, fontweight='bold')
ax2.set_title('Linear Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
ax2.legend()

# 3. Classification Model F1 Scores
ax3 = axes[1, 0]
f1_models = ['Logistic Regression', 'Decision Tree']
f1_scores = [logistic_f1, dt_f1]
bars2 = ax3.barh(f1_models, f1_scores, color=['#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
ax3.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax3.set_title('Classification Models: F1 Score Comparison', fontsize=14, fontweight='bold')
ax3.set_xlim([0, 1])
for bar in bars2:
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
             f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=11)

# 4. Moisture Category Distribution
ax4 = axes[1, 1]
category_counts = all_predictions['Logistic_Category'].value_counts().sort_index()
# Create labels dynamically based on actual categories present
category_mapping = {0.0: 'Low (<30)', 1.0: 'Medium (30-60)', 2.0: 'High (>60)'}
category_labels = [category_mapping.get(cat, f'Category {int(cat)}') for cat in category_counts.index]
colors_pie = ['#FFB6B9', '#FEC8D8', '#FFDFD3'][:len(category_counts)]
ax4.pie(category_counts, labels=category_labels, autopct='%1.1f%%', 
        colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Moisture Category Distribution\n(Logistic Regression)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ model_comparison.png saved")
plt.show()

# Additional: Confusion Matrix for Logistic Regression
from sklearn.metrics import confusion_matrix
import itertools

logistic_pred_pd = logistic_predictions.select("Moisture_Category", "prediction").toPandas()
cm = confusion_matrix(logistic_pred_pd['Moisture_Category'], logistic_pred_pd['prediction'])

# Get actual unique labels for confusion matrix
unique_labels = sorted(logistic_pred_pd['Moisture_Category'].unique())
cm_labels = [category_mapping.get(float(label), f'Category {int(label)}') for label in unique_labels]

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Logistic Regression', fontsize=16, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, cm_labels, rotation=45)
plt.yticks(tick_marks, cm_labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black",
             fontweight='bold', fontsize=12)

plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ confusion_matrix.png saved")
plt.show()

print("\n" + "="*60)
print("✅ ALL DONE! Files saved:")
print("   - predictions.csv")
print("   - all_model_predictions.csv")
print("   - model_metrics.csv")
print("   - model_comparison.png")
print("   - confusion_matrix.png")
print("\n📊 Models Trained:")
print("   1. Linear Regression (Regression)")
print("   2. Logistic Regression (Classification)")
print("   3. Decision Tree (Classification)")
print("="*60)

spark.stop()
