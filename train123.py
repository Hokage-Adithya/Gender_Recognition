import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
# New imports for metrics
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utils import load_data, split_data, create_model

# load the dataset
X, y = load_data()
# split the data into training, validation and testing sets
data = split_data(X, y, test_size=0.1, valid_size=0.1)
# construct the model
model = create_model()

# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir="logs")
# define early stopping to stop training after 5 epochs of not improving
# Note: It's better to monitor validation loss or accuracy
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

batch_size = 64
epochs = 100

# train the model using the training set and validating using validation set
model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping])

# save the model to a file
model.save("results/model.h5")


# =============================================================================
#           NEW AND EXPANDED EVALUATION SECTION
# =============================================================================
print(f"\nEvaluating the model using {len(data['X_test'])} samples...")

# 1. Get the true labels
y_true = data["y_test"]

# 2. Get the model's predictions as probabilities
y_pred_probs = model.predict(data["X_test"])

# 3. Convert probabilities to binary predictions (0 or 1) using a 0.5 threshold
y_pred = (y_pred_probs > 0.5).astype(int)

# 4. Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred) # recall is the same as sensitivity
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# 5. Print the results
print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity (Recall): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\n--- Confusion Matrix ---")
print("        Predicted 0   Predicted 1")
print(f"Actual 0:   {conf_matrix[0][0]:<10}  {conf_matrix[0][1]:<10}")
print(f"Actual 1:   {conf_matrix[1][0]:<10}  {conf_matrix[1][1]:<10}")
print("(0: Female, 1: Male)")
print("--------------------------")