# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D, Flatten, BatchNormalization
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping
import seaborn as sns

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def define_model(
        timesteps, 
        num_features, 
        num_classes, 
        conv_filters, 
        kernel_size, 
        lstm_units, 
        dropout_conv, 
        dropout_lstm, 
        learning_rate, 
        kernel_regularizer_l1,
        kernel_regularizer_l2
    ):

    model = Sequential()

    # Conv1D layer
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', 
                     input_shape=(timesteps, num_features), kernel_regularizer=l1(kernel_regularizer_l1)))
    model.add(BatchNormalization())  # Add BatchNormalization
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_conv))

    # LSTM layer
    model.add(LSTM(
        lstm_units, 
        activation='tanh', 
        recurrent_activation='sigmoid', 
        recurrent_dropout=0, 
        dropout=0, 
        unroll=False, 
        use_bias=True, 
        kernel_regularizer=l2(kernel_regularizer_l2)
    ))

    model.add(BatchNormalization())  # Add BatchNormalization
    model.add(Dropout(dropout_lstm))

    # Flatten layer
    model.add(Flatten())

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        
    return model

def balance_sequences_per_seed(dt, n, list_seed_name, label):
    if len(list_seed_name)==0:
        return dt
    for seed in list_seed_name:
        sequeces = dt[dt['label'] == label]
        sequence_numbers_query = sequeces.query(f'seed_name == {seed}')['sample_id'].unique()
        random_indices = np.random.choice(sequence_numbers_query, n, replace=False)
        sequences_to_remove = np.setdiff1d(sequence_numbers_query, random_indices)
        dt = dt[~dt['sample_id'].isin(sequences_to_remove)]
    return dt  

def training_process(timesteps, num_classes, num_features, X_train, X_val, Y_train, Y_val, model_parameters, batch_size):
    # Create a study and optimize
    
    model = define_model(
        timesteps, 
        num_features, 
        num_classes, 
        model_parameters['conv_filters'], 
        model_parameters['kernel_size'], 
        model_parameters['lstm_units'], 
        model_parameters['dropout_conv'], 
        model_parameters['dropout_lstm'], 
        model_parameters['learning_rate'], 
        model_parameters['kernel_regularizer_l1'],
        model_parameters['kernel_regularizer_l2']
    )
    patience = 10

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Convert Y_train and Y_val to one-hot encoding
    Y_train = to_categorical(Y_train, num_classes)
    Y_val = to_categorical(Y_val, num_classes)

    # Train the model with early stopping
    history = model.fit(
        X_train, Y_train, 
        epochs=100, 
        batch_size=batch_size, 
        validation_data=(X_val, Y_val),  
        callbacks=[early_stopping]
    )

    return history, model

def checking_data(label_mapping, timesteps, n_classes, n_features, X_balanced_normalized, Y_balanced, X_train, X_val, X_test, y_train, y_val, y_test):
    print("===== Data Information =====")
    print(f"Timesteps: {timesteps}")
    print(f"Number of classes: {n_classes}")
    print(f"Number of features: {n_features}")
    
    print("\n===== Label Mapping =====")
    for label, numerical_value in label_mapping.items():
        print(f"'{label}' -> {numerical_value}")
    
    print("\n===== Balanced and Normalized Data =====")
    print(f"Shape of X_balanced_normalized: {X_balanced_normalized.shape}")
    print(f"Shape of Y_balanced: {Y_balanced.shape}")
    
    print("\n===== Training Set =====")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    print("\n===== Validation Set =====")
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of y_val: {y_val.shape}")
    
    print("\n===== Test Set =====")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    
    print("\n===== Training Set Statistics =====")
    train_counts = np.bincount(y_train.flatten())
    for numerical_value, count in enumerate(train_counts):
        label = list(label_mapping.keys())[list(label_mapping.values()).index(numerical_value)]
        print(f"Class '{label}' (Value {numerical_value}): {count} samples")

    print("\n===== Validation Set Statistics =====")
    train_counts = np.bincount(y_val.flatten())
    for numerical_value, count in enumerate(train_counts):
        label = list(label_mapping.keys())[list(label_mapping.values()).index(numerical_value)]
        print(f"Class '{label}' (Value {numerical_value}): {count} samples")

    print("\n===== Test Set Statistics =====")
    test_counts = np.bincount(y_test.flatten())
    for numerical_value, count in enumerate(test_counts):
        label = list(label_mapping.keys())[list(label_mapping.values()).index(numerical_value)]
        print(f"Class '{label}' (Value {numerical_value}): {count} samples")

def show_data_comparison(label_mapping, X_before_balancing, Y_before_balancing, X_balanced_normalized, Y_balanced):
    print("===== Label Mapping =====")
    for label, numerical_value in label_mapping.items():
        print(f"'{label}' -> {numerical_value}")

    print("\n===== Data Before Balancing =====")
    print(f"Shape of X_before_balancing: {X_before_balancing.shape}")
    print(f"Shape of Y_before_balancing: {Y_before_balancing.shape}")
    print("Sample count per class (before balancing):")
    for numerical_value, count in enumerate(np.bincount(Y_before_balancing)):
        label = list(label_mapping.keys())[list(label_mapping.values()).index(numerical_value)]
        print(f"Class '{label}' (Value {numerical_value}): {count} samples")

    print("\n===== Data After Balancing =====")
    print(f"Shape of X_balanced_normalized: {X_balanced_normalized.shape}")
    print(f"Shape of Y_balanced: {Y_balanced.shape}")
    print("Sample count per class (after balancing):")
    for numerical_value, count in enumerate(np.bincount(Y_balanced)):
        label = list(label_mapping.keys())[list(label_mapping.values()).index(numerical_value)]
        print(f"Class '{label}' (Value {numerical_value}): {count} samples")

def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    
    # Convert y_train and y_test to one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=2)
    y_test_one_hot = to_categorical(y_test, num_classes=2)
    
    # Get predicted probabilities
    y_train_pred_proba = model.predict(X_train)
    y_test_pred_proba = model.predict(X_test)
    
    # Get predicted classes (class with highest probability)
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Print metrics
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    
    # Plot a beautiful confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Happy (0)', 'Predicted Neutral (1)'], yticklabels=['True Happy (0)', 'True Neutral (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # ROC AUC Score for binary classification (using class 1 probability)
    fpr_train, tpr_train, _ = roc_curve(y_train_one_hot[:, 1], y_train_pred_proba[:, 1])
    roc_auc_train = roc_auc_score(y_train_one_hot[:, 1], y_train_pred_proba[:, 1])
    
    fpr_test, tpr_test, _ = roc_curve(y_test_one_hot[:, 1], y_test_pred_proba[:, 1])
    roc_auc_test = roc_auc_score(y_test_one_hot[:, 1], y_test_pred_proba[:, 1])
    
    # Plot ROC curves
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (AUC = %0.2f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='Test ROC curve (AUC = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
    return accuracy, precision, recall, f1, conf_matrix

def plot_learning_curves(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'][:60])
    plt.plot(history.history['val_loss'][:60])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    
    plt.show()

## Pre process data =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Function to load and clean the dataset
def load_and_clean_dataset(dataset_path):
    SEQUENCES = pd.read_csv(dataset_path)
    if 'Unnamed: 0' in SEQUENCES.columns:
        SEQUENCES.drop(columns=['Unnamed: 0'], inplace=True)
    return SEQUENCES

# Function to filter the desired emotions
def filter_emotions(SEQUENCES, emotions_query):
    return SEQUENCES.query(emotions_query)

# Function to encode labels into numerical values
def encode_labels(SEQUENCES_ENCODED):
    encoder = LabelEncoder()
    SEQUENCES_ENCODED['label_numerical'] = encoder.fit_transform(SEQUENCES_ENCODED['label'])
    label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print("Label Mapping:", label_mapping)
    return SEQUENCES_ENCODED, label_mapping

def select_features(SEQUENCES_ENCODED):
    # Selects the features to be used
    features = SEQUENCES_ENCODED.columns.to_list()[4:26]
    print(f"Features being used: {features}")
    
    # Counts the number of samples per 'sample_id'
    sample_counts = SEQUENCES_ENCODED.sample_id.value_counts()
    len_sample_max = sample_counts.max()  # Gets the maximum value
    
    # Finds the 'sample_id' with the largest number of occurrences
    max_sample_id = sample_counts.idxmax()
    
    # Filters the rows corresponding to the 'sample_id' with the maximum value
    max_sample_items = SEQUENCES_ENCODED[SEQUENCES_ENCODED.sample_id == max_sample_id]
    
    # Returns the features, number of features, the maximum value, and the corresponding rows
    return features, len(features), len_sample_max, max_sample_items

# Function to pad sequences
def pad_sequences(SEQUENCES_ENCODED, features, len_sample_max):
    grouped_data = []
    for sample_id, group in SEQUENCES_ENCODED.groupby('sample_id'):
        sequence_features = group[features]
        if len(sequence_features) < len_sample_max:
            N_rows = len_sample_max - len(sequence_features)
            pad = pd.DataFrame(np.zeros((N_rows, len(features))), columns=sequence_features.columns)
            sequence_features_pad = pd.concat([pad, sequence_features], ignore_index=True)
        else:
            sequence_features_pad = sequence_features
        label = SEQUENCES_ENCODED[SEQUENCES_ENCODED.sample_id == sample_id].iloc[0].label_numerical
        grouped_data.append((sequence_features_pad, label))
    
    X = np.array([item[0] for item in grouped_data])
    Y = np.array([item[1] for item in grouped_data])
    return X, Y

# Function to balance the data
def balance_classes(X, Y):
    label_counts = np.bincount(Y)
    min_count = np.min(label_counts)

    X_balanced = []
    Y_complete = []
    for label in np.unique(Y):
        idx = np.where(Y == label)[0]
        selected_idx = np.random.choice(idx, min_count, replace=False)
        X_balanced.append(X[selected_idx])
        Y_complete.append(Y[selected_idx])
    
    X_balanced = np.vstack(X_balanced)
    Y_complete = np.hstack(Y_complete)
    return X_balanced, Y_complete

# Function to normalize the data (using external function)
def normalize_data(X_balanced):
    # Create a new list to store the normalized matrices
    X_normalized = []
    
    for matrix in X_balanced:
        # Convert the matrix to a numpy array, if it is not already
        matrix = np.array(matrix)
        
        # Normalize the matrix between 0 and 1
        matrix_min = matrix.min()
        matrix_max = matrix.max()
        matrix_normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        
        # Add the normalized matrix to the list
        X_normalized.append(matrix_normalized)
    
    return np.array(X_normalized)

# Function to split the data into train, validation, and test sets
def split_data(X, Y, train_size=0.7, val_size=0.15, test_size=0.15):
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=42, shuffle=True)
    val_relative_size = val_size / (train_size + val_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_relative_size, stratify=Y_train_val, random_state=42, shuffle=True)

    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Main function for preprocessing that uses the smaller functions
def preprocess_data_current_dataset(dataset_path, emotions, train_size=0.7, val_size=0.15, test_size=0.15):
    SEQUENCES = load_and_clean_dataset(dataset_path)
    SEQUENCES = filter_emotions(SEQUENCES, emotions)
    
    SEQUENCES_ENCODED, label_mapping = encode_labels(SEQUENCES)
    
    features, n_features, len_sample_max, max_sample_items = select_features(SEQUENCES_ENCODED)
    timesteps = len_sample_max
    n_classes = len(label_mapping)
        
    X, Y = pad_sequences(SEQUENCES_ENCODED, features, len_sample_max)
    X_before_balancing, Y_before_balancing = X.copy(), Y.copy()

    X_balanced, Y_complete = balance_classes(X, Y)
    X_complete = normalize_data(X_balanced)
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X_complete, Y_complete, train_size, val_size, test_size)

    return {
        'label_mapping': label_mapping,
        'timesteps': timesteps,
        'n_classes': n_classes,
        'n_features': n_features,
        'X_before_balancing': X_before_balancing,
        'Y_before_balancing': Y_before_balancing,
        'X_complete': X_complete,
        'Y_complete': Y_complete,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_val': Y_val,
        'Y_test': Y_test
    }
