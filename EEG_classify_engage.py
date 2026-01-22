import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from PLOT_Function import plot_confusion_matrix_custom
from models import EEG_Transformer, EEG_1D_CNN

save_dir = 'saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

file_path = 'D:/Research/EEGdata/train/train_set_drop_new.csv' 
raw = pd.read_csv(file_path)
raw_data = raw.values
X = raw_data[:, :24] 
y = raw_data[:, 24]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

def train_sklearn_model(name, model_instance, X_train, y_train, X_test, y_test):
    print(f"\n{'='*20} Processing {name} {'='*20}")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('minmax', MinMaxScaler()),
        ('classifier', model_instance)
    ])
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_scaled = sc.transform(X_train)
    X_test_scaled = sc.transform(X_test)
    min_max_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    X_train_scaled = min_max_scaler.fit_transform(X_train_scaled)
    X_test_scaled = min_max_scaler.fit_transform(X_test_scaled)
    scoring = {'acc': 'accuracy'}
    cv_results = cross_validate(model_instance, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)
    mean_cv_acc = np.mean(cv_results['test_acc'])
    std_cv_acc = np.std(cv_results['test_acc'])
    print(f"[{name}] 5-Fold CV Accuracy: {mean_cv_acc:.4f} (+/- {std_cv_acc:.4f})")
    model_instance.fit(X_train_scaled,y_train.astype('int'))
    joblib.dump(model_instance, os.path.join(save_dir, f'{name}_model.pkl'))
    print(f"[{name}] Model saved.")
    y_pred = model_instance.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"[{name}] Test Set Accuracy:  {acc:.4f}")
    print(f"[{name}] Test Set Precision: {prec:.4f}")
    print(f"[{name}] Test Set Recall:    {rec:.4f}")
    print(f"[{name}] Test Set F1-Score:  {f1_weighted:.4f}")
    y_test_binary = np.where(y_test > 0, 1, 0)
    y_pred_binary = np.where(y_pred > 0, 1, 0)
    acc_bin = accuracy_score(y_test_binary, y_pred_binary)
    prec_bin = precision_score(y_test_binary, y_pred_binary)
    rec_bin = recall_score(y_test_binary, y_pred_binary)
    f1_bin = f1_score(y_test_binary, y_pred_binary)
    print("=== 二分类 (HRI实际应用) 指标 ===")
    print(f"Accuracy: {acc_bin:.4f}")
    print(f"Precision: {prec_bin:.4f}")
    print(f"Recall: {rec_bin:.4f}")
    print(f"F1-Score: {f1_bin:.4f}")
    emotion_labels = ['not_engaged', "normal_engaged", "engaged"]
    plot_confusion_matrix_custom(y_test, y_pred, classes=emotion_labels, normalize=True,
                                 title=f'Normalized Confusion Matrix of {name}')
    plt.savefig(f'figs/{name}_CM.png', dpi=300, bbox_inches='tight')
    plt.show()
    return acc

def run_pytorch_epoch(model, loader, criterion, optimizer, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(is_train):
        for data, target in loader:
            if is_train:
                optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if is_train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item() 
    return running_loss / total, correct / total

def train_pytorch_pipeline(name, model_class, X_train, y_train, X_test, y_test):
    print(f"\n{'='*20} Processing {name} {'='*20}")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    print("Running 5-Fold Cross Validation...")
    fold = 1
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_tr_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        sc_internal = StandardScaler()
        mm_internal = MinMaxScaler()
        X_tr_fold = mm_internal.fit_transform(sc_internal.fit_transform(X_tr_fold))
        X_val_fold = mm_internal.transform(sc_internal.transform(X_val_fold))
        X_tr_t = torch.FloatTensor(X_tr_fold).unsqueeze(1).to(device)
        y_tr_t = torch.LongTensor(y_tr_fold.astype(int)).to(device)
        X_val_t = torch.FloatTensor(X_val_fold).unsqueeze(1).to(device)
        y_val_t = torch.LongTensor(y_val_fold.astype(int)).to(device)
        loader_tr = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
        loader_val = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)
        model_cv = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_cv.parameters(), lr=0.001)
        for _ in range(50):
            run_pytorch_epoch(model_cv, loader_tr, criterion, optimizer, is_train=True)
        _, val_acc = run_pytorch_epoch(model_cv, loader_val, criterion, optimizer, is_train=False)
        cv_accuracies.append(val_acc)
        fold += 1
        print(f"[{name}] 5-Fold CV Accuracy: {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies):.4f})")
    final_sc = StandardScaler()
    final_mm = MinMaxScaler()
    X_train_scaled = final_mm.fit_transform(final_sc.fit_transform(X_train))
    X_test_scaled = final_mm.transform(final_sc.transform(X_test))
    joblib.dump(final_sc, os.path.join(save_dir, 'dl_std_scaler.pkl'))
    joblib.dump(final_mm, os.path.join(save_dir, 'dl_minmax_scaler.pkl'))
    X_train_t = torch.FloatTensor(X_train_scaled).unsqueeze(1).to(device)
    y_train_t = torch.LongTensor(y_train.astype(int)).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).unsqueeze(1).to(device)
    y_test_t = torch.LongTensor(y_test.astype(int)).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    final_model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_history = []
    best_loss = float('inf')
    for epoch in range(100):
        train_loss, _ = run_pytorch_epoch(final_model, train_loader, criterion, optimizer, is_train=True)
        loss_history.append(train_loss)
    torch.save(final_model.state_dict(), os.path.join(save_dir, f'{name}_model.pth'))
    print(f"[{name}] Model saved.")
    plt.figure(figsize=(6, 2))
    plt.plot(loss_history)
    plt.title(f'{name} Training Loss')
    plt.show()
    final_model.eval()
    with torch.no_grad():
        output = final_model(X_test_t)
        _, y_pred_t = torch.max(output, 1)
        y_pred = y_pred_t.cpu().numpy()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"[{name}] Test Set Accuracy:  {acc:.4f}")
    print(f"[{name}] Test Set Precision: {prec:.4f}")
    print(f"[{name}] Test Set Recall:    {rec:.4f}")
    print(f"[{name}] Test Set F1-Score:  {f1_weighted:.4f}")
    y_test_binary = np.where(y_test > 0, 1, 0)
    y_pred_binary = np.where(y_pred > 0, 1, 0)
    acc_bin = accuracy_score(y_test_binary, y_pred_binary)
    prec_bin = precision_score(y_test_binary, y_pred_binary) 
    rec_bin = recall_score(y_test_binary, y_pred_binary)
    f1_bin = f1_score(y_test_binary, y_pred_binary)

    print("=== 二分类 (HRI实际应用) 指标 ===")
    print(f"Accuracy: {acc_bin:.4f}")
    print(f"Precision: {prec_bin:.4f}")
    print(f"Recall: {rec_bin:.4f}")
    print(f"F1-Score: {f1_bin:.4f}")
    emotion_labels = ['not_engaged', "normal_engaged", "engaged"]
    plot_confusion_matrix_custom(y_test, y_pred, classes=emotion_labels, normalize=True,
                                 title=f'Normalized Confusion Matrix of {name}')
    plt.savefig(f'figs/{name}_CM.png', dpi=300, bbox_inches='tight')
    plt.show()
    return acc


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    models = {
        "SVM": SVC(C=1, kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression" : LogisticRegression(),
        "DecisionTree" : DecisionTreeRegressor()
    }
    sklearn_results = {}
    for name, model in models.items():
        acc = train_sklearn_model(name, model, X_train, y_train, X_test, y_test)
        sklearn_results[name] = acc

    train_pytorch_pipeline("1D-CNN", EEG_1D_CNN, X_train, y_train, X_test, y_test)

    train_pytorch_pipeline("Transformer", EEG_Transformer, X_train, y_train, X_test, y_test)