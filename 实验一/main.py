import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data():
    data_dir = "processed_8_9"
    train_data = torch.load(os.path.join(data_dir, "training.pt"))#这里返回（图像数据，标签数据）
    test_data = torch.load(os.path.join(data_dir, "test.pt"))
    
    X_train, y_train = train_data#X_train保存图像数据，y_train保存标签数据
    X_test, y_test = test_data
    
    X_train = X_train.numpy().reshape(-1, 28*28)#-1是个数
    X_test = X_test.numpy().reshape(-1, 28*28)
    
    #二分类 (8->0, 9->1)
    y_train = (y_train.numpy() == 9).astype(int)
    y_test = (y_test.numpy() == 9).astype(int)
    
    scaler = StandardScaler()
    scaler.fit(X_train)#用scaler记录训练集的每个特征值的均值和标准差
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集形状: {X_train_scaled.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test_scaled.shape}, 标签形状: {y_test.shape}")
    print(f"训练集类别分布: 0(8): {np.sum(y_train==0)}, 1(9): {np.sum(y_train==1)}")
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def train_svm_models(X_train, y_train, X_test, y_test):
    print("\n--- 线性核SVM ---")
    svm_linear = SVC(kernel='linear', C=0.5, random_state=42)
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    acc_linear = accuracy_score(y_test, y_pred_linear)
    print(f"测试集准确率: {acc_linear:.4f}")
    print("分类报告:\n", classification_report(y_test, y_pred_linear))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred_linear))
    
    print("\n--- 高斯核SVM ---")
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"测试集准确率: {acc_rbf:.4f}")
    print("分类报告:\n", classification_report(y_test, y_pred_rbf))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred_rbf))
    
    return {
        "linear": {"model": svm_linear, "acc": acc_linear, "pred": y_pred_linear},
        "rbf": {"model": svm_rbf, "acc": acc_rbf, "pred": y_pred_rbf}
    }

class LinearClassifier:
    def __init__(self, input_dim, learning_rate=0.001):
        self.w = np.random.randn(input_dim) * 0.01 
        self.b = 0.0 
        self.lr = learning_rate
        
    def forward(self, X):
        return X @ self.w + self.b  #算分
    
    def predict(self, X):
        return (self.forward(X) >= 0).astype(int)  

class HingeLossClassifier(LinearClassifier):
    def compute_loss(self, X, y):
        y_hinge = 2 * y - 1  # 转换为[-1, 1]
        margin = 1 - y_hinge * self.forward(X)
        return np.mean(np.maximum(0, margin))  # hinge损失
    
    def update_weights(self, X, y):
        y_hinge = 2 * y - 1
        scores = self.forward(X)
        mask = (y_hinge * scores) < 1  # 筛选损失不为0的样本
        
        if np.any(mask):
            dw = -np.mean(y_hinge[mask, np.newaxis] * X[mask], axis=0)
            db = -np.mean(y_hinge[mask])
            
            # 梯度下降更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

class CrossEntropyClassifier(LinearClassifier):
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y):
        y_pred = self.sigmoid(self.forward(X))
        return -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
    
    def update_weights(self, X, y):
        y_pred = self.sigmoid(self.forward(X))
        error = y_pred - y 
        
        dw = (X.T @ error) / X.shape[0] 
        db = np.mean(error)
        
        self.w -= self.lr * dw
        self.b -= self.lr * db

def train_manual_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=64):
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        # 随机批次训练
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            loss = model.compute_loss(X_batch, y_batch)
            total_loss += loss
            
            model.update_weights(X_batch, y_batch)
        
        avg_loss = total_loss / (len(X_train) // batch_size)
        train_losses.append(avg_loss)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_accs.append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - 损失: {avg_loss:.4f} - 测试准确率: {acc:.4f}")
    
    y_pred_final = model.predict(X_test)
    print(f"最终测试准确率: {accuracy_score(y_test, y_pred_final):.4f}")
    print("分类报告:\n", classification_report(y_test, y_pred_final))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred_final))
    
    return model, train_losses, test_accs

def compare_results(svm_results, hinge_results, ce_results):
    #可视化
    accs = {
        "linearSVM": svm_results["linear"]["acc"],
        "rbfSVM": svm_results["rbf"]["acc"],
        "Hinge Loss": hinge_results["acc"],
        "Cross-Entropy": ce_results["acc"]
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(accs.keys(), accs.values(), color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0.8, 1.0)
    plt.ylabel("accuracy")
    plt.title("compare")
    for i, v in enumerate(accs.values()):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center')
    plt.show()
    
    # 损失曲线比较
    plt.figure(figsize=(10, 6))
    plt.plot(hinge_results["losses"], label="Hinge Loss")
    plt.plot(ce_results["losses"], label="Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("the curl of the loss")
    plt.legend()
    plt.show()

def main():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    svm_results = train_svm_models(X_train, y_train, X_test, y_test)
    
    # 训练手动实现的线性模型
    input_dim = X_train.shape[1]
    
    print("\n--- Hinge Loss线性模型 ---")
    hinge_model = HingeLossClassifier(input_dim, learning_rate=0.0001)
    hinge_trained, hinge_losses, hinge_accs = train_manual_model(
        hinge_model, X_train, y_train, X_test, y_test, epochs=100, batch_size=64
    )
    
    print("\n--- Cross-Entropy Loss线性模型 ---")
    ce_model = CrossEntropyClassifier(input_dim, learning_rate=0.001)
    ce_trained, ce_losses, ce_accs = train_manual_model(
        ce_model, X_train, y_train, X_test, y_test, epochs=100, batch_size=64
    )
    
    hinge_results = {
        "model": hinge_trained,
        "acc": hinge_accs[-1],
        "losses": hinge_losses
    }
    
    ce_results = {
        "model": ce_trained,
        "acc": ce_accs[-1],
        "losses": ce_losses
    }
    
    compare_results(svm_results, hinge_results, ce_results)

if __name__ == "__main__":
    main()