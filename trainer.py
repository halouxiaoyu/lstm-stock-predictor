"""
模型训练模块
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.scaler_X = StandardScaler()
        self.scaler_y = None  # 可选：如果需要归一化target
    
    def prepare_data(self, X, y, train_ratio=0.8):
        """准备训练和测试数据（先分割再归一化，避免数据泄漏）"""
        n_samples = len(X)
        split_idx = int(n_samples * train_ratio)
        
        # 分割
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # 归一化
        X_train_scaled = self._normalize_X(X_train, fit=True)
        X_test_scaled = self._normalize_X(X_test, fit=False)
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def _normalize_X(self, X, fit=False):
        """归一化特征X"""
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        if fit:
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler_X.transform(X_reshaped)
        
        return X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    def train(self, X_train, y_train, epochs=200, lr=0.001, verbose=True, task_type='regression'):
        """训练模型"""
        if task_type == 'classification':
            # 分类任务：使用BCEWithLogitsLoss
            from sklearn.utils.class_weight import compute_class_weight
            
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            if verbose:
                print(f"  [类别权重] 类别0(跌): {class_weights[0]:.2f}, 类别1(涨): {class_weights[1]:.2f}")
        else:
            # 回归任务：使用MSE（经过测试，MSE效果最好）
            criterion = nn.MSELoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        # 去掉多余的维度，但保留至少1维
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        
        return predictions
    
    def evaluate(self, X_test, y_test, task_type='regression'):
        """评估模型"""
        predictions = self.predict(X_test)
        
        if task_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
            
            # 应用sigmoid并转换为类别
            probs = torch.sigmoid(torch.FloatTensor(predictions)).numpy()
            y_pred_binary = (probs > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary, zero_division=0)
            recall = recall_score(y_test, y_pred_binary, zero_division=0)
            f1 = f1_score(y_test, y_pred_binary, zero_division=0)
            
            # 计算AUC
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, probs)
            else:
                auc = 0.0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred_binary)
            }
            
            return metrics, predictions, probs
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            
            return {'rmse': rmse, 'mae': mae}, predictions

