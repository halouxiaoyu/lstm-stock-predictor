"""
LSTM股票预测工具 - 主程序
使用模块化架构，结构清晰
"""
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from models import LSTMPredictor, VolatilityPredictor, DirectionPredictor
from trainer import Trainer
from prediction_helper import SequenceHelper, OutputHelper

import numpy as np
import torch
import random
import os


# 设置随机种子
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def predict_return(stock_data):
    """预测收益率"""
    print("\n" + "=" * 60)
    print("[LSTM收益率预测]")
    print("=" * 60)
    
    # 特征工程
    print("\n[步骤1] 计算特征...")
    fe = FeatureEngineer()
    stock_data = fe.calculate_features(stock_data)
    stock_data = fe.calculate_target_return(stock_data)
    
    # 创建序列
    print("\n[步骤2] 创建时间序列...")
    seq_helper = SequenceHelper()
    X, y, dates = seq_helper.create_sequences(stock_data, fe.FEATURES, 'target_return', lookback=60)
    print(f"[OK] 创建完成: {len(X)} 个样本")
    
    if len(X) < 100:
        print("错误: 数据量不足")
        return None
    
    # 训练模型
    print("\n[步骤3] 训练模型...")
    set_all_seeds(42)
    model = LSTMPredictor()
    trainer = Trainer(model)
    
    X_train, y_train, X_test, y_test = trainer.prepare_data(X, y, train_ratio=0.8)
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    trainer.train(X_train, y_train, epochs=200)
    print("[OK] 模型训练完成")
    
    # 评估
    print("\n[步骤4] 评估模型...")
    metrics, predictions = trainer.evaluate(X_test, y_test, task_type='regression')
    print(f"  测试集 RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")
    
    # 预测明天
    print("\n[步骤5] 预测明天收益率...")
    recent_data = seq_helper.prepare_recent_data(stock_data, fe.FEATURES, lookback=60)
    recent_scaled = trainer._normalize_X(recent_data.reshape(1, 60, 27), fit=False)
    predicted_return = trainer.predict(recent_scaled)[0]
    
    current_price = stock_data.iloc[-1]['close']
    tomorrow_price = current_price * (1 + predicted_return)
    
    print(f"\n[预测结果]")
    print(f"  当前价格: {current_price:.2f}")
    print(f"  预测收益率: {predicted_return*100:.2f}%")
    print(f"  预测价格: {tomorrow_price:.2f}")
    
    return predicted_return


def predict_volatility(stock_data, forward_days=5):
    """预测波动率"""
    print("\n" + "=" * 60)
    print(f"[LSTM波动率预测] - 预测未来{forward_days}天波动率")
    print("=" * 60)
    
    # 特征工程
    print("\n[步骤1] 计算特征...")
    fe = FeatureEngineer()
    stock_data = fe.calculate_features(stock_data)
    stock_data = fe.calculate_target_volatility(stock_data, forward_days=forward_days)
    
    print(f"\n[波动率特征统计]")
    print(f"  当前波动率（20日均值）: {stock_data['volatility'].mean():.4f}")
    print(f"  目标波动率（未来{forward_days}天）: {stock_data['target_volatility'].mean():.4f}")
    
    # 创建序列
    print("\n[步骤2] 创建时间序列...")
    seq_helper = SequenceHelper()
    X, y, dates = seq_helper.create_sequences(stock_data, fe.FEATURES, 'target_volatility', lookback=60)
    print(f"[OK] 创建完成: {len(X)} 个样本")
    
    if len(X) < 100:
        print("错误: 数据量不足")
        return None
    
    # 训练模型
    print("\n[步骤3] 训练模型...")
    set_all_seeds(42)
    model = VolatilityPredictor()
    trainer = Trainer(model)
    
    X_train, y_train, X_test, y_test = trainer.prepare_data(X, y, train_ratio=0.8)
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    trainer.train(X_train, y_train, epochs=200)
    print("[OK] 模型训练完成")
    
    # 评估
    print("\n[步骤4] 评估模型...")
    metrics, predictions = trainer.evaluate(X_test, y_test, task_type='regression')
    print(f"  测试集 RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")
    
    # 显示方差分析
    print(f"  [方差分析] 真实标准差: {y_test.std():.6f}, 预测标准差: {predictions.std():.6f}")
    print(f"  [方差分析] 预测方差占比: {predictions.std()/y_test.std()*100:.1f}%")
    
    # 显示测试集对比
    OutputHelper.format_test_comparison(y_test, predictions, dates, n_samples=20)
    
    # 计算风险指标
    print("\n[步骤5] 计算风险指标...")
    volatility_high = np.percentile(predictions, 75)
    volatility_low = np.percentile(predictions, 25)
    print(f"  低风险 (< {volatility_low:.4f}): 25%分位数")
    print(f"  中风险: {volatility_low:.4f} - {volatility_high:.4f}")
    print(f"  高风险 (> {volatility_high:.4f}): 75%分位数")
    
    # 预测未来波动率
    print("\n[步骤6] 预测未来波动率...")
    recent_data = seq_helper.prepare_recent_data(stock_data, fe.FEATURES, lookback=60)
    recent_scaled = trainer._normalize_X(recent_data.reshape(1, 60, 27), fit=False)
    predicted_vol = trainer.predict(recent_scaled)[0]
    
    z_score = 1.96  # 95%置信度
    predicted_var = predicted_vol * z_score
    
    # 风险等级和建议
    if predicted_vol > volatility_high:
        risk_level = "高风险"
        position_suggestion = "建议减仓或空仓"
    elif predicted_vol < volatility_low:
        risk_level = "低风险"
        position_suggestion = "可适度加仓"
    else:
        risk_level = "中风险"
        position_suggestion = "建议保持当前仓位"
    
    current_price = stock_data.iloc[-1]['close']
    stop_loss_price = current_price * (1 - predicted_var)
    
    # 格式化输出
    OutputHelper.format_risk_results(
        predicted_vol, predicted_var, risk_level, position_suggestion,
        current_price, stop_loss_price, forward_days
    )
    
    return predicted_vol


def predict_direction(stock_data):
    """预测涨跌方向（分类）"""
    print("\n" + "=" * 60)
    print("[LSTM涨跌分类预测]")
    print("=" * 60)
    
    # 特征工程
    print("\n[步骤1] 计算特征...")
    fe = FeatureEngineer()
    stock_data = fe.calculate_features(stock_data)
    stock_data = fe.calculate_target_direction(stock_data)
    
    # 创建序列
    print("\n[步骤2] 创建时间序列...")
    seq_helper = SequenceHelper()
    X, y, dates = seq_helper.create_sequences(stock_data, fe.FEATURES, 'target_direction', lookback=60)
    print(f"[OK] 创建完成: {len(X)} 个样本")
    
    if len(X) < 100:
        print("错误: 数据量不足")
        return None
    
    # 训练模型
    print("\n[步骤3] 训练模型...")
    set_all_seeds(42)
    model = DirectionPredictor()
    trainer = Trainer(model)
    
    X_train, y_train, X_test, y_test = trainer.prepare_data(X, y, train_ratio=0.8)
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    trainer.train(X_train, y_train, epochs=200, task_type='classification')
    print("[OK] 模型训练完成")
    
    # 评估
    print("\n[步骤4] 评估模型...")
    metrics, predictions, probs = trainer.evaluate(X_test, y_test, task_type='classification')
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print("\n[混淆矩阵]")
    print(metrics['confusion_matrix'])
    
    # 预测明天
    print("\n[步骤5] 预测明天涨跌...")
    recent_data = seq_helper.prepare_recent_data(stock_data, fe.FEATURES, lookback=60)
    recent_scaled = trainer._normalize_X(recent_data.reshape(1, 60, 27), fit=False)
    predicted_logits = trainer.predict(recent_scaled)[0]
    predicted_prob = torch.sigmoid(torch.FloatTensor([predicted_logits])).item()
    predicted_direction = "涨" if predicted_prob > 0.5 else "跌"
    
    print(f"\n[预测结果]")
    print(f"  预测方向: {predicted_direction}")
    print(f"  上涨概率: {predicted_prob:.2f}")
    
    return predicted_direction, predicted_prob


def main():
    """主函数"""
    print("=" * 60)
    print("[LSTM股票预测工具]")
    print("=" * 60)
    
    # 数据源选择
    print("\n[1] 选择数据源:")
    print("   1. tushare")
    print("   2. CSV文件")
    print("   3. 模拟数据")
    choice = input("\n请选择(1-3，默认1): ").strip() or "1"
    
    # 加载数据
    dl = DataLoader()
    if choice == "1":
        stock_code = input("请输入股票代码(默认000001.SZ): ").strip() or "000001.SZ"
        stock_data = dl.load_from_tushare(stock_code)
    elif choice == "2":
        csv_file = input("请输入CSV文件路径: ").strip()
        stock_data = dl.load_from_csv(csv_file)
    else:
        stock_data = dl.generate_mock_data()
    
    if stock_data is None:
        print("无法获取数据")
        return
    
    print(f"[OK] 成功获取 {len(stock_data)} 天数据")
    
    # 预测类型选择
    print("\n[2] 选择预测类型:")
    print("   1. 预测收益率")
    print("   2. 预测波动率")
    print("   3. 预测涨跌方向（分类）")
    pred_type = input("\n请选择(1/2/3，默认2): ").strip() or "2"
    
    # 执行预测
    if pred_type == "1":
        predict_return(stock_data)
    elif pred_type == "3":
        predict_direction(stock_data)
    else:
        forward_days = input("预测未来多少天的波动率(默认5): ").strip()
        forward_days = int(forward_days) if forward_days else 5
        predict_volatility(stock_data, forward_days=forward_days)


if __name__ == "__main__":
    main()

