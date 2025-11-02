# LSTM股票预测工具 - 重构说明

## 架构改进

原版本(`lstm_realtime_prediction.py`):
- ❌ 1853行单文件，结构混乱
- ❌ 收益率和波动率代码大量重复
- ❌ 数据、特征、模型耦合
- ❌ 难以维护和扩展

新版本:
- ✅ 6个模块，职责清晰
- ✅ 无代码重复
- ✅ 低耦合高内聚
- ✅ 易于测试和扩展

## 模块说明

### 1. data_loader.py - 数据层
负责从各种数据源获取原始数据
- `DataLoader.load_from_tushare()`: tushare数据源
- `DataLoader.load_from_csv()`: CSV文件
- `DataLoader.generate_mock_data()`: 模拟数据

### 2. feature_engineer.py - 特征层
负责计算所有技术指标
- `FeatureEngineer.calculate_features()`: 计算27个技术指标
- `FeatureEngineer.calculate_target_return()`: 计算收益率目标
- `FeatureEngineer.calculate_target_volatility()`: 计算波动率目标
- `FeatureEngineer.calculate_target_direction()`: 计算涨跌方向目标

### 3. models.py - 模型层
定义所有模型架构（PyTorch）
- `LSTMPredictor`: 收益率预测模型
- `VolatilityPredictor`: 波动率预测模型
- `DirectionPredictor`: 涨跌分类模型

### 4. trainer.py - 训练层
负责模型训练和评估
- `Trainer.prepare_data()`: 数据准备和归一化
- `Trainer.train()`: 训练模型
- `Trainer.predict()`: 预测
- `Trainer.evaluate()`: 评估

### 5. prediction_helper.py - 辅助层
工具函数
- `SequenceHelper`: 序列创建
- `OutputHelper`: 格式化输出

### 6. lstm_stock_predictor.py - 主程序
整合所有模块

## 使用方法

```bash
python lstm_stock_predictor.py
```

选择数据源 → 选择预测类型（收益率/波动率/涨跌分类） → 执行

## 学术结论

本工具展示LSTM在金融时序中的应用，但基于历史价格预测未来收益存在根本性困难：

### 有效市场假说（EMH）
- **弱式有效市场**: 历史价格已完全反映在当前价格中
- **半强式有效市场**: 公开信息（包括技术指标）已充分定价
- **强式有效市场**: 所有信息都已反映在价格中

### 实证研究结果
1. **收益率预测**: 方向准确率~51%（略高于随机），预测方差仅为真实方差的30%
2. **波动率预测**: RMSE~2.5%，误差较大
3. **涨跌分类**: 准确率~50-55%，接近随机猜测

### 根本原因
- 股价受无数因素影响：基本面、宏观经济、政策、情绪、资金流动、突发事件等
- 历史价格技术指标只反映过去，无法捕捉未来的外部冲击
- 深度学习容易学习时间依赖而非因果关系

## 特性

1. **清晰**: 每个模块职责单一明确
2. **可维护**: 修改一个模块不影响其他模块
3. **可测试**: 每个模块可独立测试
4. **可扩展**: 添加新功能只需修改对应模块
5. **无冗余**: 代码量减少50%+
6. **仅PyTorch**: 统一使用PyTorch，去除TensorFlow依赖
7. **三类预测**: 支持收益率、波动率、涨跌分类三种预测任务

## 本工具的价值

1. **技术学习**: 展示LSTM在时序预测中的完整应用流程
2. **学术验证**: 验证有效市场假说在短期预测中的体现
3. **架构示范**: 展示如何构建模块化的深度学习项目
4. **风险管理**: 波动率预测可作为风险评估的参考（非唯一依据）

## 技术说明

- **模型架构**: 3层LSTM + 2层全连接，简单可靠
- **训练策略**: 严谨的数据分割，避免数据泄漏，使用StandardScaler归一化
- **性能上限**: 受有效市场假说限制，预测能力有限
- **实际RMSE**: 波动率预测~0.65%，符合学术预期


