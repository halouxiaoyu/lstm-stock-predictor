"""
预测辅助模块
负责创建序列、格式化输出等
"""
import numpy as np
import pandas as pd


class SequenceHelper:
    """序列创建辅助类"""
    
    @staticmethod
    def create_sequences(data, features, target_col, lookback=60):
        """创建时间序列数据"""
        X, y, dates = [], [], []
        
        for i in range(lookback, len(data)):
            if pd.isna(data[target_col].iloc[i]):
                continue  # 跳过NaN
            X.append(data[features].iloc[i-lookback:i].values)
            y.append(data[target_col].iloc[i])
            dates.append(data['date'].iloc[i])
        
        return np.array(X), np.array(y), np.array(dates)
    
    @staticmethod
    def prepare_recent_data(data, features, lookback=60):
        """准备最近N天的数据用于预测"""
        recent_data = data.tail(lookback)[features].values
        return recent_data


class OutputHelper:
    """输出格式化辅助类"""
    
    @staticmethod
    def format_test_comparison(y_test, predictions, dates_test, n_samples=20):
        """格式化测试集对比输出"""
        print("\n[测试集预测对比] 最近{}个样本:".format(n_samples))
        print("-" * 80)
        print(f"{'日期':<15} {'真实值':<15} {'预测值':<15} {'误差':<15} {'误差%':<10}")
        print("-" * 80)
        
        n_show = min(n_samples, len(y_test))
        recent_indices = range(len(y_test) - n_show, len(y_test))
        
        for idx in recent_indices:
            actual = y_test[idx]
            pred = predictions[idx]
            error = pred - actual
            error_pct = (error / (actual + 1e-8)) * 100
            
            date_str = str(dates_test[idx])[:10] if len(dates_test) > 0 else "N/A"
            print(f"{date_str:<15} {actual:>12.4f} {pred:>12.4f} {error:>12.4f} {error_pct:>8.2f}")
        
        print("-" * 80)
    
    @staticmethod
    def format_risk_results(predicted_vol, predicted_var, risk_level, position_suggestion, 
                           current_price, stop_loss_price, forward_days):
        """格式化风险预测结果"""
        print("\n" + "=" * 60)
        print("[波动率预测结果]")
        print("=" * 60)
        print(f"预测未来{forward_days}天波动率: {predicted_vol:.4f} ({predicted_vol*100:.2f}%)")
        print(f"预测VaR (95%置信度): {predicted_var:.4f} ({predicted_var*100:.2f}%)")
        print(f"\n[风险等级] {risk_level}")
        print(f"[仓位建议] {position_suggestion}")
        print(f"\n[止损建议]")
        print(f"  当前价格: {current_price:.2f}")
        print(f"  建议止损价: {stop_loss_price:.2f} ({predicted_var*100:.2f}% 下方)")
        print("\n[风险提示] 波动率预测仅供参考，不构成投资建议")
        print("=" * 60)

