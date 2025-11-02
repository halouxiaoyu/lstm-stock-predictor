"""
特征工程模块
负责计算所有技术指标
"""
import numpy as np
import pandas as pd


class FeatureEngineer:
    """特征工程师 - 计算27个技术指标"""
    
    # 特征列表
    FEATURES = [
        'open', 'high', 'low', 'close', 'volume', 
        'ma5', 'ma10', 'ma20', 'returns', 'volatility',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'volume_ma20', 'volume_ratio', 'momentum_5', 'momentum_10', 'price_position',
        'atr_14', 'atr_ratio', 'price_change_pct', 'volume_change_pct'
    ]
    
    @staticmethod
    def calculate_features(df):
        """计算所有技术指标"""
        df = df.copy()
        
        # 处理异常值
        df = df.replace(0, np.nan)
        
        # MA
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # Returns & Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_mid'] - (bb_std_val * bb_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-10)
        
        # 成交量
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1e-10)
        
        # Momentum
        df['momentum_5'] = df['returns'].rolling(window=5).sum()
        df['momentum_10'] = df['returns'].rolling(window=10).sum()
        
        # Price Position
        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-10)
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # Change Rates
        df['price_change_pct'] = df['close'].pct_change()
        df['volume_change_pct'] = df['volume'].pct_change()
        
        # Fill NaN
        df = df.bfill().ffill()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill().ffill()
        
        return df
    
    @staticmethod
    def calculate_target_return(df):
        """计算目标变量：收益率"""
        df['target_return'] = df['close'].pct_change().shift(-1)
        return df
    
    @staticmethod
    def calculate_target_volatility(df, forward_days=5):
        """计算目标变量：波动率"""
        df['target_volatility'] = np.nan
        for i in range(len(df) - forward_days):
            future_ret = df['returns'].iloc[i+1:i+1+forward_days]
            if len(future_ret) == forward_days and not future_ret.isna().all():
                df.iloc[i, df.columns.get_loc('target_volatility')] = future_ret.std()
        return df
    
    @staticmethod
    def calculate_target_direction(df):
        """计算目标变量：涨跌方向（分类）"""
        df['target_direction'] = (df['close'].pct_change().shift(-1) > 0).astype(float)
        return df

