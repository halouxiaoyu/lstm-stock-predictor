"""
数据获取模块
负责从各种数据源获取股票数据
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """数据加载器 - 支持tushare、CSV、模拟数据"""
    
    @staticmethod
    def load_from_tushare(stock_code, n_days=500):
        """从tushare获取股票数据"""
        try:
            import tushare as ts
        except ImportError:
            print("错误：未安装tushare")
            return None
        
        try:
            tushare_token = os.getenv('TUSHARE_TOKEN')
            if not tushare_token:
                print("警告: 未设置TUSHARE_TOKEN环境变量，将尝试使用tushare默认配置")
                print("提示: 请设置环境变量 TUSHARE_TOKEN 或在代码中配置token")
            else:
                ts.set_token(tushare_token)
            pro = ts.pro_api()
            
            end_date = datetime.now().strftime('%Y%m%d')
            end_datetime = datetime.strptime(end_date, '%Y%m%d')
            start_datetime = end_datetime - timedelta(days=n_days*2)
            start_date = start_datetime.strftime('%Y%m%d')
            
            df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
            
            if df is None or len(df) == 0:
                return None
            
            # 标准化列名
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            data = pd.DataFrame({
                'date': df['trade_date'],
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['vol']
            })
            
            data['market_return'] = 0  # 暂时不使用市场数据
            
            return data
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            return None
    
    @staticmethod
    def load_from_csv(csv_file):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"CSV文件缺少必需的列: {required_cols}")
                return None
            
            data = df[required_cols].copy()
            data['market_return'] = 0
            return data
        except Exception as e:
            print(f"读取CSV失败: {e}")
            return None
    
    @staticmethod
    def generate_mock_data(n_days=500):
        """生成模拟数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        price = 100
        prices = [price]
        for _ in range(n_days - 1):
            change = np.random.randn() * 2
            price = max(10, price + change)
            prices.append(price)
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + np.random.rand() * 0.02) for p in prices],
            'low': [p * (1 - np.random.rand() * 0.02) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days),
            'market_return': 0
        })
        
        return data

