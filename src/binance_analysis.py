"""
币安交易策略分析工具

该工具用于分析币安交易所BTC/USDT交易对的数据，实现基于MACD的交易策略回测。
主要功能包括：
1. 获取历史K线数据
2. 计算技术指标
3. 生成交易信号
4. 执行策略回测
5. 生成分析报告

作者：[何女士]
日期：[2025-05-13]
"""

import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from jinja2 import Template
import requests
from requests.exceptions import RequestException
import time

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7899'  
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7899'  

def create_client(max_retries=3, timeout=30):
    """
    创建币安客户端，包含重试机制
    
    参数:
        max_retries (int): 最大重试次数
        timeout (int): 超时时间（秒）
    
    返回:
        Client: 币安客户端实例
    """
    for attempt in range(max_retries):
        try:
            client = Client()
            # 测试连接
            client.ping()
            return client
        except RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"无法连接到币安服务器: {str(e)}")
            print(f"连接失败，正在重试 ({attempt + 1}/{max_retries})...\n错误信息: {e}")  # Added error info for debugging
            time.sleep(2)  # 等待2秒后重试

# 初始化币安客户端
client = create_client()

def  get_historical_klines(symbol, interval, start_str, end_str):
    """
    获取历史K线数据
    
    参数:
        symbol (str): 交易对名称，如 'BTCUSDT'
        interval (str): K线间隔，如 '4h'
        start_str (str): 开始时间，格式 'YYYY-MM-DD'
        end_str (str): 结束时间，格式 'YYYY-MM-DD'
    
    返回:
        pd.DataFrame: 包含K线数据的DataFrame
    """
    try:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str
        )
        
        # 创建DataFrame并设置列名
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 转换数据类型
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)  # 设置时间戳为索引
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return None

def calculate_indicators(df):
    """
    计算技术指标
    
    参数:
        df (pd.DataFrame): 包含K线数据的DataFrame
    
    返回:
        pd.DataFrame: 添加了技术指标的DataFrame
    """
    # 计算MACD（动量震荡指标）
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # 计算VWAP（成交量加权平均价格）
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df

def generate_signals(df):
    """
    生成交易信号
    
    参数:
        df (pd.DataFrame): 包含技术指标的DataFrame
    
    返回:
        pd.DataFrame: 添加了交易信号的DataFrame
    """
    df['signal'] = 0
    # MACD金叉（买入信号）
    df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'signal'] = 1
    # MACD死叉（卖出信号）
    df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'signal'] = -1
    
    return df

def backtest_strategy(df, initial_capital=10000, position_size=0.5, fee=0.0001, slippage=0.0001):
    """
    回测策略
    
    参数:
        df (pd.DataFrame): 包含交易信号的DataFrame
        initial_capital (float): 初始资金
        position_size (float): 每次交易使用的资金比例
        fee (float): 交易手续费率 (0.01%)
        slippage (float): 滑点率 (0.01%)
    
    返回:
        pd.DataFrame: 添加了回测结果的DataFrame
    """
    # 创建新的DataFrame来存储回测结果
    result_df = df.copy()
    result_df['position'] = 0
    result_df['holdings'] = 0
    result_df['cash'] = initial_capital
    result_df['total_assets'] = 0.0  # 初始化 total_assets 列
    
    # 确保索引是时间戳
    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.index = pd.to_datetime(result_df.index)
    
    # 初始化第一行的总资产
    result_df.iloc[0, result_df.columns.get_loc('total_assets')] = initial_capital

    for i in range(1, len(result_df)):
        # 继承前一时刻的状态
        result_df.iloc[i, result_df.columns.get_loc('holdings')] = result_df['holdings'].iloc[i-1]
        result_df.iloc[i, result_df.columns.get_loc('cash')] = result_df['cash'].iloc[i-1]
        result_df.iloc[i, result_df.columns.get_loc('position')] = result_df['position'].iloc[i-1]

        if result_df['signal'].iloc[i] == 1:  # 买入信号
            if result_df['cash'].iloc[i] > 0: # 使用当前行的cash进行判断
                # 计算可买入数量，考虑滑点
                buy_amount = result_df['cash'].iloc[i] * position_size
                actual_price = result_df['close'].iloc[i] * (1 + slippage)
                # 考虑手续费和最小交易单位（0.0001 BTC）
                buy_quantity = buy_amount / actual_price * (1 - fee)
                buy_quantity = np.floor(buy_quantity * 10000) / 10000
                
                if buy_quantity > 0: # 确保买入数量大于0
                    result_df.iloc[i, result_df.columns.get_loc('holdings')] += buy_quantity
                    result_df.iloc[i, result_df.columns.get_loc('cash')] -= (buy_quantity * actual_price)
                    result_df.iloc[i, result_df.columns.get_loc('position')] = 1

        elif result_df['signal'].iloc[i] == -1:  # 卖出信号
            if result_df['holdings'].iloc[i] > 0: # 使用当前行的holdings进行判断
                # 计算卖出金额，考虑滑点
                actual_price = result_df['close'].iloc[i] * (1 - slippage)
                sell_amount = result_df['holdings'].iloc[i] * actual_price * (1 - fee)
                result_df.iloc[i, result_df.columns.get_loc('cash')] += sell_amount
                result_df.iloc[i, result_df.columns.get_loc('holdings')] = 0
                result_df.iloc[i, result_df.columns.get_loc('position')] = 0
        
        # 计算当前行的总资产
        result_df.iloc[i, result_df.columns.get_loc('total_assets')] = result_df['cash'].iloc[i] + result_df['holdings'].iloc[i] * result_df['close'].iloc[i]
    
    return result_df

def calculate_metrics(df):
    """
    计算策略指标
    
    参数:
        df (pd.DataFrame): 包含回测结果的DataFrame
    
    返回:
        dict: 包含各项指标的字典
    """
    # 计算收益率
    total_return = (df['total_assets'].iloc[-1] / df['total_assets'].iloc[0] - 1) * 100
    
    # 计算年化收益率
    start_date = pd.to_datetime(df.index[0])
    end_date = pd.to_datetime(df.index[-1])
    total_hours = (end_date - start_date).total_seconds() / 3600
    years = total_hours / (365 * 24)
    
    # 使用对数收益率计算年化收益率
    if total_return != 0:
        annual_return = (np.exp(np.log(1 + total_return/100) / years) - 1) * 100
    else:
        annual_return = 0
    
    # 计算最大回撤
    df['peak'] = df['total_assets'].cummax()
    df['drawdown'] = (df['total_assets'] - df['peak']) / df['peak'] * 100
    max_drawdown = df['drawdown'].min()
    
    # 计算最大回撤区间
    max_drawdown_idx = df['drawdown'].idxmin()
    peak_before_drawdown = df.loc[:max_drawdown_idx, 'total_assets'].idxmax()
    
    # 计算回撤持续时间
    drawdown_start = pd.to_datetime(peak_before_drawdown)
    drawdown_end = pd.to_datetime(max_drawdown_idx)
    
    # 计算回撤恢复时间
    recovery_idx = df.loc[max_drawdown_idx:].index[df.loc[max_drawdown_idx:, 'total_assets'] >= df.loc[peak_before_drawdown, 'total_assets']]
    if len(recovery_idx) > 0:
        recovery_time = pd.to_datetime(recovery_idx[0])
        max_drawdown_period = (recovery_time - drawdown_start).total_seconds() / 3600
    else:
        max_drawdown_period = (drawdown_end - drawdown_start).total_seconds() / 3600
    
    # 将持续时间向上取整到最近的4小时
    max_drawdown_period = np.ceil(max_drawdown_period / 4) * 4
    
    # 计算胜率
    trades = df[df['signal'] != 0]
    profitable_trades = len(trades[trades['total_assets'] > trades['total_assets'].shift(1)])
    win_rate = profitable_trades / len(trades) if len(trades) > 0 else 0
    
    # 计算平均持仓时间
    holding_periods = []
    entry_time = None
    
    for i in range(1, len(df)):
        if df['position'].iloc[i] == 1 and df['position'].iloc[i-1] == 0:  # 开仓
            entry_time = df.index[i]
        elif df['position'].iloc[i] == 0 and df['position'].iloc[i-1] == 1:  # 平仓
            if entry_time is not None:
                holding_period = (pd.to_datetime(df.index[i]) - pd.to_datetime(entry_time)).total_seconds() / 3600
                holding_periods.append(holding_period)
            entry_time = None
    
    # 处理最后一个持仓周期
    if entry_time is not None:
        holding_period = (pd.to_datetime(df.index[-1]) - pd.to_datetime(entry_time)).total_seconds() / 3600
        holding_periods.append(holding_period)
    
    # 计算平均持仓时间，考虑最小单位
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0
    # 将持仓时间向上取整到最近的4小时
    avg_holding_period = np.ceil(avg_holding_period / 4) * 4
    
    # 计算夏普比率
    df['daily_returns'] = df['total_assets'].pct_change()
    sharpe_ratio = np.sqrt(365) * df['daily_returns'].mean() / df['daily_returns'].std() if df['daily_returns'].std() != 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'avg_holding_period': avg_holding_period,
        'max_drawdown_period': max_drawdown_period,
        'max_drawdown_start': peak_before_drawdown,
        'max_drawdown_end': max_drawdown_idx
    }

def calculate_buy_and_hold(df):
    """
    计算Buy-and-Hold策略的表现
    
    参数:
        df (pd.DataFrame): 包含价格数据的DataFrame
    
    返回:
        dict: 包含Buy-and-Hold策略指标的字典
    """
    # 计算总收益率
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    # 计算年化收益率
    start_date = pd.to_datetime(df.index[0])
    end_date = pd.to_datetime(df.index[-1])
    days = max((end_date - start_date).days, 1)
    annual_return = (1 + total_return/100) ** (365/days) - 1
    
    # 计算最大回撤
    df['peak'] = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['peak']) / df['peak'] * 100
    max_drawdown = df['drawdown'].min()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown
    }

def plot_results(df, metrics):
    """
    绘制结果图表
    
    参数:
        df (pd.DataFrame): 包含回测结果的DataFrame
        metrics (dict): 包含策略指标的字典
    """
    # 创建子图，增加交易量
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('BTC/USDT Price', 'Volume', 'MACD', 'Portfolio Value'),
                       row_heights=[0.35, 0.15, 0.25, 0.25])

    # 添加K线图
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='BTC/USDT'),
                 row=1, col=1)

    # 添加交易量柱状图
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='lightblue'), row=2, col=1)

    # 添加买入信号
    buy_signals = df[df['signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index,
                            y=buy_signals['low'] * 0.99,
                            mode='markers',
                            marker=dict(symbol='triangle-up', size=10, color='green'),
                            name='Buy Signal'),
                 row=1, col=1)

    # 添加卖出信号
    sell_signals = df[df['signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index,
                            y=sell_signals['high'] * 1.01,
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=10, color='red'),
                            name='Sell Signal'),
                 row=1, col=1)

    # 添加MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'],
                            name='MACD', line=dict(color='blue')),
                 row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'],
                            name='Signal', line=dict(color='orange')),
                 row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'],
                        name='Histogram', marker_color='gray'),
                 row=3, col=1)

    # 添加资产曲线
    fig.add_trace(go.Scatter(x=df.index, y=df['total_assets'],
                            name='Portfolio Value', line=dict(color='green')),
                 row=4, col=1)
    
    # 添加最大回撤区间
    fig.add_trace(go.Scatter(
        x=[metrics['max_drawdown_start'], metrics['max_drawdown_end']],
        y=[df.loc[metrics['max_drawdown_start'], 'total_assets'],
           df.loc[metrics['max_drawdown_end'], 'total_assets']],
        mode='lines+markers',
        line=dict(color='red', width=2, dash='dash'),
        name='Max Drawdown Period',
        showlegend=True
    ), row=4, col=1)

    # 更新布局
    fig.update_layout(
        title='BTC/USDT Trading Strategy Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        height=1200
    )

    # 确保reports目录存在
    os.makedirs('reports', exist_ok=True)

    # 保存图表
    fig.write_html('reports/trading_analysis.html')

def generate_report(df, metrics, bh_metrics):
    """
    生成HTML报告
    
    参数:
        df (pd.DataFrame): 包含回测结果的DataFrame
        metrics (dict): 包含策略指标的字典
        bh_metrics (dict): 包含Buy-and-Hold策略指标的字典
    """
    # 数据验证
    # 过滤掉非数值型的指标，如时间戳
    numeric_metrics_for_validation = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not all(isinstance(x, (int, float)) for x in numeric_metrics_for_validation.values()):
        raise ValueError("Metrics contains non-numeric values")
    
    # 格式化数字
    formatted_metrics = {
        'total_return': f"{metrics['total_return']:.2f}%",
        'annual_return': f"{metrics['annual_return']:.2f}%",
        'max_drawdown': f"{metrics['max_drawdown']:.2f}%",
        'win_rate': f"{metrics['win_rate']:.2f}%",
        'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
        'avg_holding_period': f"{metrics['avg_holding_period']:.1f}小时",
        'max_drawdown_period': f"{metrics['max_drawdown_period']:.1f}小时"
    }
    
    # 计算月度收益
    df['month'] = df.index.strftime('%Y-%m')
    monthly_returns = df.groupby('month')['total_assets'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
    ).reset_index()
    monthly_returns.columns = ['月份', '收益率(%)']
    monthly_returns['收益率(%)'] = monthly_returns['收益率(%)'].apply(lambda x: f"{x:.2f}%")
    
    # 生成月度收益表格HTML
    monthly_returns_html = monthly_returns.to_html(index=False, classes='table')
    
    # 获取最高价和最低价出现的时间
    max_price_time = df.loc[df['high'].idxmax()].name
    min_price_time = df.loc[df['low'].idxmin()].name

    # 计算每小时平均交易量
    df['hour'] = df.index.hour
    hourly_avg_volume = df.groupby('hour')['volume'].mean().mean()

    # 统计每小时总成交量
    volume_by_hour = df.groupby(df.index.hour)['volume'].sum()
    max_volume_hour = volume_by_hour.idxmax()
    print(f"交易量最大时段: {max_volume_hour}:00")

    # 统计价格波动最大时段
    df['range'] = df['high'] - df['low']
    range_by_hour = df.groupby(df.index.hour)['range'].mean()
    max_range_hour = range_by_hour.idxmax()
    print(f"价格波动最大时段: {max_range_hour}:00")

    if 'trades' in df.columns:
        trades_by_hour = df.groupby(df.index.hour)['trades'].sum()
        max_trades_hour = trades_by_hour.idxmax()
        print(f"交易频率最高时段: {max_trades_hour}:00")

    # 统计大单交易最集中时段
    big_trade_threshold = df['volume'].quantile(0.95)  # 取前5%为大单
    df['big_trade'] = df['volume'] > big_trade_threshold
    big_trades_by_hour = df.groupby(df.index.hour)['big_trade'].sum()
    max_big_trades_hour = big_trades_by_hour.idxmax()
    print(f"大单交易最集中时段: {max_big_trades_hour}:00")
    
    # 生成HTML报告
    template = """
    <html>
    <head>
        <title>交易策略分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>交易策略分析报告</h1>
            
            <h2>策略表现指标</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                </tr>
                <tr>
                    <td>总收益率</td>
                    <td class="{}">{}</td>
                </tr>
                <tr>
                    <td>年化收益率</td>
                    <td class="{}">{}</td>
                </tr>
                <tr>
                    <td>最大回撤</td>
                    <td class="negative">{}</td>
                </tr>
                <tr>
                    <td>胜率</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td>夏普比率</td>
                    <td class="{}">{}</td>
                </tr>
                <tr>
                    <td>平均持仓时间</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td>最大回撤持续时间</td>
                    <td>{}</td>
                </tr>
            </table>
            
            <h2>与Buy-and-Hold策略对比</h2>
            <table>
                <tr>
                    <th>指标</th>
                    <th>策略</th>
                    <th>Buy-and-Hold</th>
                </tr>
                <tr>
                    <td>总收益率</td>
                    <td class="{}">{}</td>
                    <td class="{}">{}</td>
                </tr>
                <tr>
                    <td>年化收益率</td>
                    <td class="{}">{}</td>
                    <td class="{}">{}</td>
                </tr>
                <tr>
                    <td>最大回撤</td>
                    <td class="negative">{}</td>
                    <td class="negative">{}</td>
                </tr>
            </table>
            
            <h2>月度收益</h2>
            {}
            
            <h2>图表分析</h2>
            <p>请查看生成的 <a href="reports/trading_analysis.html">trading_analysis.html</a> 文件获取详细图表分析。</p>
        </div>
    </body>
    </html>
    """
    
    # 格式化报告
    report = template.format(
        'positive' if metrics['total_return'] > 0 else 'negative',
        formatted_metrics['total_return'],
        'positive' if metrics['annual_return'] > 0 else 'negative',
        formatted_metrics['annual_return'],
        formatted_metrics['max_drawdown'],
        formatted_metrics['win_rate'],
        'positive' if metrics['sharpe_ratio'] > 0 else 'negative',
        formatted_metrics['sharpe_ratio'],
        formatted_metrics['avg_holding_period'],
        formatted_metrics['max_drawdown_period'],
        'positive' if metrics['total_return'] > 0 else 'negative',
        formatted_metrics['total_return'],
        'positive' if bh_metrics['total_return'] > 0 else 'negative',
        f"{bh_metrics['total_return']:.2f}%",
        'positive' if metrics['annual_return'] > 0 else 'negative',
        formatted_metrics['annual_return'],
        'positive' if bh_metrics['annual_return'] > 0 else 'negative',
        f"{bh_metrics['annual_return']:.2f}%",
        formatted_metrics['max_drawdown'],
        f"{bh_metrics['max_drawdown']:.2f}%",
        monthly_returns_html
    )
    
    # 保存报告
    with open('reports/trading_report.html', 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    try:
        # 获取历史数据
        print("正在获取历史数据...")
        klines = client.get_historical_klines(
            "BTCUSDT", 
            Client.KLINE_INTERVAL_1HOUR,
            "1 Jan, 2023"
        )
        
        print("正在处理数据...")
        # 转换为DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # 数据清洗和预处理
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # 转换数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        print("正在计算技术指标...")
        # 计算技术指标
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
        df['macd_hist'] = ta.macd(df['close'])['MACDh_12_26_9']
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        print("正在生成交易信号...")
        # 生成交易信号
        df['signal'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
        df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1
        
        print("正在回测策略...")
        # 回测策略
        results = backtest_strategy(df, initial_capital=10000, position_size=0.1, slippage=0.001)
        
        print("正在计算策略指标...")
        # 计算策略指标
        metrics = calculate_metrics(results)
        
        # 计算Buy-and-Hold策略指标
        bh_metrics = calculate_buy_and_hold(df)
        
        print("正在生成报告...")
        # 生成报告
        generate_report(results, metrics, bh_metrics)
        
        print("正在生成图表...")
        # 生成图表
        plot_results(results, metrics)
        
        print("分析完成！请查看 reports 目录下的报告文件。")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()