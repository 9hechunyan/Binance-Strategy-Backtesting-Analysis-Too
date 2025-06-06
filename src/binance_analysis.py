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

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7899'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7899'

# 初始化币安客户端（使用公共API，无需密钥）
client = Client()

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

def backtest_strategy(df, initial_capital=10000, position_size=0.5, fee=0.0001):
    """
    回测策略
    
    参数:
        df (pd.DataFrame): 包含交易信号的DataFrame
        initial_capital (float): 初始资金
        position_size (float): 每次交易使用的资金比例
        fee (float): 交易手续费率 (0.01%)
    
    返回:
        pd.DataFrame: 添加了回测结果的DataFrame
    """
    # 创建新的DataFrame来存储回测结果
    result_df = df.copy()
    result_df['position'] = 0
    result_df['capital'] = initial_capital
    result_df['holdings'] = 0
    result_df['cash'] = initial_capital
    
    # 确保索引是时间戳
    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.index = pd.to_datetime(result_df.index)
    
    for i in range(1, len(result_df)):
        if result_df['signal'].iloc[i] == 1:  # 买入信号
            if result_df['cash'].iloc[i-1] > 0:
                # 计算可买入数量
                buy_amount = result_df['cash'].iloc[i-1] * position_size
                buy_quantity = buy_amount / result_df['close'].iloc[i] * (1 - fee)
                result_df.iloc[i, result_df.columns.get_loc('holdings')] = buy_quantity
                result_df.iloc[i, result_df.columns.get_loc('cash')] = result_df['cash'].iloc[i-1] - buy_amount
                result_df.iloc[i, result_df.columns.get_loc('position')] = 1
        elif result_df['signal'].iloc[i] == -1:  # 卖出信号
            if result_df['holdings'].iloc[i-1] > 0:
                # 计算卖出金额
                sell_amount = result_df['holdings'].iloc[i-1] * result_df['close'].iloc[i] * (1 - fee)
                result_df.iloc[i, result_df.columns.get_loc('cash')] = result_df['cash'].iloc[i-1] + sell_amount
                result_df.iloc[i, result_df.columns.get_loc('holdings')] = 0
                result_df.iloc[i, result_df.columns.get_loc('position')] = 0
        else:
            result_df.iloc[i, result_df.columns.get_loc('holdings')] = result_df['holdings'].iloc[i-1]
            result_df.iloc[i, result_df.columns.get_loc('cash')] = result_df['cash'].iloc[i-1]
            result_df.iloc[i, result_df.columns.get_loc('position')] = result_df['position'].iloc[i-1]
    
    # 计算总资产
    result_df['total_assets'] = result_df['cash'] + result_df['holdings'] * result_df['close']
    
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
    days = max((end_date - start_date).days, 1)
    annual_return = (1 + total_return/100) ** (365/days) - 1
    
    # 计算最大回撤
    df['peak'] = df['total_assets'].cummax()
    df['drawdown'] = (df['total_assets'] - df['peak']) / df['peak'] * 100
    max_drawdown = df['drawdown'].min()
    
    # 计算最大回撤区间
    max_drawdown_idx = df['drawdown'].idxmin()
    peak_before_drawdown = df.loc[:max_drawdown_idx, 'total_assets'].idxmax()
    max_drawdown_period = (pd.to_datetime(max_drawdown_idx) - pd.to_datetime(peak_before_drawdown)).total_seconds() / 3600
    
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
    
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0
    
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
    os.makedirs('../reports', exist_ok=True)

    # 保存图表
    fig.write_html('../reports/trading_analysis.html')

def generate_report(df, metrics, bh_metrics):
    """
    生成HTML报告
    
    参数:
        df (pd.DataFrame): 包含回测结果的DataFrame
        metrics (dict): 包含策略指标的字典
        bh_metrics (dict): 包含Buy-and-Hold策略指标的字典
    """
    # 计算月度收益
    df['month'] = df.index.strftime('%Y-%m')
    monthly_returns = df.groupby('month')['total_assets'].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
    ).reset_index()
    monthly_returns.columns = ['月份', '收益率(%)']
    
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

    # 统计波动性最大时段
    df['return'] = df['close'].pct_change()
    volatility_by_hour = df.groupby(df.index.hour)['return'].std()
    max_volatility_hour = volatility_by_hour.idxmax()
    print(f"波动性最大时段: {max_volatility_hour}:00")

    # 统计动量最大时段
    momentum_by_hour = df.groupby(df.index.hour)['return'].mean()
    max_momentum_hour = momentum_by_hour.idxmax()
    print(f"动量最大时段: {max_momentum_hour}:00")

    # 统计信息
    context = {
        'total_return': f"{metrics['total_return']:.2f}",
        'annual_return': f"{metrics['annual_return']*100:.2f}",
        'max_drawdown': f"{metrics['max_drawdown']:.2f}",
        'win_rate': f"{metrics['win_rate']*100:.2f}",
        'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
        'avg_holding_period': f"{metrics['avg_holding_period']:.2f}",
        'monthly_returns_html': monthly_returns_html,
        'max_price': f"{df['high'].max():.2f}",
        'max_price_time': max_price_time,
        'min_price': f"{df['low'].min():.2f}",
        'min_price_time': min_price_time,
        'avg_volume': f"{df['volume'].mean():.2f}",
        'hourly_avg_volume': f"{hourly_avg_volume:.2f}",
        'total_trades': len(df[df['signal'] != 0]),
        'buy_signals': len(df[df['signal'] == 1]),
        'sell_signals': len(df[df['signal'] == -1]),
        'max_drawdown_period': f"{metrics['max_drawdown_period']:.2f}",
        'bh_total_return': f"{bh_metrics['total_return']:.2f}",
        'bh_annual_return': f"{bh_metrics['annual_return']*100:.2f}",
        'bh_max_drawdown': f"{bh_metrics['max_drawdown']:.2f}",
    }

    # 读取模板并用Jinja2渲染
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'report_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = Template(f.read())
    html_content = template.render(**context)

    # 确保reports目录存在
    os.makedirs('../reports', exist_ok=True)

    with open('../reports/trading_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """
    主函数：执行完整的策略分析流程
    """
    # 获取数据
    df = get_historical_klines(
        symbol='BTCUSDT',
        interval='4h',
        start_str='2025-01-01',
        end_str='2025-04-01'
    )
    
    if df is None:
        print("无法获取数据，请检查网络连接或代理设置")
        return
    
    # 计算指标
    df = calculate_indicators(df)
    
    # 生成信号
    df = generate_signals(df)
    
    # 回测策略（使用0.01%的手续费）
    df = backtest_strategy(df, initial_capital=10000, position_size=0.5, fee=0.0001)
    
    # 计算指标
    metrics = calculate_metrics(df)
    
    # 打印结果
    print("\n策略表现指标:")
    print(f"总收益率: {metrics['total_return']:.2f}%")
    print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
    print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
    print(f"胜率: {metrics['win_rate']*100:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"平均持仓时间: {metrics['avg_holding_period']:.2f}小时")
    
    # 绘制图表
    plot_results(df, metrics)
    
    # 计算Buy-and-Hold策略表现
    bh_metrics = calculate_buy_and_hold(df)
    
    # 生成报告
    generate_report(df, metrics, bh_metrics)

if __name__ == "__main__":
    main()