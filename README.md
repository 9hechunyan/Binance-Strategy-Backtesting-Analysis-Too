# 币安交易策略分析工具

这是一个基于Python的加密货币交易策略分析工具，使用币安(Binance)API获取数据，实现MACD策略的回测和分析。

## 功能特点

- 从币安获取BTCUSDT的4小时K线数据
- 计算技术指标（MACD、VWAP等）
- 实现交易策略回测
- 生成详细的策略分析报告
- 可视化交易结果

## 环境要求

- Python 3.8+
- 网络连接（需要访问币安API）
- 代理设置（如果需要）

## 安装步骤

1. 克隆或下载本项目到本地

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保您的网络环境可以访问币安API。如果需要使用代理，请在代码中修改代理设置：
```python
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7899'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7899'
```

2. 运行分析程序：
```bash
python binance_analysis.py
```

3. 查看生成的分析报告：
- `trading_analysis.html`：交互式图表
- `trading_report.html`：详细策略分析报告

## 注意事项

1. API访问限制：
   - 币安API有访问频率限制
   - 建议使用代理服务器访问
   - 如果遇到访问限制，可以：
     * 增加请求间隔时间
     * 使用代理服务器
     * 考虑使用币安的测试网络

2. 数据获取：
   - 默认获取2025年的数据（示例）
   - 实际使用时请修改为历史数据
   - 可以通过修改`main()`函数中的日期范围来调整

3. 策略参数：
   - 初始资金：10,000 USDT
   - 交易比例：50%
   - 手续费：0.01%
   - 可以根据需要调整这些参数

## 输出说明

程序会生成两个主要文件：

1. `trading_analysis.html`：
   - 包含K线图
   - MACD指标
   - 交易信号
   - 资产曲线
   - 最大回撤区间标注

2. `trading_report.html`：
   - 策略表现摘要
   - 月度收益分析
   - 交易统计
   - 与Buy-and-Hold策略对比

## 常见问题解决

1. API连接问题：
   - 检查网络连接
   - 确认代理设置
   - 验证API密钥（如果需要）

2. 数据获取失败：
   - 检查日期范围是否有效
   - 确认交易对名称正确
   - 验证网络连接状态

3. 图表生成问题：
   - 确保plotly正确安装
   - 检查浏览器兼容性
   - 验证文件写入权限 