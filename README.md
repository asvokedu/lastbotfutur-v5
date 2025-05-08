Setting 
1.BTCUSDT
    df['volatility'] = df['close'].rolling(window=2).std()
    df['reward_thresh'] = df['volatility'] * 0.00008
    df['risk_thresh'] = df['volatility'] * 0.00005
2.
