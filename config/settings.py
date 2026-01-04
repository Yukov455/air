"""
Конфигурация системы мониторинга.
Настройки загружаются из переменных окружения.
"""

import os


class Settings:
    # API Keys
    ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', 'R89IMMWUAE7HUOCY')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'demo')
    
    # Intervals (seconds)
    DATA_UPDATE_INTERVAL = int(os.getenv('DATA_UPDATE_INTERVAL', 300))  # 5 min
    MODEL_TRAIN_INTERVAL = int(os.getenv('MODEL_TRAIN_INTERVAL', 3600))  # 1 hour
    
    # Paths
    DB_PATH = os.getenv('DB_PATH', 'data/monitoring.db')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/universal_model.pkl')
    LOG_PATH = os.getenv('LOG_PATH', 'logs/monitoring.log')
    
    # Server
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Auto-start
    AUTO_START_SCHEDULER = os.getenv('AUTO_START_SCHEDULER', 'true').lower() == 'true'
    
    # Tickers to track
    TICKERS = os.getenv('TICKERS', 'AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,V,JNJ').split(',')


settings = Settings()
