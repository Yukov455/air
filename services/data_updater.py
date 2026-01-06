"""
Сервис автоматического обновления котировок и новостей.
Поддерживает несколько источников с автоматическим переключением.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
import json
import sqlite3
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not installed, historical data download will be limited")


class DataSource(Enum):
    YAHOO = "yahoo"
    ALPHAVANTAGE = "alphavantage"
    FINNHUB = "finnhub"
    SYNTHETIC = "synthetic"


@dataclass
class UpdateStats:
    """Статистика обновлений"""
    last_update: str
    quotes_updated: int
    news_updated: int
    errors: int
    source_used: str
    duration_seconds: float


class MultiSourceDataUpdater:
    """
    Загрузчик данных с несколькими источниками и автоматическим fallback.
    """
    
    # API ключи (из переменных окружения)
    ALPHAVANTAGE_KEY = os.getenv('ALPHAVANTAGE_API_KEY', 'R89IMMWUAE7HUOCY')
    FINNHUB_KEY = os.getenv('FINNHUB_API_KEY', 'demo')
    
    # Тикеры для отслеживания
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    
    def __init__(self, db_path: str = "data/stock_analytics.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Статистика
        self.stats = {
            'total_quotes_fetched': 0,
            'total_news_fetched': 0,
            'total_errors': 0,
            'last_update': None,
            'source_stats': {s.value: {'success': 0, 'fail': 0} for s in DataSource}
        }
        
        # Порядок источников (приоритет)
        self.quote_sources = [DataSource.YAHOO, DataSource.ALPHAVANTAGE, DataSource.FINNHUB]
        self.news_sources = [DataSource.FINNHUB, DataSource.ALPHAVANTAGE]
        
        self._ensure_tables()
        logger.info("MultiSourceDataUpdater initialized")
    
    def _ensure_tables(self):
        """Создание таблиц если не существуют"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица котировок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                source TEXT,
                fetched_at TEXT,
                UNIQUE(ticker, date)
            )
        ''')
        
        # Таблица новостей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                title TEXT NOT NULL,
                summary TEXT,
                url TEXT,
                source TEXT,
                published_at TEXT,
                sentiment REAL,
                fetched_at TEXT,
                UNIQUE(title, published_at)
            )
        ''')
        
        # Таблица статистики обновлений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS update_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                quotes_updated INTEGER,
                news_updated INTEGER,
                errors INTEGER,
                source_used TEXT,
                duration_seconds REAL
            )
        ''')
        
        # Таблица метрик модели
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                training_samples INTEGER,
                validation_samples INTEGER,
                feature_count INTEGER,
                model_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ==================== КОТИРОВКИ ====================
    
    def fetch_quotes(self, ticker: str) -> Tuple[Optional[pd.DataFrame], DataSource]:
        """
        Получение котировок с автоматическим fallback.
        Возвращает DataFrame и использованный источник.
        """
        for source in self.quote_sources:
            try:
                df = None
                
                if source == DataSource.YAHOO:
                    df = self._fetch_yahoo_quotes(ticker)
                elif source == DataSource.ALPHAVANTAGE:
                    df = self._fetch_alphavantage_quotes(ticker)
                elif source == DataSource.FINNHUB:
                    df = self._fetch_finnhub_quotes(ticker)
                
                if df is not None and not df.empty:
                    self.stats['source_stats'][source.value]['success'] += 1
                    logger.info(f"Fetched {len(df)} quotes for {ticker} from {source.value}")
                    return df, source
                    
            except Exception as e:
                self.stats['source_stats'][source.value]['fail'] += 1
                logger.warning(f"Failed to fetch {ticker} from {source.value}: {e}")
                continue
        
        # Fallback на синтетические данные
        logger.warning(f"All sources failed for {ticker}, using synthetic data")
        return self._generate_synthetic_quotes(ticker), DataSource.SYNTHETIC
    
    def _fetch_yahoo_quotes(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Yahoo Finance через chart API"""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {'interval': '1d', 'range': f'{days}d'}
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        result = data.get('chart', {}).get('result', [])
        
        if not result:
            return None
        
        timestamps = result[0].get('timestamp', [])
        quote = result[0].get('indicators', {}).get('quote', [{}])[0]
        
        if not timestamps:
            return None
        
        df = pd.DataFrame({
            'date': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
            'open': quote.get('open', []),
            'high': quote.get('high', []),
            'low': quote.get('low', []),
            'close': quote.get('close', []),
            'volume': quote.get('volume', [])
        })
        df['ticker'] = ticker
        df['source'] = 'yahoo'
        
        return df
    
    def _fetch_alphavantage_quotes(self, ticker: str) -> Optional[pd.DataFrame]:
        """Alpha Vantage API"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'apikey': self.ALPHAVANTAGE_KEY,
            'outputsize': 'compact'
        }
        
        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Error Message' in data or 'Note' in data:
            return None
        
        time_series = data.get('Time Series (Daily)', {})
        
        if not time_series:
            return None
        
        rows = []
        for date, values in time_series.items():
            rows.append({
                'date': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume']),
                'ticker': ticker,
                'source': 'alphavantage'
            })
        
        return pd.DataFrame(rows)
    
    def _fetch_finnhub_quotes(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Finnhub API"""
        url = "https://finnhub.io/api/v1/stock/candle"
        
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=days)).timestamp())
        
        params = {
            'symbol': ticker,
            'resolution': 'D',
            'from': start,
            'to': end,
            'token': self.FINNHUB_KEY
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('s') != 'ok' or not data.get('t'):
            return None
        
        df = pd.DataFrame({
            'date': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in data['t']],
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v'],
            'ticker': ticker,
            'source': 'finnhub'
        })
        
        return df
    
    def _generate_synthetic_quotes(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Генерация синтетических котировок"""
        np.random.seed(hash(ticker) % 10000 + int(datetime.now().timestamp()) // 86400)
        
        base_prices = {
            'AAPL': 250, 'MSFT': 430, 'GOOGL': 190, 'AMZN': 220,
            'NVDA': 140, 'META': 600, 'TSLA': 450, 'JPM': 240,
            'V': 315, 'JNJ': 145
        }
        
        base = base_prices.get(ticker, 100)
        
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
        
        prices = [base]
        for _ in range(days - 1):
            change = np.random.normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': prices,
            'volume': [np.random.randint(10000000, 100000000) for _ in prices],
            'ticker': ticker,
            'source': 'synthetic'
        })
        
        return df
    
    # ==================== НОВОСТИ ====================
    
    def fetch_news(self, ticker: str = None, limit: int = 20) -> Tuple[List[Dict], DataSource]:
        """
        Получение новостей с автоматическим fallback.
        """
        for source in self.news_sources:
            try:
                news = None
                
                if source == DataSource.FINNHUB:
                    news = self._fetch_finnhub_news(ticker, limit)
                elif source == DataSource.ALPHAVANTAGE:
                    news = self._fetch_alphavantage_news(ticker, limit)
                
                if news:
                    self.stats['source_stats'][source.value]['success'] += 1
                    logger.info(f"Fetched {len(news)} news for {ticker or 'general'} from {source.value}")
                    return news, source
                    
            except Exception as e:
                self.stats['source_stats'][source.value]['fail'] += 1
                logger.warning(f"Failed to fetch news from {source.value}: {e}")
                continue
        
        # Fallback
        return self._generate_synthetic_news(ticker, limit), DataSource.SYNTHETIC
    
    def _fetch_finnhub_news(self, ticker: str = None, limit: int = 20) -> List[Dict]:
        """Finnhub News API"""
        if ticker:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'token': self.FINNHUB_KEY
            }
        else:
            url = "https://finnhub.io/api/v1/news"
            params = {'category': 'general', 'token': self.FINNHUB_KEY}
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        news = []
        for item in data[:limit]:
            news.append({
                'ticker': ticker or 'GENERAL',
                'title': item.get('headline', ''),
                'summary': item.get('summary', ''),
                'url': item.get('url', ''),
                'source': item.get('source', 'Finnhub'),
                'published_at': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                'sentiment': self._analyze_sentiment(item.get('headline', ''))
            })
        
        return news
    
    def _fetch_alphavantage_news(self, ticker: str = None, limit: int = 20) -> List[Dict]:
        """Alpha Vantage News API"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.ALPHAVANTAGE_KEY,
            'limit': limit
        }
        
        if ticker:
            params['tickers'] = ticker
        
        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if 'feed' not in data:
            return []
        
        news = []
        for item in data['feed'][:limit]:
            sentiment = 0
            if 'overall_sentiment_score' in item:
                sentiment = float(item['overall_sentiment_score'])
            
            news.append({
                'ticker': ticker or 'GENERAL',
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'url': item.get('url', ''),
                'source': item.get('source', 'AlphaVantage'),
                'published_at': item.get('time_published', datetime.now().isoformat()),
                'sentiment': sentiment
            })
        
        return news
    
    def _generate_synthetic_news(self, ticker: str, limit: int) -> List[Dict]:
        """Генерация синтетических новостей"""
        templates = [
            "{ticker} показывает стабильную динамику на торгах",
            "Аналитики обновили прогноз по {ticker}",
            "{ticker}: квартальные результаты соответствуют ожиданиям",
            "Инвесторы следят за развитием ситуации с {ticker}",
            "Технический анализ {ticker}: ключевые уровни",
        ]
        
        news = []
        for i in range(min(limit, len(templates))):
            news.append({
                'ticker': ticker or 'MARKET',
                'title': templates[i].format(ticker=ticker or 'Рынок'),
                'summary': f"Подробный анализ ситуации...",
                'url': '#',
                'source': 'Stock Analytics',
                'published_at': (datetime.now() - timedelta(hours=i*2)).isoformat(),
                'sentiment': np.random.uniform(-0.3, 0.3)
            })
        
        return news
    
    def _analyze_sentiment(self, text: str) -> float:
        """Простой анализ тональности"""
        positive = ['growth', 'profit', 'surge', 'gain', 'rise', 'up', 'beat', 'strong', 'рост', 'прибыль']
        negative = ['loss', 'drop', 'fall', 'decline', 'down', 'miss', 'weak', 'crash', 'падение', 'убыток']
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    # ==================== СОХРАНЕНИЕ В БД ====================
    
    def save_quotes(self, data) -> int:
        """Сохранение котировок в БД. Принимает DataFrame или список словарей."""
        # Конвертируем в список если DataFrame
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return 0
            quotes_list = data.to_dict('records')
        elif isinstance(data, list):
            if not data:
                return 0
            quotes_list = data
        else:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved = 0
        for row in quotes_list:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO quotes 
                    (ticker, date, open, high, low, close, volume, source, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['ticker'], row['date'], row['open'], row['high'],
                    row['low'], row['close'], row['volume'], 
                    row.get('source', 'synthetic'),
                    datetime.now().isoformat()
                ))
                saved += 1
            except Exception as e:
                logger.error(f"Error saving quote: {e}")
        
        conn.commit()
        conn.close()
        
        return saved
    
    def save_news(self, news: List[Dict]) -> int:
        """Сохранение новостей в БД"""
        if not news:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved = 0
        for item in news:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news 
                    (ticker, title, summary, url, source, published_at, sentiment, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item['ticker'], item['title'], item.get('summary', ''),
                    item.get('url', ''), item['source'], item['published_at'],
                    item.get('sentiment', 0), datetime.now().isoformat()
                ))
                if cursor.rowcount > 0:
                    saved += 1
            except Exception as e:
                logger.error(f"Error saving news: {e}")
        
        conn.commit()
        conn.close()
        
        return saved
    
    def save_update_stats(self, stats: UpdateStats):
        """Сохранение статистики обновления"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO update_stats 
            (timestamp, quotes_updated, news_updated, errors, source_used, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            stats.last_update, stats.quotes_updated, stats.news_updated,
            stats.errors, stats.source_used, stats.duration_seconds
        ))
        
        conn.commit()
        conn.close()
    
    # ==================== ПОЛНОЕ ОБНОВЛЕНИЕ ====================
    
    def update_all(self) -> UpdateStats:
        """
        Полное обновление всех данных.
        """
        start_time = time.time()
        quotes_updated = 0
        news_updated = 0
        errors = 0
        sources_used = set()
        
        logger.info("Starting full data update...")
        
        # Обновляем котировки для всех тикеров
        for ticker in self.TICKERS:
            try:
                df, source = self.fetch_quotes(ticker)
                if df is not None and not df.empty:
                    saved = self.save_quotes(df)
                    quotes_updated += saved
                    sources_used.add(source.value)
                time.sleep(0.5)  # Задержка между запросами
            except Exception as e:
                logger.error(f"Error updating quotes for {ticker}: {e}")
                errors += 1
        
        # Обновляем новости
        for ticker in self.TICKERS:
            try:
                news, source = self.fetch_news(ticker, limit=10)
                if news:
                    saved = self.save_news(news)
                    news_updated += saved
                    sources_used.add(source.value)
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Error updating news for {ticker}: {e}")
                errors += 1
        
        # Общие новости
        try:
            news, source = self.fetch_news(None, limit=20)
            if news:
                saved = self.save_news(news)
                news_updated += saved
        except Exception as e:
            errors += 1
        
        duration = time.time() - start_time
        
        stats = UpdateStats(
            last_update=datetime.now().isoformat(),
            quotes_updated=quotes_updated,
            news_updated=news_updated,
            errors=errors,
            source_used=','.join(sources_used),
            duration_seconds=round(duration, 2)
        )
        
        self.save_update_stats(stats)
        self.stats['last_update'] = stats.last_update
        self.stats['total_quotes_fetched'] += quotes_updated
        self.stats['total_news_fetched'] += news_updated
        self.stats['total_errors'] += errors
        
        logger.info(f"Update complete: {quotes_updated} quotes, {news_updated} news, {errors} errors in {duration:.1f}s")
        
        return stats
    
    def get_stats(self) -> Dict:
        """Получение текущей статистики"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Общее количество записей
        cursor.execute("SELECT COUNT(*) FROM quotes")
        total_quotes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM news")
        total_news = cursor.fetchone()[0]
        
        # Последние обновления
        cursor.execute("SELECT * FROM update_stats ORDER BY id DESC LIMIT 10")
        recent_updates = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_quotes': total_quotes,
            'total_news': total_news,
            'last_update': self.stats['last_update'],
            'source_stats': self.stats['source_stats'],
            'recent_updates': recent_updates
        }


    def fetch_historical_data(self, years: int = 2) -> Dict:
        """
        Загрузка исторических данных за указанное количество лет.
        
        Args:
            years: Количество лет для загрузки (по умолчанию 2)
        
        Returns:
            Статистика загрузки
        """
        logger.info(f"Starting historical data download for {years} years...")
        start_time = time.time()
        
        quotes_updated = 0
        news_updated = 0
        errors = 0
        
        # Проверяем наличие yfinance
        if yf is None:
            logger.error("yfinance not installed, cannot download historical data")
            return {
                'status': 'error',
                'message': 'yfinance not installed',
                'quotes_downloaded': 0,
                'news_downloaded': 0,
                'errors': 1,
                'duration_seconds': 0,
                'period': f'{years} years'
            }
        
        period = f"{years}y"  # Yahoo Finance формат: 1y, 2y, 5y
        
        # Загружаем котировки по одному тикеру через yfinance
        for ticker in self.TICKERS:
            ticker_quotes = 0
            try:
                logger.info(f"Fetching {years}y historical quotes for {ticker}...")
                
                # Пробуем yfinance
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period, timeout=30)
                    
                    if not hist.empty:
                        quotes = []
                        for date, row in hist.iterrows():
                            try:
                                quotes.append({
                                    'ticker': ticker,
                                    'date': date.strftime('%Y-%m-%d'),
                                    'open': float(row['Open']),
                                    'high': float(row['High']),
                                    'low': float(row['Low']),
                                    'close': float(row['Close']),
                                    'volume': int(row['Volume']) if pd.notna(row['Volume']) else 0
                                })
                            except:
                                continue
                        
                        if quotes:
                            saved = self.save_quotes(quotes)
                            ticker_quotes = saved
                            quotes_updated += saved
                            logger.info(f"Saved {saved} historical quotes for {ticker} from Yahoo")
                except Exception as e:
                    logger.warning(f"Yahoo Finance failed for {ticker}: {e}")
                
                # Если Yahoo не сработал - генерируем синтетические данные
                if ticker_quotes == 0:
                    logger.info(f"Generating synthetic historical data for {ticker}...")
                    quotes = self._generate_synthetic_history(ticker, years)
                    if quotes:
                        saved = self.save_quotes(quotes)
                        quotes_updated += saved
                        logger.info(f"Saved {saved} synthetic historical quotes for {ticker}")
                
                # Пауза между запросами
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching historical data for {ticker}: {e}")
                errors += 1
        
        # Загружаем исторические новости
        logger.info("Fetching historical news...")
        for ticker in self.TICKERS:
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ticker,
                    'limit': 200,
                    'apikey': self.ALPHAVANTAGE_KEY
                }
                
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    feed = data.get('feed', [])
                    
                    news_items = []
                    for item in feed:
                        news_items.append({
                            'ticker': ticker,
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'url': item.get('url', ''),
                            'source': item.get('source', 'alphavantage'),
                            'published_at': item.get('time_published', ''),
                            'sentiment': float(item.get('overall_sentiment_score', 0))
                        })
                    
                    if news_items:
                        saved = self.save_news(news_items)
                        news_updated += saved
                        logger.info(f"Saved {saved} historical news for {ticker}")
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching historical news for {ticker}: {e}")
                errors += 1
        
        duration = time.time() - start_time
        
        result = {
            'status': 'success',
            'quotes_downloaded': quotes_updated,
            'news_downloaded': news_updated,
            'errors': errors,
            'duration_seconds': round(duration, 2),
            'period': f'{years} years'
        }
        
        logger.info(f"Historical download complete: {quotes_updated} quotes, {news_updated} news in {duration:.1f}s")
        
        return result
    
    def _generate_synthetic_history(self, ticker: str, years: int = 2) -> List[Dict]:
        """Генерация синтетических исторических данных"""
        # Базовые цены для разных тикеров
        base_prices = {
            'AAPL': 150, 'MSFT': 350, 'GOOGL': 140, 'AMZN': 180,
            'NVDA': 500, 'META': 350, 'TSLA': 250, 'JPM': 180,
            'V': 280, 'JNJ': 160
        }
        
        base_price = base_prices.get(ticker, 100)
        quotes = []
        
        # Генерируем данные за указанное количество лет
        days = years * 252  # Торговых дней в году
        current_date = datetime.now()
        price = base_price * 0.7  # Начинаем с цены ниже текущей
        
        for i in range(days, 0, -1):
            date = current_date - timedelta(days=i)
            
            # Пропускаем выходные
            if date.weekday() >= 5:
                continue
            
            # Случайное изменение цены (тренд вверх)
            daily_return = np.random.normal(0.0005, 0.02)  # Небольшой положительный тренд
            price = price * (1 + daily_return)
            
            # OHLC
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = low + (high - low) * np.random.random()
            close = low + (high - low) * np.random.random()
            volume = int(np.random.uniform(10_000_000, 100_000_000))
            
            quotes.append({
                'ticker': ticker,
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        return quotes


# Глобальный экземпляр
_updater = None

def get_data_updater() -> MultiSourceDataUpdater:
    global _updater
    if _updater is None:
        _updater = MultiSourceDataUpdater()
    return _updater
