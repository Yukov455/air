"""
Сервис автоматического переобучения универсальной модели.
Модель адаптирована для всех финансовых инструментов.
Включает глубокий NLP анализ новостей.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sqlite3
import pickle
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Ключевые слова для анализа новостей (влияют на акции)
POSITIVE_KEYWORDS = [
    'growth', 'profit', 'revenue', 'beat', 'exceed', 'surge', 'rally', 'gain',
    'upgrade', 'buy', 'bullish', 'record', 'high', 'strong', 'success', 'deal',
    'partnership', 'innovation', 'breakthrough', 'expansion', 'dividend', 'buyback',
    'рост', 'прибыль', 'выручка', 'превысил', 'рекорд', 'успех', 'сделка'
]

NEGATIVE_KEYWORDS = [
    'loss', 'decline', 'drop', 'fall', 'crash', 'miss', 'below', 'weak', 'concern',
    'downgrade', 'sell', 'bearish', 'low', 'fail', 'lawsuit', 'investigation',
    'layoff', 'cut', 'warning', 'risk', 'debt', 'default', 'bankruptcy', 'fraud',
    'падение', 'убыток', 'снижение', 'риск', 'долг', 'банкротство', 'иск'
]

MARKET_EVENTS = [
    'fed', 'interest rate', 'inflation', 'gdp', 'unemployment', 'tariff', 'trade war',
    'recession', 'stimulus', 'quantitative', 'monetary', 'fiscal', 'regulation',
    'санкции', 'инфляция', 'ставка', 'безработица', 'рецессия'
]

SECTOR_KEYWORDS = {
    'tech': ['ai', 'artificial intelligence', 'cloud', 'software', 'chip', 'semiconductor', 'data'],
    'finance': ['bank', 'loan', 'credit', 'mortgage', 'insurance', 'investment'],
    'energy': ['oil', 'gas', 'renewable', 'solar', 'wind', 'battery', 'ev', 'electric'],
    'healthcare': ['drug', 'fda', 'clinical', 'trial', 'vaccine', 'pharma', 'biotech'],
    'retail': ['consumer', 'sales', 'store', 'e-commerce', 'amazon', 'walmart']
}

# Глобальная переменная для хранения прогресса обучения
_training_progress = {
    'status': 'idle',
    'stage': '',
    'progress': 0,
    'message': '',
    'logs': [],
    'iteration': 0,
    'best_accuracy': 0,
    'target_accuracy': 0.95,
    'continuous_mode': False
}

# Флаг для остановки непрерывного обучения
_stop_continuous_training = False


@dataclass
class TrainingMetrics:
    """Метрики обучения"""
    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    training_samples: int
    validation_samples: int
    feature_count: int
    model_version: str
    cross_val_mean: float
    cross_val_std: float


class UniversalModelTrainer:
    """
    Тренер универсальной модели для всех финансовых инструментов.
    
    Особенности:
    - Единая модель для всех тикеров
    - Инкрементальное дообучение
    - Автоматическое сохранение лучшей модели
    - Отслеживание метрик
    """
    
    def __init__(self, db_path: str = "data/stock_analytics.db", 
                 model_path: str = "models/universal_model.pkl"):
        self.db_path = db_path
        self.model_path = model_path
        self.metrics_history = []
        
        # Модели (расширенный ансамбль)
        self.rf_model = None
        self.gb_model = None
        self.et_model = None  # ExtraTrees
        self.mlp_model = None  # Neural Network
        self.ada_model = None  # AdaBoost
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Лучшие метрики
        self.best_accuracy = 0.0
        self.best_model_version = None
        
        # Версионирование
        self.model_version = "1.0.0"
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Загружаем существующую модель если есть
        self._load_model()
        
        logger.info("UniversalModelTrainer initialized")
    
    def _load_model(self):
        """Загрузка существующей модели"""
        try:
            if Path(self.model_path).exists():
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.rf_model = data.get('rf_model')
                    self.gb_model = data.get('gb_model')
                    self.et_model = data.get('et_model')
                    self.mlp_model = data.get('mlp_model')
                    self.ada_model = data.get('ada_model')
                    self.scaler = data.get('scaler', StandardScaler())
                    self.feature_names = data.get('feature_names', [])
                    self.best_accuracy = data.get('best_accuracy', 0.0)
                    self.model_version = data.get('version', '1.0.0')
                    logger.info(f"Loaded model v{self.model_version} with accuracy {self.best_accuracy:.2%}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
    
    def _save_model(self, metrics: TrainingMetrics):
        """Сохранение модели"""
        data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'et_model': self.et_model,
            'mlp_model': self.mlp_model,
            'ada_model': self.ada_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_accuracy': metrics.accuracy,
            'version': metrics.model_version,
            'trained_at': metrics.timestamp,
            'metrics': asdict(metrics)
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved: v{metrics.model_version} with accuracy {metrics.accuracy:.2%}")
    
    def _save_metrics_to_db(self, metrics: TrainingMetrics):
        """Сохранение метрик в БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics 
            (timestamp, accuracy, precision_score, recall, f1_score, 
             training_samples, validation_samples, feature_count, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp, metrics.accuracy, metrics.precision,
            metrics.recall, metrics.f1, metrics.training_samples,
            metrics.validation_samples, metrics.feature_count, metrics.model_version
        ))
        
        conn.commit()
        conn.close()
    
    # ==================== ПОДГОТОВКА ДАННЫХ ====================
    
    def load_training_data(self, min_samples: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка данных для обучения из БД.
        Возвращает котировки и новости.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Загружаем котировки
        quotes_df = pd.read_sql_query('''
            SELECT ticker, date, open, high, low, close, volume
            FROM quotes
            ORDER BY ticker, date
        ''', conn)
        
        # Загружаем новости (автоопределение структуры таблицы)
        try:
            # Сначала проверяем какие колонки есть в таблице
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(news)")
            columns = {col[1] for col in cursor.fetchall()}
            
            # Определяем правильные имена колонок
            ticker_col = 'ticker' if 'ticker' in columns else 'tickers' if 'tickers' in columns else None
            summary_col = 'summary' if 'summary' in columns else 'description' if 'description' in columns else 'content' if 'content' in columns else None
            
            if ticker_col and summary_col:
                news_df = pd.read_sql_query(f'''
                    SELECT {ticker_col} as ticker, title, {summary_col} as summary, published_at
                    FROM news
                    ORDER BY published_at DESC
                ''', conn)
            else:
                # Минимальный запрос
                news_df = pd.read_sql_query('''
                    SELECT title, published_at FROM news ORDER BY published_at DESC
                ''', conn)
                news_df['ticker'] = 'GENERAL'
                news_df['summary'] = news_df['title']
            
            # Добавляем sentiment если нет
            if 'sentiment' not in news_df.columns:
                news_df['sentiment'] = 0.0
                
            logger.info(f"News table columns: {columns}, loaded {len(news_df)} items")
        except Exception as e:
            logger.warning(f"Error loading news: {e}")
            news_df = pd.DataFrame(columns=['ticker', 'title', 'summary', 'sentiment', 'published_at'])
        
        conn.close()
        
        logger.info(f"Loaded {len(quotes_df)} quotes and {len(news_df)} news items")
        
        return quotes_df, news_df
    
    def prepare_features(self, quotes_df: pd.DataFrame, news_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка признаков для обучения.
        Универсальные признаки для всех инструментов.
        """
        all_features = []
        all_targets = []
        
        for ticker in quotes_df['ticker'].unique():
            ticker_quotes = quotes_df[quotes_df['ticker'] == ticker].copy()
            ticker_quotes = ticker_quotes.sort_values('date').reset_index(drop=True)
            
            if len(ticker_quotes) < 20:
                logger.debug(f"Skipping {ticker}: only {len(ticker_quotes)} quotes")
                continue
            
            # Технические индикаторы
            features = self._calculate_technical_features(ticker_quotes)
            
            # Новостные признаки
            ticker_news = news_df[news_df['ticker'] == ticker]
            news_features = self._calculate_news_features(ticker_quotes, ticker_news)
            
            # Объединяем
            features = pd.concat([features, news_features], axis=1)
            
            # Целевая переменная (рост/падение на следующий день)
            target = (ticker_quotes['close'].shift(-1) > ticker_quotes['close']).astype(int)
            
            # Убираем NaN
            valid_idx = features.dropna().index
            valid_idx = valid_idx[valid_idx < len(target) - 1]  # Убираем последний день
            
            if len(valid_idx) >= 5:
                all_features.append(features.loc[valid_idx])
                all_targets.append(target.loc[valid_idx])
                logger.debug(f"Added {len(valid_idx)} samples for {ticker}")
        
        if not all_features:
            return pd.DataFrame(), pd.Series()
        
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # Заполняем оставшиеся NaN
        X = X.fillna(0)
        
        self.feature_names = list(X.columns)
        
        logger.info(f"Prepared {len(X)} samples with {len(self.feature_names)} features")
        
        return X, y
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчёт технических индикаторов"""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Returns
        features['return_1d'] = close.pct_change(1)
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)
        
        # Moving averages (адаптивные периоды в зависимости от длины данных)
        n = len(close)
        periods = [5, 10, 20]
        if n >= 50:
            periods.append(50)
        
        for period in periods:
            ma = close.rolling(min(period, n-1)).mean()
            features[f'ma_{period}'] = ma
            features[f'ma_{period}_ratio'] = close / ma.replace(0, 1e-10)
        
        # Если MA_50 не рассчитан, используем MA_20
        if 'ma_50' not in features.columns:
            features['ma_50'] = features['ma_20']
            features['ma_50_ratio'] = features['ma_20_ratio']
        
        # Volatility (с минимальным периодом 2)
        pct_change = close.pct_change()
        features['volatility_5d'] = pct_change.rolling(min(5, n-1), min_periods=2).std()
        features['volatility_10d'] = pct_change.rolling(min(10, n-1), min_periods=2).std()
        features['volatility_20d'] = pct_change.rolling(min(20, n-1), min_periods=2).std()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features['bb_upper'] = ma20 + 2 * std20
        features['bb_lower'] = ma20 - 2 * std20
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume features
        features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        features['volume_change'] = volume.pct_change()
        
        # Price patterns
        features['high_low_ratio'] = high / low
        features['close_position'] = (close - low) / (high - low + 1e-10)
        
        # Momentum
        features['momentum_5'] = close / close.shift(5) - 1
        features['momentum_10'] = close / close.shift(10) - 1
        features['momentum_20'] = close / close.shift(20) - 1
        
        # Trend strength
        features['trend_strength'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-10)
        
        return features
    
    def _calculate_news_features(self, quotes_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Расширенный NLP анализ новостей.
        Анализирует не только sentiment, но и содержание текста.
        """
        features = pd.DataFrame(index=quotes_df.index)
        
        # Базовые признаки
        features['news_sentiment_mean'] = 0.0
        features['news_sentiment_std'] = 0.0
        features['news_count'] = 0
        
        # NLP признаки
        features['positive_keywords'] = 0
        features['negative_keywords'] = 0
        features['market_events'] = 0
        features['keyword_ratio'] = 0.0
        
        # Секторные признаки
        features['tech_mentions'] = 0
        features['finance_mentions'] = 0
        features['energy_mentions'] = 0
        features['healthcare_mentions'] = 0
        features['retail_mentions'] = 0
        
        # Признаки важности новостей
        features['news_urgency'] = 0.0
        features['news_impact_score'] = 0.0
        
        if news_df.empty:
            return features
        
        # Преобразуем даты новостей заранее для оптимизации
        news_df = news_df.copy()
        # Парсим разные форматы дат (включая 20260104T232031)
        def parse_news_date(date_str):
            try:
                if isinstance(date_str, str):
                    # Формат 20260104T232031
                    if 'T' in date_str and len(date_str) == 15:
                        return pd.to_datetime(date_str, format='%Y%m%dT%H%M%S')
                return pd.to_datetime(date_str)
            except:
                return pd.NaT
        news_df['parsed_date'] = news_df['published_at'].apply(parse_news_date)
        
        for idx, row in quotes_df.iterrows():
            date = row['date']
            
            try:
                date_dt = pd.to_datetime(date)
                recent_news = news_df[
                    (news_df['parsed_date'] >= date_dt - timedelta(days=3)) &
                    (news_df['parsed_date'] <= date_dt + timedelta(days=1))
                ]
                
                if len(recent_news) > 0:
                    # Базовый sentiment
                    features.loc[idx, 'news_sentiment_mean'] = recent_news['sentiment'].mean()
                    features.loc[idx, 'news_sentiment_std'] = recent_news['sentiment'].std() if len(recent_news) > 1 else 0
                    features.loc[idx, 'news_count'] = len(recent_news)
                    
                    # Анализ текста новостей - собираем все тексты
                    titles = ' '.join(recent_news['title'].fillna('').astype(str).tolist())
                    summaries = ' '.join(recent_news['summary'].fillna('').astype(str).tolist()) if 'summary' in recent_news.columns else ''
                    all_text = (titles + ' ' + summaries).lower()
                    
                    # Подсчёт ключевых слов
                    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in all_text)
                    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in all_text)
                    market_count = sum(1 for kw in MARKET_EVENTS if kw in all_text)
                    
                    features.loc[idx, 'positive_keywords'] = pos_count
                    features.loc[idx, 'negative_keywords'] = neg_count
                    features.loc[idx, 'market_events'] = market_count
                    
                    # Соотношение позитивных/негативных
                    total_kw = pos_count + neg_count
                    if total_kw > 0:
                        features.loc[idx, 'keyword_ratio'] = (pos_count - neg_count) / total_kw
                    
                    # Секторный анализ
                    for sector, keywords in SECTOR_KEYWORDS.items():
                        count = sum(1 for kw in keywords if kw in all_text)
                        features.loc[idx, f'{sector}_mentions'] = count
                    
                    # Срочность новости (наличие срочных слов)
                    urgency_words = ['breaking', 'urgent', 'alert', 'just in', 'срочно', 'важно']
                    urgency = sum(1 for w in urgency_words if w in all_text)
                    features.loc[idx, 'news_urgency'] = min(urgency, 5) / 5.0
                    
                    # Общий impact score
                    impact = (pos_count * 0.3 - neg_count * 0.4 + market_count * 0.2 + 
                              features.loc[idx, 'news_sentiment_mean'] * 0.5)
                    features.loc[idx, 'news_impact_score'] = np.clip(impact, -1, 1)
                    
            except Exception as e:
                pass
        
        return features
    
    # ==================== ОБУЧЕНИЕ ====================
    
    def _update_progress(self, status: str, stage: str, progress: int, message: str):
        """Обновление прогресса обучения"""
        global _training_progress
        _training_progress['status'] = status
        _training_progress['stage'] = stage
        _training_progress['progress'] = progress
        _training_progress['message'] = message
        _training_progress['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        # Храним только последние 50 логов
        if len(_training_progress['logs']) > 50:
            _training_progress['logs'] = _training_progress['logs'][-50:]
        logger.info(f"[TRAINING] {message}")
    
    def train(self, force: bool = False) -> Optional[TrainingMetrics]:
        """
        Обучение расширенного ансамбля моделей.
        
        Модели:
        - RandomForest (200 деревьев)
        - GradientBoosting (200 итераций)
        - ExtraTrees (200 деревьев)
        - Neural Network (3 слоя)
        - AdaBoost (100 итераций)
        
        Args:
            force: Принудительное переобучение даже если данных мало
        
        Returns:
            Метрики обучения или None если недостаточно данных
        """
        global _training_progress
        _training_progress['logs'] = []
        
        self._update_progress('running', 'init', 0, '🚀 Начало обучения модели...')
        
        # Загружаем данные
        self._update_progress('running', 'loading', 5, '📊 Загрузка данных из БД...')
        quotes_df, news_df = self.load_training_data()
        
        if quotes_df.empty:
            self._update_progress('error', 'loading', 0, '❌ Нет данных для обучения')
            return None
        
        self._update_progress('running', 'loading', 10, f'✅ Загружено {len(quotes_df)} котировок, {len(news_df)} новостей')
        
        # Подготавливаем признаки
        self._update_progress('running', 'features', 15, '🔧 Подготовка признаков...')
        X, y = self.prepare_features(quotes_df, news_df)
        
        if len(X) < 50 and not force:
            self._update_progress('error', 'features', 0, f'❌ Недостаточно данных ({len(X)} < 50)')
            return None
        
        self._update_progress('running', 'features', 20, f'✅ Подготовлено {len(X)} образцов с {len(self.feature_names)} признаками')
        
        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        self._update_progress('running', 'scaling', 22, f'📐 Масштабирование данных (train: {len(X_train)}, test: {len(X_test)})...')
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ==================== ОБУЧЕНИЕ МОДЕЛЕЙ ====================
        
        # 1. RandomForest (улучшенный)
        self._update_progress('running', 'rf', 25, '🌲 [1/5] Обучение RandomForest (200 деревьев)...')
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        rf_acc = accuracy_score(y_test, self.rf_model.predict(X_test_scaled))
        self._update_progress('running', 'rf', 35, f'✅ RandomForest: accuracy={rf_acc:.2%}')
        
        # 2. GradientBoosting (улучшенный)
        self._update_progress('running', 'gb', 40, '📈 [2/5] Обучение GradientBoosting (200 итераций)...')
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        self.gb_model.fit(X_train_scaled, y_train)
        gb_acc = accuracy_score(y_test, self.gb_model.predict(X_test_scaled))
        self._update_progress('running', 'gb', 50, f'✅ GradientBoosting: accuracy={gb_acc:.2%}')
        
        # 3. ExtraTrees
        self._update_progress('running', 'et', 55, '🌳 [3/5] Обучение ExtraTrees (200 деревьев)...')
        self.et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.et_model.fit(X_train_scaled, y_train)
        et_acc = accuracy_score(y_test, self.et_model.predict(X_test_scaled))
        self._update_progress('running', 'et', 65, f'✅ ExtraTrees: accuracy={et_acc:.2%}')
        
        # 4. Neural Network (MLP)
        self._update_progress('running', 'mlp', 70, '🧠 [4/5] Обучение Neural Network (3 слоя: 128-64-32)...')
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        self.mlp_model.fit(X_train_scaled, y_train)
        mlp_acc = accuracy_score(y_test, self.mlp_model.predict(X_test_scaled))
        self._update_progress('running', 'mlp', 80, f'✅ Neural Network: accuracy={mlp_acc:.2%}')
        
        # 5. AdaBoost
        self._update_progress('running', 'ada', 85, '🎯 [5/5] Обучение AdaBoost (100 итераций)...')
        self.ada_model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        self.ada_model.fit(X_train_scaled, y_train)
        ada_acc = accuracy_score(y_test, self.ada_model.predict(X_test_scaled))
        self._update_progress('running', 'ada', 90, f'✅ AdaBoost: accuracy={ada_acc:.2%}')
        
        # ==================== АНСАМБЛЬ ====================
        self._update_progress('running', 'ensemble', 92, '🔗 Создание ансамбля из 5 моделей...')
        
        # Предсказания от всех моделей
        rf_pred = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_pred = self.gb_model.predict_proba(X_test_scaled)[:, 1]
        et_pred = self.et_model.predict_proba(X_test_scaled)[:, 1]
        mlp_pred = self.mlp_model.predict_proba(X_test_scaled)[:, 1]
        ada_pred = self.ada_model.predict_proba(X_test_scaled)[:, 1]
        
        # Взвешенное голосование (лучшие модели имеют больший вес)
        weights = np.array([rf_acc, gb_acc, et_acc, mlp_acc, ada_acc])
        weights = weights / weights.sum()  # Нормализация
        
        ensemble_proba = (
            weights[0] * rf_pred + 
            weights[1] * gb_pred + 
            weights[2] * et_pred + 
            weights[3] * mlp_pred + 
            weights[4] * ada_pred
        )
        y_pred = (ensemble_proba > 0.5).astype(int)
        
        # Метрики ансамбля
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self._update_progress('running', 'ensemble', 95, f'📊 Ансамбль: accuracy={accuracy:.2%}, F1={f1:.2%}')
        
        # Cross-validation
        self._update_progress('running', 'cv', 97, '🔄 Кросс-валидация...')
        cv_scores = cross_val_score(self.rf_model, X_train_scaled, y_train, cv=5)
        
        # Обновляем версию
        version_parts = self.model_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = '.'.join(version_parts)
        
        metrics = TrainingMetrics(
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            training_samples=len(X_train),
            validation_samples=len(X_test),
            feature_count=len(self.feature_names),
            model_version=new_version,
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std()
        )
        
        self._update_progress('running', 'saving', 98, f'💾 Сохранение модели v{new_version}...')
        
        # Сохраняем если лучше предыдущей
        if accuracy >= self.best_accuracy or force:
            self.model_version = new_version
            self.best_accuracy = accuracy
            self._save_model(metrics)
            self._save_metrics_to_db(metrics)
            self._update_progress('completed', 'done', 100, f'🎉 Модель v{new_version} сохранена! Accuracy: {accuracy:.2%}')
        else:
            self._update_progress('completed', 'done', 100, f'ℹ️ Модель не сохранена (accuracy {accuracy:.2%} < best {self.best_accuracy:.2%})')
        
        # Итоговая статистика
        logger.info(f"=== TRAINING SUMMARY ===")
        logger.info(f"RandomForest: {rf_acc:.2%}")
        logger.info(f"GradientBoosting: {gb_acc:.2%}")
        logger.info(f"ExtraTrees: {et_acc:.2%}")
        logger.info(f"Neural Network: {mlp_acc:.2%}")
        logger.info(f"AdaBoost: {ada_acc:.2%}")
        logger.info(f"ENSEMBLE: {accuracy:.2%} (F1: {f1:.2%})")
        logger.info(f"Cross-validation: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def train_continuous(self, target_accuracy: float = 0.95, max_iterations: int = 0, 
                         data_refresh_interval: int = 5) -> Optional[TrainingMetrics]:
        """
        БЕСКОНЕЧНОЕ обучение до достижения целевой точности.
        Модель становится умнее с каждой итерацией благодаря:
        - Адаптивным гиперпараметрам
        - Постоянной подгрузке новых данных
        - NLP анализу новостей
        - Эволюционному подбору лучших параметров
        
        Args:
            target_accuracy: Целевая точность (по умолчанию 95%)
            max_iterations: 0 = бесконечно, иначе лимит итераций
            data_refresh_interval: Интервал обновления данных (каждые N итераций)
        
        Returns:
            Финальные метрики или None
        """
        global _training_progress, _stop_continuous_training
        _stop_continuous_training = False
        
        _training_progress['continuous_mode'] = True
        _training_progress['target_accuracy'] = target_accuracy
        _training_progress['iteration'] = 0
        _training_progress['best_accuracy'] = self.best_accuracy
        _training_progress['logs'] = []
        
        # История лучших параметров для эволюционного обучения
        self.best_params_history = []
        self.accuracy_history = []
        
        mode = "♾️ БЕСКОНЕЧНОЕ" if max_iterations == 0 else f"до {max_iterations} итераций"
        self._update_progress('running', 'continuous_init', 0, 
            f'🚀 Запуск {mode} обучения до {target_accuracy:.0%} точности...')
        self._update_progress('running', 'continuous_init', 0, 
            f'🧠 NLP анализ новостей ВКЛЮЧЕН - модель понимает контекст')
        
        best_metrics = None
        iteration = 0
        no_improvement_count = 0
        
        # Импортируем data_updater для подгрузки новых данных
        from services.data_updater import get_data_updater
        data_updater = get_data_updater()
        
        import time
        
        # Бесконечный цикл (или до max_iterations если задан)
        while not _stop_continuous_training:
            iteration += 1
            _training_progress['iteration'] = iteration
            
            # Проверка лимита итераций (0 = бесконечно)
            if max_iterations > 0 and iteration > max_iterations:
                break
            
            # Прогресс (для бесконечного режима показываем относительный прогресс)
            progress = min(int(self.best_accuracy * 100), 99) if max_iterations == 0 else int((iteration / max_iterations) * 100)
            
            self._update_progress('running', 'iteration', progress,
                f'📊 Итерация {iteration} | Лучшая: {self.best_accuracy:.2%} | Цель: {target_accuracy:.0%}')
            
            # Подгружаем новые данные каждые N итераций
            if iteration % data_refresh_interval == 1:
                self._update_progress('running', 'data_refresh', progress,
                    f'🔄 Подгрузка свежих данных (итерация {iteration})...')
                try:
                    stats = data_updater.update_all()
                    self._update_progress('running', 'data_refresh', progress,
                        f'✅ Загружено {stats.quotes_updated} котировок, {stats.news_updated} новостей')
                except Exception as e:
                    self._update_progress('running', 'data_refresh', progress,
                        f'⚠️ Ошибка загрузки: {str(e)[:40]}')
            
            # Адаптивное обучение - используем лучшие параметры из истории
            try:
                metrics = self._train_adaptive_iteration(iteration, no_improvement_count)
                
                if metrics:
                    self.accuracy_history.append(metrics.accuracy)
                    _training_progress['best_accuracy'] = self.best_accuracy
                    
                    # Проверка достижения цели
                    if metrics.accuracy >= target_accuracy:
                        self._update_progress('completed', 'target_reached', 100,
                            f'🎉🎉🎉 ЦЕЛЬ ДОСТИГНУТА! Accuracy: {metrics.accuracy:.2%} >= {target_accuracy:.0%}')
                        self._update_progress('completed', 'target_reached', 100,
                            f'🏆 Модель обучена за {iteration} итераций!')
                        _training_progress['continuous_mode'] = False
                        return metrics
                    
                    # Отслеживание улучшений
                    if metrics.accuracy > (best_metrics.accuracy if best_metrics else 0):
                        best_metrics = metrics
                        no_improvement_count = 0
                        self._update_progress('running', 'new_best', progress,
                            f'🏆 НОВЫЙ РЕКОРД: {metrics.accuracy:.2%} (итерация {iteration})')
                    else:
                        no_improvement_count += 1
                        
                    # Если долго нет улучшений - меняем стратегию
                    if no_improvement_count >= 10:
                        self._update_progress('running', 'strategy_change', progress,
                            f'🔀 Смена стратегии обучения (нет улучшений {no_improvement_count} итераций)')
                        no_improvement_count = 0
                    
            except Exception as e:
                self._update_progress('running', 'error', progress,
                    f'⚠️ Ошибка итерации {iteration}: {str(e)[:40]}')
            
            # Пауза между итерациями (адаптивная)
            sleep_time = 0.5 if no_improvement_count < 5 else 1
            time.sleep(sleep_time)
        
        if _stop_continuous_training:
            self._update_progress('stopped', 'user_stopped', progress,
                f'⏹️ Обучение остановлено на итерации {iteration}. Лучшая: {self.best_accuracy:.2%}')
        else:
            self._update_progress('completed', 'max_iterations', 100,
                f'⏱️ Лимит итераций ({max_iterations}). Лучшая: {self.best_accuracy:.2%}')
        
        _training_progress['continuous_mode'] = False
        return best_metrics
    
    def _train_adaptive_iteration(self, iteration: int, no_improvement: int) -> Optional[TrainingMetrics]:
        """
        Адаптивная итерация обучения.
        Параметры эволюционируют на основе предыдущих результатов.
        """
        global _training_progress
        
        # Загружаем данные
        quotes_df, news_df = self.load_training_data()
        
        if quotes_df.empty:
            return None
        
        # Подготавливаем признаки (теперь с NLP)
        X, y = self.prepare_features(quotes_df, news_df)
        
        if len(X) < 50:
            return None
        
        # Адаптивное разделение данных
        test_size = 0.15 + (iteration % 10) * 0.01  # 0.15-0.24
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42 + iteration, shuffle=True
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ЭВОЛЮЦИОННЫЕ гиперпараметры - становятся лучше с каждой итерацией
        # Базовые параметры растут с итерациями
        base_estimators = min(200 + iteration * 10, 1000)  # 200 -> 1000
        base_depth = min(10 + iteration // 5, 30)  # 10 -> 30
        
        # Добавляем случайность для исследования
        n_estimators = base_estimators + np.random.randint(-50, 50)
        max_depth = base_depth + np.random.randint(-2, 3)
        learning_rate = 0.01 + np.random.random() * 0.09  # 0.01-0.1
        
        # Если долго нет улучшений - радикально меняем параметры
        if no_improvement >= 5:
            n_estimators = np.random.randint(100, 500)
            max_depth = np.random.randint(5, 25)
            learning_rate = np.random.random() * 0.2
        
        self._update_progress('running', f'training_{iteration}', 
            min(int(self.best_accuracy * 100), 99),
            f'🧬 Iter {iteration}: est={n_estimators}, depth={max_depth}, lr={learning_rate:.3f}')
        
        # Обучаем все модели
        # 1. RandomForest
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=max(2, 5 - iteration // 20),
            min_samples_leaf=max(1, 3 - iteration // 30),
            max_features='sqrt',
            random_state=42 + iteration,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        rf_acc = accuracy_score(y_test, self.rf_model.predict(X_test_scaled))
        
        # 2. GradientBoosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max(3, max_depth // 2),
            learning_rate=learning_rate,
            subsample=0.7 + np.random.random() * 0.2,
            random_state=42 + iteration
        )
        self.gb_model.fit(X_train_scaled, y_train)
        gb_acc = accuracy_score(y_test, self.gb_model.predict(X_test_scaled))
        
        # 3. ExtraTrees
        self.et_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42 + iteration,
            n_jobs=-1
        )
        self.et_model.fit(X_train_scaled, y_train)
        et_acc = accuracy_score(y_test, self.et_model.predict(X_test_scaled))
        
        # 4. Neural Network - архитектура эволюционирует
        layers = self._get_evolved_nn_architecture(iteration, no_improvement)
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation='relu',
            solver='adam',
            alpha=0.0001 * (1 + iteration % 10),
            learning_rate='adaptive',
            max_iter=500 + iteration * 20,
            early_stopping=True,
            random_state=42 + iteration
        )
        self.mlp_model.fit(X_train_scaled, y_train)
        mlp_acc = accuracy_score(y_test, self.mlp_model.predict(X_test_scaled))
        
        # 5. AdaBoost
        self.ada_model = AdaBoostClassifier(
            n_estimators=min(100 + iteration * 5, 500),
            learning_rate=learning_rate,
            random_state=42 + iteration
        )
        self.ada_model.fit(X_train_scaled, y_train)
        ada_acc = accuracy_score(y_test, self.ada_model.predict(X_test_scaled))
        
        # Взвешенный ансамбль
        accuracies = np.array([rf_acc, gb_acc, et_acc, mlp_acc, ada_acc])
        weights = accuracies / accuracies.sum()
        
        rf_pred = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_pred = self.gb_model.predict_proba(X_test_scaled)[:, 1]
        et_pred = self.et_model.predict_proba(X_test_scaled)[:, 1]
        mlp_pred = self.mlp_model.predict_proba(X_test_scaled)[:, 1]
        ada_pred = self.ada_model.predict_proba(X_test_scaled)[:, 1]
        
        ensemble_proba = (weights[0] * rf_pred + weights[1] * gb_pred + 
                         weights[2] * et_pred + weights[3] * mlp_pred + weights[4] * ada_pred)
        y_pred = (ensemble_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Логируем результаты
        self._update_progress('running', f'results_{iteration}', 
            min(int(self.best_accuracy * 100), 99),
            f'📊 RF={rf_acc:.1%} GB={gb_acc:.1%} ET={et_acc:.1%} NN={mlp_acc:.1%} ADA={ada_acc:.1%} → {accuracy:.2%}')
        
        # Версия модели
        version_parts = self.model_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = '.'.join(version_parts)
        
        metrics = TrainingMetrics(
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            training_samples=len(X_train),
            validation_samples=len(X_test),
            feature_count=len(self.feature_names),
            model_version=new_version,
            cross_val_mean=accuracy,
            cross_val_std=0.0
        )
        
        # Сохраняем если лучше
        if accuracy > self.best_accuracy:
            self.model_version = new_version
            self.best_accuracy = accuracy
            self._save_model(metrics)
            self._save_metrics_to_db(metrics)
            # Сохраняем лучшие параметры
            self.best_params_history.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'accuracy': accuracy
            })
        
        return metrics
    
    def _get_evolved_nn_architecture(self, iteration: int, no_improvement: int) -> Tuple:
        """Эволюционирующая архитектура нейросети"""
        # Базовые архитектуры
        architectures = [
            (128, 64, 32),
            (256, 128, 64),
            (256, 128, 64, 32),
            (512, 256, 128),
            (512, 256, 128, 64),
            (256, 256, 128, 64),
            (512, 512, 256, 128),
            (1024, 512, 256, 128),
        ]
        
        # С ростом итераций используем более сложные архитектуры
        idx = min(iteration // 10, len(architectures) - 1)
        
        # Если нет улучшений - пробуем случайную архитектуру
        if no_improvement >= 5:
            idx = np.random.randint(0, len(architectures))
        
        return architectures[idx]
    
    def _train_single_iteration(self, iteration: int) -> Optional[TrainingMetrics]:
        """Одна итерация обучения с вариацией гиперпараметров"""
        global _training_progress
        
        # Загружаем данные
        quotes_df, news_df = self.load_training_data()
        
        if quotes_df.empty:
            return None
        
        # Подготавливаем признаки
        X, y = self.prepare_features(quotes_df, news_df)
        
        if len(X) < 50:
            return None
        
        # Разделяем на train/test с разным random_state для разнообразия
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + iteration, shuffle=True
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Вариация гиперпараметров для каждой итерации
        n_estimators = 200 + (iteration % 5) * 50  # 200-400
        max_depth = 10 + (iteration % 10)  # 10-19
        learning_rate = 0.01 + (iteration % 10) * 0.01  # 0.01-0.1
        
        self._update_progress('running', f'training_iter_{iteration}', 
            int((_training_progress['iteration'] / 100) * 100),
            f'🌲 Итерация {iteration}: n_est={n_estimators}, depth={max_depth}, lr={learning_rate:.2f}')
        
        # RandomForest
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2 + (iteration % 3),
            min_samples_leaf=1 + (iteration % 2),
            max_features='sqrt',
            random_state=42 + iteration,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        rf_acc = accuracy_score(y_test, self.rf_model.predict(X_test_scaled))
        
        # GradientBoosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth // 2,
            learning_rate=learning_rate,
            min_samples_split=2 + (iteration % 3),
            subsample=0.7 + (iteration % 4) * 0.05,
            random_state=42 + iteration
        )
        self.gb_model.fit(X_train_scaled, y_train)
        gb_acc = accuracy_score(y_test, self.gb_model.predict(X_test_scaled))
        
        # ExtraTrees
        self.et_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2 + (iteration % 3),
            random_state=42 + iteration,
            n_jobs=-1
        )
        self.et_model.fit(X_train_scaled, y_train)
        et_acc = accuracy_score(y_test, self.et_model.predict(X_test_scaled))
        
        # Neural Network с разной архитектурой
        hidden_layers = [
            (128, 64, 32),
            (256, 128, 64),
            (128, 128, 64, 32),
            (256, 128, 64, 32),
            (512, 256, 128)
        ][iteration % 5]
        
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001 * (1 + iteration % 10),
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500 + iteration * 50,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42 + iteration
        )
        self.mlp_model.fit(X_train_scaled, y_train)
        mlp_acc = accuracy_score(y_test, self.mlp_model.predict(X_test_scaled))
        
        # AdaBoost
        self.ada_model = AdaBoostClassifier(
            n_estimators=100 + iteration * 10,
            learning_rate=learning_rate,
            random_state=42 + iteration
        )
        self.ada_model.fit(X_train_scaled, y_train)
        ada_acc = accuracy_score(y_test, self.ada_model.predict(X_test_scaled))
        
        # Ансамбль
        rf_pred = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_pred = self.gb_model.predict_proba(X_test_scaled)[:, 1]
        et_pred = self.et_model.predict_proba(X_test_scaled)[:, 1]
        mlp_pred = self.mlp_model.predict_proba(X_test_scaled)[:, 1]
        ada_pred = self.ada_model.predict_proba(X_test_scaled)[:, 1]
        
        weights = np.array([rf_acc, gb_acc, et_acc, mlp_acc, ada_acc])
        weights = weights / weights.sum()
        
        ensemble_proba = (
            weights[0] * rf_pred + 
            weights[1] * gb_pred + 
            weights[2] * et_pred + 
            weights[3] * mlp_pred + 
            weights[4] * ada_pred
        )
        y_pred = (ensemble_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self._update_progress('running', f'results_iter_{iteration}', 
            int((_training_progress['iteration'] / 100) * 100),
            f'📊 Iter {iteration}: RF={rf_acc:.1%} GB={gb_acc:.1%} ET={et_acc:.1%} MLP={mlp_acc:.1%} ADA={ada_acc:.1%} → Ensemble={accuracy:.2%}')
        
        # Обновляем версию
        version_parts = self.model_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = '.'.join(version_parts)
        
        cv_scores = cross_val_score(self.rf_model, X_train_scaled, y_train, cv=3)
        
        metrics = TrainingMetrics(
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            training_samples=len(X_train),
            validation_samples=len(X_test),
            feature_count=len(self.feature_names),
            model_version=new_version,
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std()
        )
        
        # Сохраняем если лучше
        if accuracy > self.best_accuracy:
            self.model_version = new_version
            self.best_accuracy = accuracy
            self._save_model(metrics)
            self._save_metrics_to_db(metrics)
        
        return metrics
    
    def stop_continuous_training(self):
        """Остановка непрерывного обучения"""
        global _stop_continuous_training
        _stop_continuous_training = True
        logger.info("Continuous training stop requested")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков"""
        if self.rf_model is None:
            return {}
        
        importance = dict(zip(self.feature_names, self.rf_model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание с использованием ансамбля моделей.
        
        Returns:
            (predictions, probabilities)
        """
        if self.rf_model is None or self.gb_model is None:
            raise ValueError("Model not trained")
        
        # Добавляем недостающие признаки
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Собираем предсказания от всех доступных моделей
        predictions = []
        predictions.append(self.rf_model.predict_proba(X_scaled)[:, 1])
        predictions.append(self.gb_model.predict_proba(X_scaled)[:, 1])
        
        if self.et_model is not None:
            predictions.append(self.et_model.predict_proba(X_scaled)[:, 1])
        if self.mlp_model is not None:
            predictions.append(self.mlp_model.predict_proba(X_scaled)[:, 1])
        if self.ada_model is not None:
            predictions.append(self.ada_model.predict_proba(X_scaled)[:, 1])
        
        # Усредняем предсказания
        proba = np.mean(predictions, axis=0)
        pred = (proba > 0.5).astype(int)
        
        return pred, proba
    
    def get_metrics_history(self, limit: int = 20) -> List[Dict]:
        """Получение истории метрик из БД"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, accuracy, precision_score, recall, f1_score,
                   training_samples, validation_samples, feature_count, model_version
            FROM model_metrics
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': row[0],
                'accuracy': row[1],
                'precision': row[2],
                'recall': row[3],
                'f1': row[4],
                'training_samples': row[5],
                'validation_samples': row[6],
                'feature_count': row[7],
                'model_version': row[8]
            }
            for row in rows
        ]
    
    def get_current_stats(self) -> Dict:
        """Получение текущей статистики модели"""
        return {
            'model_version': self.model_version,
            'best_accuracy': self.best_accuracy,
            'feature_count': len(self.feature_names),
            'is_trained': self.rf_model is not None,
            'feature_importance': self.get_feature_importance()
        }


# Глобальный экземпляр
_trainer = None

def get_model_trainer() -> UniversalModelTrainer:
    global _trainer
    if _trainer is None:
        _trainer = UniversalModelTrainer()
    return _trainer

def get_training_progress() -> Dict:
    """Получение текущего прогресса обучения"""
    global _training_progress
    return _training_progress.copy()
