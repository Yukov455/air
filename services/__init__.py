"""
Services module - автоматическое обновление данных и переобучение модели.
"""

from .data_updater import MultiSourceDataUpdater, get_data_updater
from .model_trainer import UniversalModelTrainer, get_model_trainer
from .scheduler import TaskScheduler, get_scheduler

__all__ = [
    'MultiSourceDataUpdater',
    'get_data_updater',
    'UniversalModelTrainer', 
    'get_model_trainer',
    'TaskScheduler',
    'get_scheduler'
]
