"""
Планировщик автоматических задач.
Управляет обновлением данных и переобучением модели.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Callable, Optional
from loguru import logger
from dataclasses import dataclass
import json


@dataclass
class TaskStatus:
    """Статус задачи"""
    name: str
    last_run: Optional[str]
    next_run: Optional[str]
    is_running: bool
    run_count: int
    error_count: int
    last_error: Optional[str]
    last_duration: float


class TaskScheduler:
    """
    Планировщик задач для автоматического обновления данных и переобучения.
    """
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Статистика
        self.stats = {
            'started_at': None,
            'total_runs': 0,
            'total_errors': 0
        }
        
        logger.info("TaskScheduler initialized")
    
    def add_task(self, name: str, func: Callable, interval_seconds: int, 
                 run_immediately: bool = False):
        """
        Добавление задачи в планировщик.
        
        Args:
            name: Имя задачи
            func: Функция для выполнения
            interval_seconds: Интервал между запусками
            run_immediately: Запустить сразу при старте
        """
        with self._lock:
            self.tasks[name] = {
                'func': func,
                'interval': interval_seconds,
                'run_immediately': run_immediately,
                'last_run': None,
                'next_run': None,
                'is_running': False,
                'run_count': 0,
                'error_count': 0,
                'last_error': None,
                'last_duration': 0
            }
        
        logger.info(f"Task '{name}' added with interval {interval_seconds}s")
    
    def remove_task(self, name: str):
        """Удаление задачи"""
        with self._lock:
            if name in self.tasks:
                del self.tasks[name]
                logger.info(f"Task '{name}' removed")
    
    def start(self):
        """Запуск планировщика"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.stats['started_at'] = datetime.now().isoformat()
        
        # Устанавливаем время следующего запуска
        now = datetime.now()
        for name, task in self.tasks.items():
            if task['run_immediately']:
                task['next_run'] = now.isoformat()
            else:
                task['next_run'] = (now + timedelta(seconds=task['interval'])).isoformat()
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info("Scheduler started")
    
    def stop(self):
        """Остановка планировщика"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def _run_loop(self):
        """Основной цикл планировщика"""
        while self.running:
            now = datetime.now()
            
            for name, task in list(self.tasks.items()):
                if task['is_running']:
                    continue
                
                next_run = datetime.fromisoformat(task['next_run']) if task['next_run'] else now
                
                if now >= next_run:
                    # Запускаем задачу в отдельном потоке
                    thread = threading.Thread(
                        target=self._execute_task,
                        args=(name,),
                        daemon=True
                    )
                    thread.start()
            
            time.sleep(1)  # Проверяем каждую секунду
    
    def _execute_task(self, name: str):
        """Выполнение задачи"""
        task = self.tasks.get(name)
        if not task:
            return
        
        with self._lock:
            task['is_running'] = True
        
        start_time = time.time()
        
        try:
            logger.info(f"Running task '{name}'...")
            task['func']()
            
            duration = time.time() - start_time
            
            with self._lock:
                task['last_run'] = datetime.now().isoformat()
                task['next_run'] = (datetime.now() + timedelta(seconds=task['interval'])).isoformat()
                task['run_count'] += 1
                task['last_duration'] = duration
                task['is_running'] = False
                self.stats['total_runs'] += 1
            
            logger.info(f"Task '{name}' completed in {duration:.1f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            
            with self._lock:
                task['last_run'] = datetime.now().isoformat()
                task['next_run'] = (datetime.now() + timedelta(seconds=task['interval'])).isoformat()
                task['error_count'] += 1
                task['last_error'] = str(e)
                task['last_duration'] = duration
                task['is_running'] = False
                self.stats['total_errors'] += 1
            
            logger.error(f"Task '{name}' failed: {e}")
    
    def run_task_now(self, name: str):
        """Принудительный запуск задачи"""
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' not found")
        
        task = self.tasks[name]
        if task['is_running']:
            raise ValueError(f"Task '{name}' is already running")
        
        thread = threading.Thread(
            target=self._execute_task,
            args=(name,),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Task '{name}' started manually")
    
    def get_task_status(self, name: str) -> Optional[TaskStatus]:
        """Получение статуса задачи"""
        task = self.tasks.get(name)
        if not task:
            return None
        
        return TaskStatus(
            name=name,
            last_run=task['last_run'],
            next_run=task['next_run'],
            is_running=task['is_running'],
            run_count=task['run_count'],
            error_count=task['error_count'],
            last_error=task['last_error'],
            last_duration=task['last_duration']
        )
    
    def get_all_status(self) -> Dict:
        """Получение статуса всех задач"""
        tasks_status = {}
        for name in self.tasks:
            status = self.get_task_status(name)
            if status:
                tasks_status[name] = {
                    'last_run': status.last_run,
                    'next_run': status.next_run,
                    'is_running': status.is_running,
                    'run_count': status.run_count,
                    'error_count': status.error_count,
                    'last_error': status.last_error,
                    'last_duration': status.last_duration
                }
        
        return {
            'running': self.running,
            'started_at': self.stats['started_at'],
            'total_runs': self.stats['total_runs'],
            'total_errors': self.stats['total_errors'],
            'tasks': tasks_status
        }


# Глобальный экземпляр
_scheduler = None

def get_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def setup_default_tasks(scheduler: TaskScheduler, 
                        data_update_interval: int = 300,  # 5 минут
                        model_train_interval: int = 3600):  # 1 час
    """
    Настройка стандартных задач.
    
    Args:
        scheduler: Планировщик
        data_update_interval: Интервал обновления данных (секунды)
        model_train_interval: Интервал переобучения модели (секунды)
    """
    from services.data_updater import get_data_updater
    from services.model_trainer import get_model_trainer
    
    updater = get_data_updater()
    trainer = get_model_trainer()
    
    # Задача обновления данных
    scheduler.add_task(
        name='data_update',
        func=updater.update_all,
        interval_seconds=data_update_interval,
        run_immediately=True
    )
    
    # Задача переобучения модели
    def train_model():
        # Сначала проверяем что есть достаточно данных
        trainer.train(force=False)
    
    scheduler.add_task(
        name='model_training',
        func=train_model,
        interval_seconds=model_train_interval,
        run_immediately=False  # Даём время на загрузку данных
    )
    
    logger.info(f"Default tasks configured: data_update every {data_update_interval}s, model_training every {model_train_interval}s")
