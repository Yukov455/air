"""
Stock Analytics Monitoring System
Автоматическое обновление данных и переобучение модели.
"""

import os
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
from loguru import logger
import uvicorn

# Настройка логирования
logger.add("logs/monitoring.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Stock Analytics Monitoring",
    description="Система мониторинга и автоматического обучения",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные сервисы
_scheduler = None
_data_updater = None
_model_trainer = None


def get_services():
    """Lazy loading сервисов"""
    global _scheduler, _data_updater, _model_trainer
    
    if _data_updater is None:
        from services.data_updater import MultiSourceDataUpdater
        _data_updater = MultiSourceDataUpdater(db_path="data/monitoring.db")
        logger.info("DataUpdater initialized")
    
    if _model_trainer is None:
        from services.model_trainer import UniversalModelTrainer
        _model_trainer = UniversalModelTrainer(
            db_path="data/monitoring.db",
            model_path="models/universal_model.pkl"
        )
        logger.info("ModelTrainer initialized")
    
    if _scheduler is None:
        from services.scheduler import TaskScheduler
        _scheduler = TaskScheduler()
        
        # Настраиваем задачи
        _scheduler.add_task(
            name='data_update',
            func=_data_updater.update_all,
            interval_seconds=int(os.getenv('DATA_UPDATE_INTERVAL', 300)),
            run_immediately=True
        )
        
        _scheduler.add_task(
            name='model_training',
            func=lambda: _model_trainer.train(force=False),
            interval_seconds=int(os.getenv('MODEL_TRAIN_INTERVAL', 3600)),
            run_immediately=False
        )
        
        logger.info("Scheduler initialized with tasks")
    
    return _scheduler, _data_updater, _model_trainer


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Главная страница - дашборд"""
    return FileResponse("webapp/dashboard.html")


@app.get("/health")
async def health_check():
    """Health check для Docker"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/status")
async def get_status():
    """Статус системы"""
    scheduler, updater, trainer = get_services()
    return scheduler.get_all_status()


@app.get("/api/data_stats")
async def get_data_stats():
    """Статистика данных"""
    scheduler, updater, trainer = get_services()
    return updater.get_stats()


@app.get("/api/model_stats")
async def get_model_stats():
    """Статистика модели"""
    scheduler, updater, trainer = get_services()
    return trainer.get_current_stats()


@app.get("/api/training_history")
async def get_training_history():
    """История обучения"""
    scheduler, updater, trainer = get_services()
    return trainer.get_metrics_history(limit=20)


@app.get("/api/start")
async def start_scheduler():
    """Запуск планировщика"""
    scheduler, updater, trainer = get_services()
    scheduler.start()
    return {"status": "started", "message": "Scheduler started"}


@app.get("/api/stop")
async def stop_scheduler():
    """Остановка планировщика"""
    scheduler, updater, trainer = get_services()
    scheduler.stop()
    return {"status": "stopped", "message": "Scheduler stopped"}


@app.get("/api/run_task/{task_name}")
async def run_task(task_name: str):
    """Запуск задачи вручную"""
    scheduler, updater, trainer = get_services()
    
    try:
        scheduler.run_task_now(task_name)
        return {"status": "started", "task": task_name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/update_data")
async def trigger_data_update():
    """Ручное обновление данных"""
    scheduler, updater, trainer = get_services()
    
    try:
        stats = updater.update_all()
        return {
            "status": "success",
            "quotes_updated": stats.quotes_updated,
            "news_updated": stats.news_updated,
            "errors": stats.errors,
            "duration": stats.duration_seconds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train_model")
async def trigger_model_training():
    """Ручное обучение модели"""
    scheduler, updater, trainer = get_services()
    
    try:
        metrics = trainer.train(force=True)
        if metrics:
            return {
                "status": "success",
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "model_version": metrics.model_version
            }
        else:
            return {"status": "skipped", "message": "Not enough data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download_historical")
async def download_historical_data(years: int = 2):
    """
    Загрузка исторических данных за указанное количество лет.
    По умолчанию 2 года.
    """
    scheduler, updater, trainer = get_services()
    
    try:
        result = updater.fetch_historical_data(years=years)
        return result
    except Exception as e:
        logger.error(f"Historical download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training_progress")
async def get_training_progress():
    """Получение прогресса обучения модели в реальном времени"""
    from services.model_trainer import get_training_progress
    return get_training_progress()


@app.post("/api/train_continuous")
async def start_continuous_training(target_accuracy: float = 0.95, max_iterations: int = 0):
    """
    Запуск БЕСКОНЕЧНОГО обучения до достижения целевой точности.
    max_iterations=0 означает бесконечное обучение.
    Обучение происходит в фоновом режиме с:
    - Эволюционными гиперпараметрами
    - NLP анализом новостей
    - Автоматической подгрузкой данных
    """
    import threading
    scheduler, updater, trainer = get_services()
    
    def run_continuous():
        trainer.train_continuous(
            target_accuracy=target_accuracy,
            max_iterations=max_iterations,  # 0 = бесконечно
            data_refresh_interval=5
        )
    
    thread = threading.Thread(target=run_continuous, daemon=True)
    thread.start()
    
    mode = "♾️ БЕСКОНЕЧНОЕ" if max_iterations == 0 else f"до {max_iterations} итераций"
    return {
        "status": "started",
        "target_accuracy": target_accuracy,
        "max_iterations": max_iterations,
        "mode": mode,
        "message": f"{mode} обучение запущено. Цель: {target_accuracy:.0%}. NLP анализ включен."
    }


@app.post("/api/stop_training")
async def stop_continuous_training():
    """Остановка непрерывного обучения"""
    scheduler, updater, trainer = get_services()
    trainer.stop_continuous_training()
    return {"status": "stopping", "message": "Запрос на остановку отправлен"}


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Получение последних логов"""
    log_file = Path("logs/monitoring.log")
    
    if not log_file.exists():
        return {"logs": [], "message": "Log file not found"}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return {"logs": [line.strip() for line in recent_lines]}
    except Exception as e:
        return {"logs": [], "error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Автозапуск при старте"""
    auto_start = os.getenv('AUTO_START_SCHEDULER', 'true').lower() == 'true'
    
    if auto_start:
        scheduler, updater, trainer = get_services()
        scheduler.start()
        logger.info("Scheduler auto-started")


@app.on_event("shutdown")
async def shutdown_event():
    """Остановка при выключении"""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
