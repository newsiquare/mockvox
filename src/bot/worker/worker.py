from celery import Celery
from bot.config import celery_config

celeryApp = Celery("worker")
celeryApp.config_from_object(celery_config)

celeryApp.autodiscover_tasks()