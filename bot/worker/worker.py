from celery import Celery
from bot.config import celery_config

app = Celery("worker")
app.config_from_object(celery_config)

app.autodiscover_tasks()