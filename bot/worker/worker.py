from celery import Celery

app = Celery("worker")
app.config_from_object(celery_config)

app.autodiscover_tasks()