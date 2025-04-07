from .worker import app

@app.task(name="inference", bind=True)
def inference_task(self, file_name: str):
    pass