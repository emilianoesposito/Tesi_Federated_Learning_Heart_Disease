# utils/parallel_training.py
import psutil
import time
import pandas as pd
from threading import Thread

class SystemResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.history = []
        self.thread = None

    def start(self):
        self.monitoring = True
        self.history = []
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while self.monitoring:
            try:
                self.history.append({
                    'cpu': psutil.cpu_percent(),
                    'ram': psutil.virtual_memory().percent
                })
            except: pass
            time.sleep(0.5)

    def stop(self):
        self.monitoring = False
        if self.thread: self.thread.join()
        return pd.DataFrame(self.history)