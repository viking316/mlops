# src/logger.py
import logging
import os

log_dir = "logs"
log_file = os.path.join(os.path.dirname(__file__),"..","app.log")

logging.basicConfig(
    filename=log_file,
    filemode = 'w',
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
print("Logger file has been imported")
