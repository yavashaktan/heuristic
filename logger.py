# logger.py
import sys, time, pathlib, threading

LOG_PATH = pathlib.Path("benchmark.log")
_lock = threading.Lock()

def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {level:<5} | {msg}"
    with _lock:
        print(line, flush=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
