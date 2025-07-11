# autoppia_web_agents_subnet/utils/log_writer.py

import os
import json
from datetime import datetime
from pathlib import Path

# Toggle this to quickly enable/disable logging
ENABLE_LOGGING = True

# You can adjust this to point logs wherever you want
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def write_log(data: dict, prefix: str = "log") -> str:
    """
    Write structured JSON logs to a timestamped file.
    
    Args:
        data (dict): The dictionary to log.
        prefix (str): Prefix to use in the filename.
    
    Returns:
        str: Full path to the written log file.
    """
    if not ENABLE_LOGGING:
        return ""

    if isinstance(prefix, dict):
        prefix = prefix.get("event") or prefix.get("error") or "log"
    prefix = str(prefix).replace(" ", "_").replace("/", "_")[:40]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{prefix}-{timestamp}.json"
    filepath = LOG_DIR / filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(filepath)
    except Exception as e:
        print(f"[LogWriter] Failed to write log: {e}")
        return ""

def log_jsonl(entry: dict, filename: str = "log_stream.jsonl") -> str:
    """
    Appends a dictionary entry to a `.jsonl` file for streaming logs.

    Args:
        entry (dict): A single JSON-serializable log entry.
        filename (str): The filename inside the LOG_DIR.

    Returns:
        str: Full path to the log file.
    """
    if not ENABLE_LOGGING:
        return ""

    filepath = LOG_DIR / filename
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return str(filepath)
    except Exception as e:
        print(f"[LogWriter] Failed to append to JSONL log: {e}")
        return ""

def prune_old_logs(days: int = 7):
    """
    Deletes log files older than the given number of days in the LOG_DIR.

    Args:
        days (int): Files older than this will be removed.
    """
    if not ENABLE_LOGGING:
        return

    cutoff = datetime.now().timestamp() - (days * 86400)
    for file in LOG_DIR.glob("*.json*"):
        try:
            if file.stat().st_mtime < cutoff:
                file.unlink()
                print(f"[LogWriter] Deleted old log: {file.name}")
        except Exception as e:
            print(f"[LogWriter] Error pruning {file.name}: {e}")

