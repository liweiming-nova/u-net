import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.utils import ensure_dir


class HistoryManager:
    def __init__(self, history_file: Path):
        self.history_file = history_file
        ensure_dir(str(self.history_file.parent))
        if not self.history_file.exists():
            self._write_records([])

    def _read_records(self) -> List[Dict]:
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _write_records(self, records: List[Dict]) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def list_records(self) -> List[Dict]:
        return self._read_records()

    def append_record(self, record: Dict, max_items: int = 200) -> None:
        records = self._read_records()
        records.insert(0, record)
        if len(records) > max_items:
            records = records[:max_items]
        self._write_records(records)

    def load_json(self, upload_json_file: Path, max_items: int = 200) -> int:
        with open(upload_json_file, "r", encoding="utf-8") as f:
            incoming = json.load(f)

        if not isinstance(incoming, list):
            raise ValueError("JSON format invalid: root must be a list")

        records = self._read_records()
        seen = {str(item.get("id", "")) for item in records}
        added = 0

        for item in incoming:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            if not item_id:
                continue
            if item_id in seen:
                continue
            records.append(item)
            seen.add(item_id)
            added += 1

        records.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
        if len(records) > max_items:
            records = records[:max_items]
        self._write_records(records)
        return added


def build_history_record(
    item_id: str,
    source_name: str,
    raw_url: str,
    mask_url: str,
    overlay_url: str,
) -> Dict:
    return {
        "id": item_id,
        "source_name": source_name,
        "raw_url": raw_url,
        "mask_url": mask_url,
        "overlay_url": overlay_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
