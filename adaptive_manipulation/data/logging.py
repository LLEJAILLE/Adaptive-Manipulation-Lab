"""Incremental CSV logging, including truncation after checkpoint restore."""

import csv
from pathlib import Path

class CSVLog:
    def __init__(self, path, fields, resume_step=None):
        path = Path(path)
        # Discard rows newer than the restored checkpoint after a crash or when
        # resuming an earlier snapshot into the same output directory.
        if path.exists() and resume_step is not None:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = [row for row in csv.DictReader(stream) if int(row["global_step"]) <= resume_step]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        exists = path.exists() and path.stat().st_size > 0
        self.stream = path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        if not exists:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.close()
