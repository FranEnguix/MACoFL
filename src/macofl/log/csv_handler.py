import logging
from pathlib import Path
from typing import Optional


class CsvFileHandler(logging.FileHandler):
    def __init__(
        self,
        path: Path,
        header: str,
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = False,
    ):
        self.header = header
        file_exists = path.exists() and path.is_file()
        super().__init__(path, mode=mode, encoding=encoding, delay=delay)
        if not file_exists or mode == "w" or path.stat().st_size == 0:
            if not self.stream:
                self.stream = self._open()
            self.stream.write(self.header + "\n")
            self.stream.flush()
