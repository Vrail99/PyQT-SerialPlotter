"""
File stream manager: streams acquisition data to a CSV file in real time.
"""

import os
from typing import Optional, List

from core.error_handler import ErrorHandler


class FileStreamManager:
    """Manages real-time streaming of sample data to a CSV file."""

    def __init__(self, error_handler: ErrorHandler) -> None:
        self.error_handler = error_handler
        self.output_file = None
        self.filename: Optional[str] = None
        self.is_streaming = False

    def start_streaming(self, filename: str, column_names: List[str]) -> bool:
        if self.output_file:
            self.stop_streaming()
        try:
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
            self.output_file = open(filename, "w+")
            self.filename = filename
            self.output_file.write("Time," + ",".join(column_names) + "\n")
            self.output_file.flush()
            self.is_streaming = True
            self.error_handler.info(f"Started streaming to {filename}")
            return True
        except Exception as e:
            self.error_handler.error(f"Failed to start file streaming: {e}")
            self.is_streaming = False
            return False

    def write_sample(self, timestamp: float, values: list) -> bool:
        if not self.is_streaming or not self.output_file:
            return False
        try:
            self.output_file.write(f"{timestamp}," + ",".join(str(v) for v in values) + "\n")
            return True
        except Exception as e:
            self.error_handler.error(f"Error writing to file: {e}")
            return False

    def flush(self) -> None:
        if self.output_file and not self.output_file.closed:
            self.output_file.flush()

    def stop_streaming(self) -> bool:
        try:
            if self.output_file and not self.output_file.closed:
                self.output_file.flush()
                self.output_file.close()
            self.is_streaming = False
            if self.filename:
                self.error_handler.info(f"Stopped streaming to {self.filename}")
            return True
        except Exception as e:
            self.error_handler.error(f"Error stopping file stream: {e}")
            return False

    def is_active(self) -> bool:
        return self.is_streaming and self.output_file is not None

    def close(self) -> None:
        self.stop_streaming()
