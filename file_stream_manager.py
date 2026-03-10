"""
File streaming management for data export.

Handles CSV file writing and header management.
"""

import time
import os
from typing import Optional, List
from error_handler import ErrorHandler


class FileStreamManager:
    """
    Manages streaming data to CSV files.
    
    Handles file lifecycle, header writing, and data logging.
    """
    
    def __init__(self, error_handler: ErrorHandler):
        """
        Initialize file stream manager.
        
        Args:
            error_handler: Error handler for logging
        """
        self.error_handler = error_handler
        self.output_file = None
        self.filename = None
        self.is_streaming = False
    
    def start_streaming(self, filename: str, column_names: List[str]) -> bool:
        """
        Start streaming data to a file.
        
        Args:
            filename: Path to output file
            column_names: List of column names
            
        Returns:
            True if successful
        """
        try:
            # Close existing file if open
            if self.output_file:
                self.stop_streaming()
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
            
            # Open file
            self.output_file = open(filename, "w+")
            self.filename = filename
            
            # Write header
            header = "Time," + ",".join(column_names)
            self.output_file.write(header + "\n")
            self.output_file.flush()
            
            self.is_streaming = True
            self.error_handler.info(f"Started streaming to {filename}")
            return True
            
        except Exception as e:
            self.error_handler.error(f"Failed to start file streaming: {e}")
            self.is_streaming = False
            return False
    
    def write_sample(self, timestamp: float, values: list) -> bool:
        """
        Write a data sample to file.
        
        Args:
            timestamp: Sample timestamp
            values: List of channel values
            
        Returns:
            True if successful
        """
        if not self.is_streaming or not self.output_file:
            return False
        
        try:
            line = f"{timestamp}"
            for val in values:
                line += f",{val}"
            
            self.output_file.write(line + "\n")
            return True
            
        except Exception as e:
            self.error_handler.error(f"Error writing to file: {e}")
            return False
    
    def flush(self) -> None:
        """Flush file buffer."""
        if self.output_file and not self.output_file.closed:
            self.output_file.flush()
    
    def stop_streaming(self) -> bool:
        """
        Stop streaming data to file.
        
        Returns:
            True if successful
        """
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
        """Check if streaming is active."""
        return self.is_streaming and self.output_file is not None
    
    def close(self) -> None:
        """Close file stream."""
        self.stop_streaming()
