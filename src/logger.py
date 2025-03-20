import sqlite3
from datetime import datetime
from pathlib import Path
import os

class Logger:
    def __init__(self, model_name="YOLOv8"):
        """Initialize the logger with database connection."""
        # Get the directory of the current file
        current_dir = Path(__file__).parent
        # Get or create the 'logs' directory
        self.logs_dir = current_dir.parent / "logs"
        self.db_path = self.logs_dir / "detection_logs.db"
        
        self.ai_model_used = model_name

        # Create directory only if it doesn't exist
        if not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created logs directory at: {self.logs_dir}")
        
        # Initialize database connection
        db_exists = self.db_path.exists()
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        
        # Create table only if database is new
        if not db_exists:
            self._create_table()
            print(f"Created new database at: {self.db_path}")
        
        # Initialize session statistics
        self._reset_session_stats()

    def _create_table(self):
        """Create the detection_logs table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_objects_detected INTEGER NOT NULL,
                right_side_objects INTEGER NOT NULL,
                left_side_objects INTEGER NOT NULL,
                successful_detections INTEGER NOT NULL,
                failed_detections INTEGER NOT NULL,
                changed_side_detections INTEGER NOT NULL,
                ai_model_used TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def update_stats(self, detection_type, count=1):
        """Update session statistics."""
        if detection_type in self.session_stats:
            self.session_stats[detection_type] += count

    def log_detection(self, is_right_side=True, is_successful=True):
        """Log a single detection event."""
        self.session_stats["total_objects_detected"] += 1
        if is_right_side:
            self.session_stats["right_side_objects"] += 1
        else:
            self.session_stats["left_side_objects"] += 1
        
        if is_successful:
            self.session_stats["successful_detections"] += 1
        else:
            self.session_stats["failed_detections"] += 1

    def save_session(self):
        """Save the current session statistics to database."""
        try:
            self.cursor.execute("""
                INSERT INTO detection_logs (
                    total_objects_detected,
                    right_side_objects,
                    left_side_objects,
                    successful_detections,
                    failed_detections,
                    changed_side_detections,
                    ai_model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_stats["total_objects_detected"],
                self.session_stats["right_side_objects"],
                self.session_stats["left_side_objects"],
                self.session_stats["successful_detections"],
                self.session_stats["failed_detections"],
                self.session_stats["changed_side_detections"],
                self.session_stats["ai_model_used"]
            ))
            self.conn.commit()
            print("Session statistics saved successfully")
            
            # Reset session statistics
            self._reset_session_stats()
        except sqlite3.Error as e:
            print(f"Error saving session statistics: {e}")

    def _reset_session_stats(self):
        """Reset session statistics to initial values."""
        self.session_stats = {
            "total_objects_detected": 0,
            "right_side_objects": 0,
            "left_side_objects": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "changed_side_detections": 0,
            "ai_model_used": self.ai_model_used
        }

    def get_session_stats(self, start_date=None, end_date=None):
        """Retrieve statistics for a specific time period."""
        query = "SELECT * FROM detection_logs"
        params = []
        
        if start_date and end_date:
            query += " WHERE created_at BETWEEN ? AND ?"
            params = [start_date, end_date]
        
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def __del__(self):
        """Close database connection when object is destroyed."""
        self.conn.close()

# Test the Logger class
if __name__ == "__main__":
    logger = Logger()
    print(logger.get_session_stats())