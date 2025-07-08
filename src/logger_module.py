import sqlite3
from datetime import datetime
from pathlib import Path
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

class Logger:
    def __init__(self, model_name="YOLOv8", user_id=None):
        """Initialize the logger with database connection."""
        # Get the directory of the current file
        current_dir = Path(__file__).parent
        # Get or create the 'logs' directory
        self.logs_dir = current_dir.parent / "logs"
        self.db_path = self.logs_dir / "detection_logs.db"
        
        # Create directory only if it doesn't exist
        if not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created logs directory at: {self.logs_dir}")
        
        # Initialize database connection
        db_exists = self.db_path.exists()
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        
        # Store user information
        self.user_id = user_id
        self.ai_model_used = model_name
        
        # Create table only if database is new or update existing table
        if not db_exists:
            self._create_table()
            print(f"Created new database at: {self.db_path}")
        else:
            self._update_table_schema()
            print(f"Connected to existing database at: {self.db_path}")
        
        # Initialize session statistics
        self._reset_session_stats()

    def init(self, model_name="YOLOv8", user_id=None):
        """Initialize or update logger with user information."""
        self.ai_model_used = model_name
        if user_id:
            self.user_id = user_id
        # Reset session statistics
        self._reset_session_stats()
    
    def _create_table(self):
        """Create the enhanced detection_logs table."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_start_time TIMESTAMP,
                session_end_time TIMESTAMP,
                total_objects_detected INTEGER NOT NULL,
                right_side_objects INTEGER NOT NULL,
                left_side_objects INTEGER NOT NULL,
                successful_detections INTEGER NOT NULL,
                failed_detections INTEGER NOT NULL,
                changed_side_detections INTEGER NOT NULL,
                left_sticker_errors INTEGER DEFAULT 0,
                right_sticker_errors INTEGER DEFAULT 0,
                total_processing_time REAL DEFAULT 0.0,
                average_processing_time REAL DEFAULT 0.0,
                ai_model_used TEXT NOT NULL,
                factory_code TEXT DEFAULT 'GUNES001',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _update_table_schema(self):
        """Update existing table schema to include new columns."""
        try:
            # Check if new columns exist
            self.cursor.execute("PRAGMA table_info(detection_logs)")
            columns = [column[1] for column in self.cursor.fetchall()]
            
            # Add new columns if they don't exist
            new_columns = [
                ("user_id", "TEXT"),
                ("session_start_time", "TIMESTAMP"),
                ("session_end_time", "TIMESTAMP"),
                ("left_sticker_errors", "INTEGER DEFAULT 0"),
                ("right_sticker_errors", "INTEGER DEFAULT 0"),
                ("total_processing_time", "REAL DEFAULT 0.0"),
                ("average_processing_time", "REAL DEFAULT 0.0"),
                ("factory_code", "TEXT DEFAULT 'GUNES001'")
            ]
            
            for column_name, column_type in new_columns:
                if column_name not in columns:
                    self.cursor.execute(f"ALTER TABLE detection_logs ADD COLUMN {column_name} {column_type}")
                    print(f"Added column: {column_name}")
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error updating table schema: {e}")

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

    def log_sticker_error(self, is_right_side=True):
        """Log a sticker error event."""
        if is_right_side:
            self.session_stats["right_sticker_errors"] += 1
        else:
            self.session_stats["left_sticker_errors"] += 1

    def start_session(self):
        """Mark the start of a new session."""
        self.session_stats["session_start_time"] = datetime.now()
        self._processing_times = []  # Track individual processing times

    def add_processing_time(self, processing_time):
        """Add a processing time measurement."""
        if not hasattr(self, '_processing_times'):
            self._processing_times = []
        self._processing_times.append(processing_time)
        self.session_stats["total_processing_time"] += processing_time

    def send_session_to_backend(self, session_data, access_token=None):
        """Send session data to backend API endpoint."""
        try:
            # Prepare headers
            headers = {
                "factory-code": "GUNES001",
                "Content-Type": "application/json"
            }
            
            # Add authorization header if token is available
            if access_token:
                headers["Authorization"] = f"Bearer {access_token}"
            
            # Prepare the data payload
            payload = {
                "user_id": session_data["user_id"],
                "session_start_time": session_data["session_start_time"].isoformat() if session_data["session_start_time"] else None,
                "session_end_time": session_data["session_end_time"].isoformat() if session_data["session_end_time"] else None,
                "total_objects_detected": session_data["total_objects_detected"],
                "right_side_objects": session_data["right_side_objects"],
                "left_side_objects": session_data["left_side_objects"],
                "successful_detections": session_data["successful_detections"],
                "failed_detections": session_data["failed_detections"],
                "changed_side_detections": session_data["changed_side_detections"],
                "left_sticker_errors": session_data["left_sticker_errors"],
                "right_sticker_errors": session_data["right_sticker_errors"],
                "total_processing_time": session_data["total_processing_time"],
                "average_processing_time": session_data["average_processing_time"],
                "ai_model_used": session_data["ai_model_used"],
                "factory_code": session_data["factory_code"]
            }
            
            # Send POST request to backend
            response = requests.post(
                f"{BACKEND_URL}/api/v1/session-logs",
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200 or response.status_code == 201:
                print("Session data successfully sent to backend")
                return True
            else:
                print(f"Failed to send session data to backend. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("Timeout error when sending session data to backend")
            return False
        except requests.exceptions.ConnectionError:
            print("Connection error when sending session data to backend")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Request error when sending session data to backend: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error when sending session data to backend: {str(e)}")
            return False

    def save_session(self, access_token=None):
        """Save the current session statistics to database."""
        try:
            # Calculate session end time and average processing time
            session_end_time = datetime.now()
            if hasattr(self, '_processing_times') and self._processing_times:
                avg_processing_time = sum(self._processing_times) / len(self._processing_times)
            else:
                avg_processing_time = 0.0

            self.cursor.execute("""
                INSERT INTO detection_logs (
                    user_id,
                    session_start_time,
                    session_end_time,
                    total_objects_detected,
                    right_side_objects,
                    left_side_objects,
                    successful_detections,
                    failed_detections,
                    changed_side_detections,
                    left_sticker_errors,
                    right_sticker_errors,
                    total_processing_time,
                    average_processing_time,
                    ai_model_used,
                    factory_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id,
                self.session_stats.get("session_start_time"),
                session_end_time,
                self.session_stats["total_objects_detected"],
                self.session_stats["right_side_objects"],
                self.session_stats["left_side_objects"],
                self.session_stats["successful_detections"],
                self.session_stats["failed_detections"],
                self.session_stats["changed_side_detections"],
                self.session_stats["left_sticker_errors"],
                self.session_stats["right_sticker_errors"],
                self.session_stats["total_processing_time"],
                avg_processing_time,
                self.session_stats["ai_model_used"],
                "GUNES001"  # Factory code
            ))
            self.conn.commit()
            print("Enhanced session statistics saved successfully")
            
            # Send session data to backend
            session_data = {
                "user_id": self.user_id,
                "session_start_time": self.session_stats.get("session_start_time"),
                "session_end_time": session_end_time,
                "total_objects_detected": self.session_stats["total_objects_detected"],
                "right_side_objects": self.session_stats["right_side_objects"],
                "left_side_objects": self.session_stats["left_side_objects"],
                "successful_detections": self.session_stats["successful_detections"],
                "failed_detections": self.session_stats["failed_detections"],
                "changed_side_detections": self.session_stats["changed_side_detections"],
                "left_sticker_errors": self.session_stats["left_sticker_errors"],
                "right_sticker_errors": self.session_stats["right_sticker_errors"],
                "total_processing_time": self.session_stats["total_processing_time"],
                "average_processing_time": avg_processing_time,
                "ai_model_used": self.session_stats["ai_model_used"],
                "factory_code": "GUNES001"
            }
            
            # Send data to backend
            backend_success = self.send_session_to_backend(session_data, access_token)
            if backend_success:
                print("Session data successfully synchronized with backend")
            else:
                print("Warning: Session data saved locally but failed to sync with backend")
            
            # Reset session statistics
            self._reset_session_stats()
            
            return backend_success
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
            "left_sticker_errors": 0,
            "right_sticker_errors": 0,
            "total_processing_time": 0.0,
            "ai_model_used": self.ai_model_used,
            "session_start_time": None
        }
        # Reset processing times tracker
        self._processing_times = []

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