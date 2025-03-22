from fastapi import FastAPI, HTTPException
from datetime import datetime
from typing import List, Dict
from logger import Logger

# FastAPI app instance
app = FastAPI()

@app.get("/report/")
async def get_report(date_from: datetime, date_to: datetime) -> Dict:
    """
    Get the report between two dates.
    Returns formatted statistics for the specified period.
    """
    try:
        logger_instance = Logger()
        logs = logger_instance.get_session_stats(start_date=date_from, end_date=date_to)
        
        # Convert tuple data to dictionary format
        formatted_logs = []
        for log in logs:
            stats = {
                "total_objects_detected": log[1],
                "right_side_objects": log[2],
                "left_side_objects": log[3],
                "successful_detections": log[4],
                "failed_detections": log[5],
                "changed_side_detections": log[6],
                "ai_model_used": log[7],
                "created_at": log[8]
            }
            # Calculate error rate
            if stats["total_objects_detected"] > 0:
                stats["error_rate"] = (stats["failed_detections"] / stats["total_objects_detected"]) * 100
            else:
                stats["error_rate"] = 0
                
            formatted_logs.append(stats)
            
        return {
            "status": "success",
            "data": formatted_logs,
            "period": {
                "from": date_from,
                "to": date_to
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving report: {str(e)}")