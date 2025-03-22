from fastapi import FastAPI, HTTPException
from datetime import datetime
from typing import List, Dict
from logger import Logger
from pydantic import BaseModel

class DateRange(BaseModel):
    date_from: datetime
    date_to: datetime

# FastAPI app instance
app = FastAPI()

@app.get("/report/")
async def get_report(date_range: DateRange) -> List[Dict]:
    """
    Get the report between two dates.
    Returns aggregated statistics grouped by AI model used.
    """
    try:
        logger_instance = Logger()
        logs = logger_instance.get_session_stats(
            start_date=date_range.date_from, 
            end_date=date_range.date_to)
        
        # Create a dictionary to store aggregated stats by model
        model_stats = {}

        if not logs:
            return []
        
        for log in logs:
            model_name = log[7]  # ai_model_used
            
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "total_objects_detected": 0,
                    "right_side_objects": 0,
                    "left_side_objects": 0,
                    "successful_detections": 0,
                    "failed_detections": 0,
                    "changed_side_detections": 0,
                }
            
            # Aggregate the stats
            model_stats[model_name]["total_objects_detected"] += log[1]
            model_stats[model_name]["right_side_objects"] += log[2]
            model_stats[model_name]["left_side_objects"] += log[3]
            model_stats[model_name]["successful_detections"] += log[4]
            model_stats[model_name]["failed_detections"] += log[5]
            model_stats[model_name]["changed_side_detections"] += log[6]
        
        # Calculate error rates for each model
        formatted_stats = []
        for model_name, stats in model_stats.items():
            if stats["total_objects_detected"] > 0:
                error_rate = round((stats["failed_detections"] / stats["total_objects_detected"]) * 100, 2)
            else:
                error_rate = 0.00
                
            stats["error_rate"] = error_rate
            stats["ai_model_used"] = model_name
            formatted_stats.append(stats)
            
        return formatted_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving report: {str(e)}")
    
@app.get("/report/model/{model_name}")
async def get_model_report(model_name: str, date_range: DateRange) -> Dict:
    """
    Get the report for a specific model between two dates.
    Returns aggregated statistics for the specified model.
    """
    try:
        logger_instance = Logger()
        logs = logger_instance.get_session_stats(
            start_date=date_range.date_from, 
            end_date=date_range.date_to)
        
        model_stats = {
            "total_objects_detected": 0,
            "right_side_objects": 0,
            "left_side_objects": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "changed_side_detections": 0,
        }
        
        for log in logs:
            if log[7] == model_name:  # ai_model_used
                model_stats["total_objects_detected"] += log[1]
                model_stats["right_side_objects"] += log[2]
                model_stats["left_side_objects"] += log[3]
                model_stats["successful_detections"] += log[4]
                model_stats["failed_detections"] += log[5]
                model_stats["changed_side_detections"] += log[6]
                
        if model_stats["total_objects_detected"] > 0:
            error_rate = round((model_stats["failed_detections"] / model_stats["total_objects_detected"]) * 100, 2)
        else:
            return {}
            
        model_stats["error_rate"] = error_rate
        model_stats["ai_model_used"] = model_name
        
        return model_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model report: {str(e)}")