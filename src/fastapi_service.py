from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException
from typing import List
from pathlib import Path
from datetime import datetime
import os
import shutil
from logger import Logger

# FastAPI app instance
app = FastAPI()

@app.get("/report/")
async def get_report(date_from: datetime, date_to: datetime):
    """
    Get the report between two dates.
    """
    