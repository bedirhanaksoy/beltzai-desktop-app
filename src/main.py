from comparer_module import Comparer
import cv2
import numpy as np
from collections import deque
import time
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk
from desktop_module import SequenceApp 

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

right_base_image_path = str(resources_path / "base_images/right_base_image.png")
left_base_image_path = str(resources_path / "base_images/left_base_image.png")


if __name__ == "__main__":
    app = SequenceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    #detect_and_compare_run()
