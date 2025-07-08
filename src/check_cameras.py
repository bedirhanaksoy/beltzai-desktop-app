#!/usr/bin/env python3
import cv2

def check_available_cameras():
    """Check which camera indices are available"""
    available_cameras = []
    
    print("Checking available camera indices...")
    for i in range(10):  # Check indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i}: Available and working")
                available_cameras.append(i)
            else:
                print(f"Camera index {i}: Opens but can't read frame")
            cap.release()
        else:
            print(f"Camera index {i}: Not available")
    
    if available_cameras:
        print(f"\nWorking camera indices: {available_cameras}")
        return available_cameras[0]  # Return the first working camera
    else:
        print("\nNo working cameras found!")
        return None

if __name__ == "__main__":
    working_camera = check_available_cameras()
    if working_camera is not None:
        print(f"\nRecommended camera index to use: {working_camera}")
    else:
        print("\nPlease check your camera connections.")
