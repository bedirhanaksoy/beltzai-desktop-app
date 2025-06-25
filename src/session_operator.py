import cv2
import time
from PIL import Image, ImageTk
import tkinter as tk
from comparer_module import Comparer
from sticker_module import detect_stickers  # Returns only left stickers now

class SessionOperator:
    def __init__(self, tkinter_frame, end_session_callback, model_path, right_base_image_path, left_base_image_path):
        self.tkinter_frame = tkinter_frame
        self.tkinter_frame.winfo_toplevel().geometry("1000x800")
        self.end_session_callback = end_session_callback
        self.model_path = model_path
        self.right_base_image_path = right_base_image_path
        self.left_base_image_path = left_base_image_path
        self.comparer = Comparer(camera_id=2, model_path=self.model_path)
        self.comparer.load_base_images()
        self.is_running = True
        
        # Grid system for 3 vertical sections
        self.vertical_sections = {0: {'objects': 0, 'stickers': 0}, 
                                  1: {'objects': 0, 'stickers': 0}, 
                                  2: {'objects': 0, 'stickers': 0}}
        self.frame_width = None
        self.frame_height = None
        
        # Tracking system for persistent counting
        self.tracked_objects = {}  # {track_id: {'current_section': section_id, 'previous_section': section_id}}
        self.tracked_stickers = {}  # {sticker_id: {'current_section': section_id, 'previous_section': section_id}}
        self.sticker_id_counter = 0
        self.cleanup_disabled = False  # Cleanup is always enabled
        
        # Persistent error tracking - requires 10 consecutive frames before showing error
        self.sticker_error_tracking = {}  # {track_id: {'error_type': str, 'consecutive_frames': int}}
        self.required_error_frames = 10  # Number of consecutive frames needed for error
        
        # Pile visualization toggle
        self.show_pile_visualization = True  # Flag to show/hide pile visualization
   
    def run(self):
        # TTK frames don't support bg option, they use theme styling instead
        
        # Top line
        top_line = tk.Frame(self.tkinter_frame, bg="#374151", height=3)
        top_line.pack(fill="x", pady=(0, 10))

        # Video display area
        self.video_label = tk.Label(self.tkinter_frame, bg="#E9EBFF")
        self.video_label.pack(fill="both", expand=True)

        # Overlay for pause message
        self.pause_overlay = tk.Label(
            self.video_label,
            text="Kontrol Durduruldu",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#A8A8A8"
        )
        self.pause_overlay.pack_forget()

        # Status label
        self.status_label = tk.Label(self.tkinter_frame, text="Processing...", font=("Arial", 12), bg="#E9EBFF")
        self.status_label.pack(pady=5)

        # Button frame
        button_frame = tk.Frame(self.tkinter_frame, bg="#E9EBFF")
        button_frame.pack(pady=10)

        self.stop_button = tk.Button(
            button_frame,
            text="Durdur",
            bg="#4856D4",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._stop_updates
        )
        self.stop_button.pack(side="left", padx=10)

        self.continue_button = tk.Button(
            button_frame,
            text="Devam Et",
            bg="#4CD964",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._continue_updates
        )
        self.continue_button.pack_forget()

        self.end_button = tk.Button(
            button_frame,
            text="Oturumu Bitir",
            bg="#FF0000",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._stop_process
        )
        self.end_button.pack(side="right", padx=10)

        # Bottom line
        bottom_line = tk.Frame(self.tkinter_frame, bg="#374151", height=3)
        bottom_line.pack(fill="x", side="bottom", pady=(10, 0))

        # Bind 'h' key for toggling pile visualization
        self.tkinter_frame.bind_all('<KeyPress-h>', self._on_h_key_pressed)
        self.tkinter_frame.focus_set()  # Make sure the frame can receive focus

        self._update_frame()

    def _stop_updates(self):
        """Pause frame updates without ending the session."""
        self.is_running = False
        self.stop_button.pack_forget()
        self.continue_button.pack(side="left", padx=10)

        # Start blinking
        self._blink_pause_overlay()
    
    def _continue_updates(self):
        """Resume frame updates."""
        self.is_running = True
        self.continue_button.pack_forget()
        self.stop_button.pack(side="left", padx=10)

        # Ensure pause text is hidden
        self.pause_overlay.pack_forget()

        self._update_frame()
    
    def _blink_pause_overlay(self):
        if not self.is_running:
            # Toggle visibility
            if self.pause_overlay.winfo_ismapped():
                self.pause_overlay.pack_forget()
            else:
                self.pause_overlay.pack(anchor="n", pady=10)

            # Schedule next blink
            self.tkinter_frame.after(500, self._blink_pause_overlay)

    def _initialize_vertical_sections(self, frame_width, frame_height):
        """Initialize the 3 vertical sections"""
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def _get_vertical_section(self, x, y):
        """Determine which vertical section a point (x, y) belongs to (0, 1, or 2)"""
        if self.frame_width is None:
            return None
        
        section_width = self.frame_width // 3
        section = min(int(x // section_width), 2)  # Ensure section is 0, 1, or 2
        return section
    
    def _track_object_movement(self, track_id, current_section):
        """Track object movement between sections and update counts accordingly"""
        if track_id not in self.tracked_objects:
            # New object entering from outside the frame
            self.tracked_objects[track_id] = {
                'current_section': current_section,
                'previous_section': None,
                'ever_tracked': True,  # Mark that this object has been seen
                'last_seen_frame': time.time()
            }
            if current_section is not None:
                self.vertical_sections[current_section]['objects'] += 1
                print(f"Object {track_id} entered section {current_section}")
        else:
            # Existing object - update last seen time
            self.tracked_objects[track_id]['last_seen_frame'] = time.time()
            previous_section = self.tracked_objects[track_id]['current_section']
            
            if previous_section != current_section:
                # Object moved to a different section (crossing boundary)
                if previous_section is not None:
                    # Decrement count from previous section
                    self.vertical_sections[previous_section]['objects'] -= 1
                    print(f"Object {track_id} left section {previous_section}")
                
                if current_section is not None:
                    # Increment count in new section
                    self.vertical_sections[current_section]['objects'] += 1
                    print(f"Object {track_id} entered section {current_section}")
                
                # Update tracking info
                self.tracked_objects[track_id]['previous_section'] = previous_section
                self.tracked_objects[track_id]['current_section'] = current_section
    
    def _track_sticker_movement(self, sticker_center, sticker_bbox):
        """Track sticker movement between sections and update counts accordingly"""
        current_section = self._get_vertical_section(sticker_center[0], sticker_center[1])
        
        # Find if this sticker matches any existing tracked sticker (by proximity)
        sticker_id = None
        min_distance = float('inf')
        threshold_distance = 50  # pixels
        
        for sid, sinfo in self.tracked_stickers.items():
            if 'last_position' in sinfo:
                last_pos = sinfo['last_position']
                distance = ((sticker_center[0] - last_pos[0])**2 + (sticker_center[1] - last_pos[1])**2)**0.5
                if distance < threshold_distance and distance < min_distance:
                    min_distance = distance
                    sticker_id = sid
        
        if sticker_id is None:
            # New sticker entering
            sticker_id = self.sticker_id_counter
            self.sticker_id_counter += 1
            self.tracked_stickers[sticker_id] = {
                'current_section': current_section,
                'previous_section': None,
                'last_position': sticker_center,
                'last_seen_frame': time.time()
            }
            if current_section is not None:
                self.vertical_sections[current_section]['stickers'] += 1
                print(f"Sticker {sticker_id} entered section {current_section}")
        else:
            # Existing sticker - update last seen time and position
            self.tracked_stickers[sticker_id]['last_seen_frame'] = time.time()
            previous_section = self.tracked_stickers[sticker_id]['current_section']
            
            if previous_section != current_section:
                # Sticker moved to a different section (crossing boundary)
                if previous_section is not None:
                    # Decrement count from previous section
                    self.vertical_sections[previous_section]['stickers'] -= 1
                    print(f"Sticker {sticker_id} left section {previous_section}")
                
                if current_section is not None:
                    # Increment count in new section
                    self.vertical_sections[current_section]['stickers'] += 1
                    print(f"Sticker {sticker_id} entered section {current_section}")
                
                # Update tracking info
                self.tracked_stickers[sticker_id]['previous_section'] = previous_section
                self.tracked_stickers[sticker_id]['current_section'] = current_section
            
            # Update last known position
            self.tracked_stickers[sticker_id]['last_position'] = sticker_center
    
    def _cleanup_lost_objects(self, current_track_ids):
        """Clean up sections only when no objects are detected in that section for multiple consecutive frames"""
        # Count how many objects are currently detected in each section based on YOLO detections
        current_objects_per_section = {0: 0, 1: 0, 2: 0}
        
        # Count objects that YOLO is currently tracking in each section
        for track_id in current_track_ids:
            if track_id in self.tracked_objects:
                current_section = self.tracked_objects[track_id]['current_section']
                if current_section is not None:
                    current_objects_per_section[current_section] += 1
        
        # Also count objects in each section based on current YOLO detections (more reliable)
        actual_objects_per_section = {0: 0, 1: 0, 2: 0}
        if hasattr(self.comparer, 'yolo_detections') and len(self.comparer.yolo_detections) > 0:
            for det in self.comparer.yolo_detections[0].boxes:
                if det.conf.cpu().numpy()[0] >= 0.5:  # Only count confident detections
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    section = self._get_vertical_section(center_x, center_y)
                    if section is not None:
                        actual_objects_per_section[section] += 1
        
        # Initialize section empty counters if not exists
        if not hasattr(self, 'section_empty_counters'):
            self.section_empty_counters = {0: 0, 1: 0, 2: 0}
        
        # For each section, check if it's been empty for multiple consecutive frames
        for section_id in range(3):
            # Section is considered empty if both tracking and actual detection show no objects
            section_is_empty = (current_objects_per_section[section_id] == 0 and 
                              actual_objects_per_section[section_id] == 0)
            
            if section_is_empty:
                # No objects currently detected in this section
                self.section_empty_counters[section_id] += 1
                
                # Only reset if section has been empty for 90+ consecutive frames (about 3 seconds at 30fps)
                # This prevents false resets due to temporary tracking loss or occlusion
                if self.section_empty_counters[section_id] >= 90:
                    if self.vertical_sections[section_id]['objects'] > 0:
                        print(f"Section {section_id} has been empty for {self.section_empty_counters[section_id]} frames, resetting count from {self.vertical_sections[section_id]['objects']} to 0")
                        self.vertical_sections[section_id]['objects'] = 0
                        
                        # Remove tracking data for objects that were in this section
                        objects_to_remove = []
                        for track_id, obj_info in self.tracked_objects.items():
                            if obj_info['current_section'] == section_id:
                                objects_to_remove.append(track_id)
                        
                        for track_id in objects_to_remove:
                            del self.tracked_objects[track_id]
                            print(f"Removed tracking for object {track_id} from empty section {section_id}")
                    
                    # Reset the counter after cleanup
                    self.section_empty_counters[section_id] = 0
            else:
                # Objects detected in this section, reset empty counter
                self.section_empty_counters[section_id] = 0
        
        # Update last seen time for detected objects
        current_time = time.time()
        for track_id in current_track_ids:
            if track_id in self.tracked_objects:
                self.tracked_objects[track_id]['last_seen_frame'] = current_time
    
    def _cleanup_lost_stickers(self, current_sticker_positions):
        """Only reset section sticker counts when no stickers are detected in that section for multiple consecutive frames"""
        # Count how many stickers are currently detected in each section
        current_stickers_per_section = {0: 0, 1: 0, 2: 0}
        
        for pos in current_sticker_positions:
            section = self._get_vertical_section(pos[0], pos[1])
            if section is not None:
                current_stickers_per_section[section] += 1
        
        # Initialize sticker section empty counters if not exists
        if not hasattr(self, 'sticker_section_empty_counters'):
            self.sticker_section_empty_counters = {0: 0, 1: 0, 2: 0}
        
        # For each section, check if it's been empty for multiple consecutive frames
        for section_id in range(3):
            if current_stickers_per_section[section_id] == 0:
                # No stickers currently detected in this section
                self.sticker_section_empty_counters[section_id] += 1
                
                # Only reset if section has been empty for 90+ consecutive frames (about 3 seconds at 30fps)
                if self.sticker_section_empty_counters[section_id] >= 90:
                    if self.vertical_sections[section_id]['stickers'] > 0:
                        print(f"Section {section_id} has been empty of stickers for {self.sticker_section_empty_counters[section_id]} frames, resetting count from {self.vertical_sections[section_id]['stickers']} to 0")
                        self.vertical_sections[section_id]['stickers'] = 0
                        
                        # Remove tracking data for stickers that were in this section
                        stickers_to_remove = []
                        for sticker_id, sticker_info in self.tracked_stickers.items():
                            if sticker_info['current_section'] == section_id:
                                stickers_to_remove.append(sticker_id)
                        
                        for sticker_id in stickers_to_remove:
                            del self.tracked_stickers[sticker_id]
                            print(f"Removed tracking for sticker {sticker_id} from empty section {section_id}")
                    
                    # Reset the counter after cleanup
                    self.sticker_section_empty_counters[section_id] = 0
            else:
                # Stickers detected in this section, reset empty counter
                self.sticker_section_empty_counters[section_id] = 0
        
        # Update last seen time for detected stickers
        current_time = time.time()
        for sticker_id, sticker_info in self.tracked_stickers.items():
            # Check if this sticker is still present in current frame
            for pos in current_sticker_positions:
                last_pos = sticker_info['last_position']
                distance = ((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)**0.5
                if distance < 50:  # threshold
                    # Update last seen time for found stickers
                    sticker_info['last_seen_frame'] = current_time
                    break
    
    def _draw_vertical_grid_overlay(self, frame):
        """Draw the 3 vertical sections with background colors and information"""
        if self.frame_width is None or self.frame_height is None:
            return frame
        
        # Skip drawing if pile visualization is disabled
        if not self.show_pile_visualization:
            return frame
        
        section_width = self.frame_width // 3
        
        # Draw background colors and section information
        for section_id in range(3):
            section_info = self.vertical_sections[section_id]
            objects_count = section_info['objects']
            stickers_count = section_info['stickers']
            
            # Calculate section coordinates
            x1 = section_id * section_width
            x2 = x1 + section_width
            y1 = 0
            y2 = self.frame_height
            
            # Check if counts match
            counts_match = objects_count == stickers_count
            
            # Create colored overlay
            overlay = frame.copy()
            if counts_match:
                # Green background for matching counts
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            else:
                # Yellow background for mismatched counts
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            
            # Apply overlay with transparency
            cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
            
            # Draw section borders
            border_color = (0, 255, 0) if counts_match else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 3)
            
            # Draw vertical dividing lines
            if section_id < 2:  # Don't draw line after last section
                cv2.line(frame, (x2, 0), (x2, self.frame_height), (255, 255, 255), 2)
            
            # Draw text information
            text_x = x1 + 10
            text_y = 30
            
            # Background rectangle for text readability
            text_bg_x1 = x1 + 5
            text_bg_y1 = 5
            text_bg_x2 = x1 + 120
            text_bg_y2 = 100
            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 255, 255), 1)
            
            # Section label
            cv2.putText(frame, f"Section {section_id + 1}", 
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Objects count
            cv2.putText(frame, f"Objects: {objects_count}", 
                       (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Stickers count
            cv2.putText(frame, f"Stickers: {stickers_count}", 
                       (text_x, text_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Match status
            status_text = "MATCH" if counts_match else "MISMATCH"
            status_color = (0, 255, 0) if counts_match else (0, 0, 255)
            cv2.putText(frame, status_text, 
                       (text_x, text_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
        
        return frame
    

    def reset_tracking_system(self):
        """Reset the entire tracking system"""
        # Reset all section counters
        for section_id in range(3):
            self.vertical_sections[section_id]['objects'] = 0
            self.vertical_sections[section_id]['stickers'] = 0
        
        # Clear tracking dictionaries (only objects need tracking now)
        self.tracked_objects.clear()
        # Note: stickers are now counted directly each frame, no tracking needed
        
        # Clear sticker error tracking
        self.sticker_error_tracking.clear()
        
        # Reset section empty counters
        if hasattr(self, 'section_empty_counters'):
            self.section_empty_counters = {0: 0, 1: 0, 2: 0}
        if hasattr(self, 'sticker_section_empty_counters'):
            self.sticker_section_empty_counters = {0: 0, 1: 0, 2: 0}
        if hasattr(self, 'immediate_empty_counters'):
            self.immediate_empty_counters = {0: 0, 1: 0, 2: 0}
        
        # Reset cleanup counter
        self._cleanup_counter = 0
        
        print("Tracking system reset")



    def _on_h_key_pressed(self, event):
        """Handle 'h' key press to toggle pile visualization"""
        self.show_pile_visualization = not self.show_pile_visualization
        
        if self.show_pile_visualization:
            print("Pile visualization enabled")
        else:
            print("Pile visualization disabled")

    def _update_frame(self):
        if not self.is_running:
            return

        ret, self.comparer.frame = self.comparer.cap.read()
        if not ret:
            self._stop_process()
            return

        self.comparer.frame_display = self.comparer.frame.copy()
        
        # Initialize vertical sections if not done yet
        if self.frame_width is None:
            self._initialize_vertical_sections(self.comparer.frame.shape[1], self.comparer.frame.shape[0])
        
        self.comparer.print_boxes()

        self.comparer.yolo_detections = self.comparer.model.track(self.comparer.frame, verbose=False, persist=True)
        current_time = time.time()

        self.comparer.is_right_box_empty, self.comparer.is_left_box_empty = self.comparer.check_if_box_is_empty(
            self.comparer.yolo_detections[0].boxes.data
        )

        right_status = "Empty" if self.comparer.is_right_box_empty else "Occupied"
        left_status = "Empty" if self.comparer.is_left_box_empty else "Occupied"
        self.status_label.config(text=f"Right Box: {right_status} | Left Box: {left_status}")

        # Detect both left and right stickers per frame
        all_left_stickers, all_right_stickers = detect_stickers(self.comparer.frame, conf_threshold=0.7)

        # Track current sticker positions for cleanup
        current_sticker_positions = []
        for box in all_left_stickers:
            sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
            center_x = (sx1 + sx2) / 2
            center_y = (sy1 + sy2) / 2
            current_sticker_positions.append((center_x, center_y))
        
        for box in all_right_stickers:
            sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
            center_x = (sx1 + sx2) / 2
            center_y = (sy1 + sy2) / 2
            current_sticker_positions.append((center_x, center_y))

        # Track current object track IDs for cleanup
        current_track_ids = set()

        for det in self.comparer.yolo_detections[0].boxes:
            track_id = det.id.int().cpu().numpy()[0] if det.id is not None else 0
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = det.conf.cpu().numpy()[0]
            cls = det.cls.cpu().numpy()[0]

            if conf < 0.5:
                continue

            # Add to current track IDs
            current_track_ids.add(track_id)

            # Track object movement in vertical sections
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_section = self._get_vertical_section(center_x, center_y)
            self._track_object_movement(track_id, current_section)

            part_side = self.comparer.index_side_info[track_id]  # 1 = right, 2 = left

            self.comparer.compare(x1, y1, x2, y2, cls, track_id, current_time)
            self.comparer.check(x1, x2, track_id)

            # Check for left stickers inside this part
            for box in all_left_stickers:
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    if part_side == 1:  # wrong: left sticker on right-labeled part
                        color = (0, 0, 255)  # Red
                        self._track_sticker_error(track_id, "left_on_right")
                    else:  # correct
                        color = (0, 255, 0)  # Green
                        # Reset error tracking if sticker is now correct
                        if track_id in self.sticker_error_tracking:
                            del self.sticker_error_tracking[track_id]
                    cv2.rectangle(self.comparer.frame_display, (int(sx1), int(sy1)), (int(sx2), int(sy2)), color, 2)
                    cv2.putText(self.comparer.frame_display, "L", (int(sx1), int(sy1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check for right stickers inside this part
            for box in all_right_stickers:
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                cx, cy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    if part_side == 2:  # wrong: right sticker on left-labeled part
                        color = (0, 0, 255)  # Red
                        self._track_sticker_error(track_id, "right_on_left")
                    else:  # correct
                        color = (0, 255, 0)  # Green
                        # Reset error tracking if sticker is now correct
                        if track_id in self.sticker_error_tracking:
                            del self.sticker_error_tracking[track_id]
                    cv2.rectangle(self.comparer.frame_display, (int(sx1), int(sy1)), (int(sx2), int(sy2)), color, 2)
                    cv2.putText(self.comparer.frame_display, "R", (int(sx1), int(sy1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw part bounding box
            cv2.rectangle(self.comparer.frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(self.comparer.frame_display, str(part_side),
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Clean up objects that are no longer tracked (run more frequently for better responsiveness)
        if hasattr(self, '_cleanup_counter'):
            self._cleanup_counter += 1
        else:
            self._cleanup_counter = 0
            
        # Run cleanup every 30 frames (about once per second at 30fps) for better responsiveness
        if not self.cleanup_disabled and self._cleanup_counter % 30 == 0:
            self._cleanup_lost_objects(current_track_ids)
            self._cleanup_sticker_errors(current_track_ids)
            
        # Also run immediate check for completely empty sections every frame
        self._check_empty_sections_immediate()

        # Count stickers directly in each vertical section - simple and reliable approach
        self._count_stickers_in_sections(all_left_stickers, all_right_stickers)
        
        # Clean up stickers no longer needed since we count directly each frame

        # Show warning messages if triggered in the last 1 second
        if hasattr(self.comparer, 'sticker_warning_timestamp') and time.time() - self.comparer.sticker_warning_timestamp < 1:
            if hasattr(self.comparer, 'sticker_error_type'):
                if self.comparer.sticker_error_type == "left_on_right":
                    cv2.putText(self.comparer.frame_display, "ERROR: Left Sticker on Right Part!",
                                (self.comparer.frame_display.shape[1] // 2 - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                elif self.comparer.sticker_error_type == "right_on_left":
                    cv2.putText(self.comparer.frame_display, "ERROR: Right Sticker on Left Part!",
                                (self.comparer.frame_display.shape[1] // 2 - 250, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # Draw vertical grid overlay with section information and background colors
        self.comparer.frame_display = self._draw_vertical_grid_overlay(self.comparer.frame_display)

        # Show frame on GUI
        img_rgb = cv2.cvtColor(self.comparer.frame_display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        self.tkinter_frame.after(2, self._update_frame)

    def _stop_process(self):
        self.is_running = False
        self.comparer.logger.save_session()
        self.comparer.cap.release()
        if self.end_session_callback:
            self.end_session_callback()

    def _track_sticker_error(self, track_id, error_type):
        """Track sticker errors that need to persist for multiple frames before showing warning"""
        if track_id not in self.sticker_error_tracking:
            # First time seeing this error
            self.sticker_error_tracking[track_id] = {
                'error_type': error_type,
                'consecutive_frames': 1
            }
        else:
            existing_error = self.sticker_error_tracking[track_id]
            if existing_error['error_type'] == error_type:
                # Same error continues
                existing_error['consecutive_frames'] += 1
                
                # If we've reached the threshold, trigger the warning
                if existing_error['consecutive_frames'] >= self.required_error_frames:
                    self.comparer.sticker_warning_timestamp = time.time()
                    self.comparer.sticker_error_type = error_type
                    print(f"Sticker error confirmed after {existing_error['consecutive_frames']} frames: {error_type}")
            else:
                # Different error type, reset counter
                existing_error['error_type'] = error_type
                existing_error['consecutive_frames'] = 1
    
    def _cleanup_sticker_errors(self, current_track_ids):
        """Remove error tracking for objects that are no longer being tracked"""
        errors_to_remove = []
        for track_id in self.sticker_error_tracking.keys():
            if track_id not in current_track_ids:
                errors_to_remove.append(track_id)
        
        for track_id in errors_to_remove:
            del self.sticker_error_tracking[track_id]
    
    def _check_empty_sections_immediate(self):
        """Immediate check for sections with no detections at all - runs every frame"""
        if not hasattr(self.comparer, 'yolo_detections') or len(self.comparer.yolo_detections) == 0:
            return
            
        # Count actual detections in each section from current frame
        detections_per_section = {0: 0, 1: 0, 2: 0}
        
        for det in self.comparer.yolo_detections[0].boxes:
            if det.conf.cpu().numpy()[0] >= 0.5:  # Only count confident detections
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                section = self._get_vertical_section(center_x, center_y)
                if section is not None:
                    detections_per_section[section] += 1
        
        # Initialize immediate empty counters if not exists
        if not hasattr(self, 'immediate_empty_counters'):
            self.immediate_empty_counters = {0: 0, 1: 0, 2: 0}
        
        # Check each section for immediate emptiness
        for section_id in range(3):
            if detections_per_section[section_id] == 0:
                # No detections in this section
                self.immediate_empty_counters[section_id] += 1
                
                # Reset after 60 consecutive frames (2 seconds) of no detections
                if self.immediate_empty_counters[section_id] >= 30:
                    if self.vertical_sections[section_id]['objects'] > 0:
                        print(f"IMMEDIATE: Section {section_id} has no detections for {self.immediate_empty_counters[section_id]} frames, resetting object count to 0")
                        self.vertical_sections[section_id]['objects'] = 0
                        
                        # Remove all tracking data for this section
                        objects_to_remove = []
                        for track_id, obj_info in self.tracked_objects.items():
                            if obj_info['current_section'] == section_id:
                                objects_to_remove.append(track_id)
                        
                        for track_id in objects_to_remove:
                            del self.tracked_objects[track_id]
                    
                    # Reset counter
                    self.immediate_empty_counters[section_id] = 0
            else:
                # Detections found, reset counter
                self.immediate_empty_counters[section_id] = 0

    def _count_stickers_in_sections(self, all_left_stickers, all_right_stickers):
        """Count stickers directly in each vertical section - simple and reliable"""
        # Reset sticker counts to 0 for fresh counting each frame
        for section_id in range(3):
            self.vertical_sections[section_id]['stickers'] = 0
        
        # Count left stickers in each section
        for box in all_left_stickers:
            sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
            center_x = (sx1 + sx2) / 2
            center_y = (sy1 + sy2) / 2
            section = self._get_vertical_section(center_x, center_y)
            if section is not None:
                self.vertical_sections[section]['stickers'] += 1
        
        # Count right stickers in each section
        for box in all_right_stickers:
            sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
            center_x = (sx1 + sx2) / 2
            center_y = (sy1 + sy2) / 2
            section = self._get_vertical_section(center_x, center_y)
            if section is not None:
                self.vertical_sections[section]['stickers'] += 1
