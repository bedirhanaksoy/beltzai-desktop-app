#!/usr/bin/env python3
"""
Test script for the improved vertical grid tracking system
Tests boundary-crossing logic and occlusion handling
"""

import time

class MockVerticalTracker:
    def __init__(self):
        # Grid system for 3 vertical sections
        self.vertical_sections = {0: {'objects': 0, 'stickers': 0}, 
                                  1: {'objects': 0, 'stickers': 0}, 
                                  2: {'objects': 0, 'stickers': 0}}
        self.frame_width = 640
        self.frame_height = 480
        
        # Tracking system for persistent counting
        self.tracked_objects = {}
        self.tracked_stickers = {}
        self.sticker_id_counter = 0
        self._cleanup_counter = 0
    
    def _get_vertical_section(self, x, y):
        """Determine which vertical section a point (x, y) belongs to (0, 1, or 2)"""
        if self.frame_width is None:
            return None
        
        section_width = self.frame_width // 3
        section = min(int(x // section_width), 2)
        return section
    
    def _track_object_movement(self, track_id, current_section):
        """Track object movement between sections and update counts accordingly"""
        if track_id not in self.tracked_objects:
            # New object entering from outside the frame
            self.tracked_objects[track_id] = {
                'current_section': current_section,
                'previous_section': None,
                'ever_tracked': True,
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
    
    def _cleanup_lost_objects(self, current_track_ids):
        """Remove objects that have been missing for a significant time"""
        current_time = time.time()
        objects_to_remove = []
        
        for track_id in self.tracked_objects.keys():
            if track_id not in current_track_ids:
                # Object is not currently detected
                object_info = self.tracked_objects[track_id]
                time_since_last_seen = current_time - object_info.get('last_seen_frame', current_time)
                
                # Only remove if object has been missing for more than 2 seconds
                if time_since_last_seen > 2.0:
                    objects_to_remove.append(track_id)
            else:
                # Object is detected, update last seen time
                self.tracked_objects[track_id]['last_seen_frame'] = current_time
        
        # Remove objects that have truly left the frame
        for track_id in objects_to_remove:
            object_info = self.tracked_objects[track_id]
            current_section = object_info['current_section']
            if current_section is not None:
                self.vertical_sections[current_section]['objects'] -= 1
                print(f"Object {track_id} permanently left section {current_section}")
            del self.tracked_objects[track_id]
    
    def print_status(self):
        """Print current tracking status"""
        print("\\n=== Vertical Section Status ===")
        for section_id in range(3):
            section_info = self.vertical_sections[section_id]
            objects_count = section_info['objects']
            stickers_count = section_info['stickers']
            match_status = "MATCH" if objects_count == stickers_count else "MISMATCH"
            print(f"Section {section_id + 1}: Objects={objects_count}, Stickers={stickers_count} [{match_status}]")
        print("===============================\\n")

def test_boundary_tracking():
    """Test the boundary-based tracking system"""
    tracker = MockVerticalTracker()
    
    print("Testing Vertical Grid Boundary Tracking System...")
    print(f"Frame size: {tracker.frame_width}x{tracker.frame_height}")
    print(f"Section width: {tracker.frame_width//3}")
    
    # Test 1: Object enters section 0
    print("\\n--- Test 1: Object enters section 0 (x=50) ---")
    tracker._track_object_movement(1, 0)
    tracker.print_status()
    
    # Test 2: Same object moves within section 0 (should not change count)
    print("--- Test 2: Object moves within section 0 (x=100) ---")
    tracker._track_object_movement(1, 0)  # Still in section 0
    tracker.print_status()
    
    # Test 3: Object crosses boundary from section 0 to section 1
    print("--- Test 3: Object crosses to section 1 (x=250) ---")
    tracker._track_object_movement(1, 1)
    tracker.print_status()
    
    # Test 4: Second object enters section 0
    print("--- Test 4: New object enters section 0 ---")
    tracker._track_object_movement(2, 0)
    tracker.print_status()
    
    # Test 5: First object moves to section 2
    print("--- Test 5: First object moves to section 2 ---")
    tracker._track_object_movement(1, 2)
    tracker.print_status()
    
    # Test 6: Simulate temporary occlusion (object disappears briefly)
    print("--- Test 6: Simulate object temporarily hidden (no cleanup yet) ---")
    current_ids = {2}  # Only object 2 visible
    tracker._cleanup_lost_objects(current_ids)
    tracker.print_status()
    
    # Test 7: Object reappears (should not affect count)
    print("--- Test 7: Object 1 reappears in section 2 ---")
    tracker._track_object_movement(1, 2)
    tracker.print_status()
    
    # Test 8: Simulate long absence (cleanup after 2+ seconds)
    print("--- Test 8: Simulate permanent departure (after 2+ seconds) ---")
    # Simulate time passage
    for obj_info in tracker.tracked_objects.values():
        obj_info['last_seen_frame'] = time.time() - 3.0  # 3 seconds ago
    
    current_ids = {2}  # Only object 2 visible
    tracker._cleanup_lost_objects(current_ids)
    tracker.print_status()
    
    print("Boundary tracking test completed!")

if __name__ == "__main__":
    test_boundary_tracking()
