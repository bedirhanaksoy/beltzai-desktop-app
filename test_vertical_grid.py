#!/usr/bin/env python3
"""
Test script for the 3-section vertical grid tracking system
"""

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
            # New object entering
            self.tracked_objects[track_id] = {
                'current_section': current_section,
                'previous_section': None
            }
            if current_section is not None:
                self.vertical_sections[current_section]['objects'] += 1
                print(f"Object {track_id} entered section {current_section}")
        else:
            # Existing object
            previous_section = self.tracked_objects[track_id]['current_section']
            
            if previous_section != current_section:
                # Object moved to a different section
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
            # New sticker
            sticker_id = self.sticker_id_counter
            self.sticker_id_counter += 1
            self.tracked_stickers[sticker_id] = {
                'current_section': current_section,
                'previous_section': None,
                'last_position': sticker_center
            }
            if current_section is not None:
                self.vertical_sections[current_section]['stickers'] += 1
                print(f"Sticker {sticker_id} entered section {current_section}")
        else:
            # Existing sticker
            previous_section = self.tracked_stickers[sticker_id]['current_section']
            
            if previous_section != current_section:
                # Sticker moved to a different section
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
    
    def print_sections_status(self):
        """Print current sections status"""
        print("\n=== Vertical Sections Status ===")
        for section_id in range(3):
            section_info = self.vertical_sections[section_id]
            objects_count = section_info['objects']
            stickers_count = section_info['stickers']
            match_status = "MATCH ✓" if objects_count == stickers_count else "MISMATCH ✗"
            background_color = "GREEN" if objects_count == stickers_count else "YELLOW"
            print(f"Section {section_id}: Objects={objects_count}, Stickers={stickers_count} | {match_status} | Background: {background_color}")
        print("===============================\n")

def test_vertical_tracking_system():
    """Test the vertical tracking system with simulated movements"""
    tracker = MockVerticalTracker()
    
    print("Testing 3-Section Vertical Grid Tracking System...")
    print(f"Frame size: {tracker.frame_width}x{tracker.frame_height}")
    print(f"Section width: {tracker.frame_width//3} pixels each")
    
    # Test 1: Object enters section 0 (left section)
    print("\n--- Test 1: Object enters section 0 (left) ---")
    tracker._track_object_movement(1, 0)
    tracker.print_sections_status()
    
    # Test 2: Sticker enters same section (should match - green background)
    print("--- Test 2: Sticker enters section 0 (should create match) ---")
    tracker._track_sticker_movement((50, 100), (45, 95, 55, 105))
    tracker.print_sections_status()
    
    # Test 3: Object moves to section 1 (middle section)
    print("--- Test 3: Object moves to section 1 (middle) ---")
    tracker._track_object_movement(1, 1)
    tracker.print_sections_status()
    
    # Test 4: New object enters section 2 (right section)
    print("--- Test 4: New object enters section 2 (right) ---")
    tracker._track_object_movement(2, 2)
    tracker.print_sections_status()
    
    # Test 5: Two stickers enter section 2
    print("--- Test 5: Two stickers enter section 2 ---")
    tracker._track_sticker_movement((500, 100), (495, 95, 505, 105))
    tracker._track_sticker_movement((550, 150), (545, 145, 555, 155))
    tracker.print_sections_status()
    
    # Test 6: Sticker moves to section 1 to match object there
    print("--- Test 6: First sticker from section 0 moves to section 1 ---")
    tracker._track_sticker_movement((300, 100), (295, 95, 305, 105))
    tracker.print_sections_status()
    
    print("Test completed!")
    print("\nExpected behavior:")
    print("- Section 0: Should have yellow background (0 objects, 0 stickers)")
    print("- Section 1: Should have green background (1 object, 1 sticker)")
    print("- Section 2: Should have yellow background (1 object, 2 stickers)")

if __name__ == "__main__":
    test_vertical_tracking_system()
