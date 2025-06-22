import tkinter as tk
from tkinter import messagebox
import requests
import cv2
from PIL import Image, ImageTk
from pathlib import Path
from comparer_module import Comparer
from session_operator import SessionOperator
import time
import os  # Add this import to handle file operations
from datetime import datetime

SERVER_URL = "http://example.com/validate_session"  # Replace with your actual endpoint

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

right_base_image_path = str(resources_path / "base_images/right_base_image.png")
left_base_image_path = str(resources_path / "base_images/left_base_image.png")

boxes = [
    [(50, 200), (230, 380)],  # Right box
    [(350, 200), (530, 380)]  # Left box
]


def crop_and_save(box, frame, filename):
    """Save cropped region as base image with an additional margin"""
    if box and len(box) == 2:
        x1, y1 = box[0]
        x2, y2 = box[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = max(0, x1 - 65)
        y1 = max(0, y1 - 65)
        x2 = min(frame.shape[1], x2 + 65)
        y2 = min(frame.shape[0], y2 + 65)

        cropped = frame[y1:y2, x1:x2]
        cv2.imwrite(filename, cropped)
        print(f"Saved {filename}")


class SequenceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sequence App")
        self.geometry("1000x800")
        self.resizable(False, False)
        self.cap = None
        self.update_frame_id = None
        self.selected_model_path = None  # Add instance variable for selected model path
        # self._build_login_screen()
        self._build_entrance_screen()
        # self._build_info_before_taking_base_images_screen()
        # self._build_taking_base_images_screen()

    def _validate_session(self):
        self._build_model_selection_screen()
        #self._build_base_image_screen()
        return

    def _setup_datetime_display(self):
        """Create and configure the date/time display at the top of the screen"""
        # === Top Bar (Date & Time) ===
        top_bar = tk.Frame(self, bg="#E9EBFF")
        top_bar.pack(fill="x", pady=(10, 0))

        date_label = tk.Label(top_bar, text="", font=("Arial", 15), bg="#E9EBFF")
        date_label.pack(side="left", pady=(20, 10), padx=(20, 10))

        time_label = tk.Label(top_bar, text="", font=("Arial", 15), bg="#E9EBFF")
        time_label.pack(side="left", pady=(20, 10), padx=(0, 20))

        # === Top Horizontal Line ===
        top_line = tk.Frame(self, bg="#374151", height=3)
        top_line.pack(fill="x", pady=(30, 0))

        # Initial update and start the recurring updates
        self._update_datetime(date_label, time_label)

        return top_bar, date_label, time_label, top_line

    def _update_datetime(self, date_label, time_label):
        """Update date and time labels and schedule the next update"""
        # Check if the labels still exist before updating them
        try:
            if date_label.winfo_exists() and time_label.winfo_exists():
                current_time = datetime.now()
                date_str = current_time.strftime("📅 %d.%m.%Y")
                time_str = current_time.strftime("⏰ %H:%M:%S")
                
                date_label.config(text=date_str)
                time_label.config(text=time_str)
                
                # Schedule next update
                self.datetime_update_id = self.after(1000, lambda: self._update_datetime(date_label, time_label))
        except tk.TclError:
            # The labels have been destroyed, don't reschedule
            pass

    def _build_entrance_screen(self):
        """GÜNEŞ PLASTİK | BeltzAI Vision Control Yazılımı giriş ekranı"""
        
        self._prepare_screen_transition()

        # Set window properties
        self.configure(bg="#E9EBFF")
        # self.geometry("1000x600")
        self.title("BeltzAI Vision Control Yazılımı")

        # Setup datetime display using the modular method
        _, _, _, _ = self._setup_datetime_display()

        # === Main Content Frame ===
        main_frame = tk.Frame(self, bg="#E9EBFF")
        main_frame.pack(expand=True, fill="both", pady=40, padx=40)

        # === Left Card ===
        left_card = tk.Frame(main_frame, bg="white", width=500, height=220, bd=2, relief="groove")
        left_card.pack(side="left", expand=True, padx=40)
        left_card.pack_propagate(False)

        tk.Label(left_card, text="GÜNEŞ PLASTİK", font=("Arial", 20, "bold"), bg="white").pack(anchor="w", padx=20, pady=(20, 10))
        tk.Label(left_card, text="BeltzAI Vision Control\nYazılımı", font=("Arial", 18, "bold"), bg="white", justify="left").pack(anchor="w", padx=20)

        # === Right Card ===
        right_card = tk.Frame(main_frame, bg="white", width=300, height=220, bd=2, relief="groove")
        right_card.pack(side="right", expand=True, padx=40)
        right_card.pack_propagate(False)

        tk.Label(right_card, text="Kullanıcı Numarası", bg="white", font=("Arial", 10)).pack(anchor="w", padx=20, pady=(30, 5))

        self.session_entry = tk.Entry(right_card, font=("Arial", 11))
        self.session_entry.insert(0, "Ör: 123123")
        self.session_entry.pack(padx=20, fill="x")

        giris_btn = tk.Button(
            right_card,
            text="Giriş",
            bg="#32CD32",
            fg="white",
            font=("Arial", 11, "bold"),
            command=self._validate_session
        )
        giris_btn.pack(anchor="e", padx=20, pady=20)

        # === Bottom Horizontal Line ===
        bottom_line = tk.Frame(self, bg="#374151", height=3)
        bottom_line.pack(fill="x", side="bottom", pady=(0, 50))

    def _skip_capturing_base_image(self, event=None):
        self.next_button.config(state=tk.NORMAL)

    def _update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            for i, box in enumerate(boxes):
                (x1, y1), (x2, y2) = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Right Box" if i == 0 else "Left Box"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.update_frame_id = self.after(10, self._update_frame)

    def _capture_base_image(self, event=None):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Capture Error", "Camera not available.")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Capture Error", "Failed to capture image.")
            return

        if len(boxes) == 2:
            crop_and_save(boxes[0], frame, right_base_image_path)
            crop_and_save(boxes[1], frame, left_base_image_path)

        #self.next_button.config(state=tk.NORMAL)
        self._build_info_after_taking_base_images_screen(ret, frame)

    def _build_model_selection_screen(self):
        """Build the redesigned model selection screen."""
        self._prepare_screen_transition()

        self.configure(bg="#E9EBFF")

        # Top bar + line
        _, _, _, _ = self._setup_datetime_display()

        page_indicator = tk.Label(self, text="1/5", font=("Arial", 18, "bold"), bg="#E9EBFF")
        page_indicator.place(relx=1.0, rely=0.0, anchor="ne", x=-20, y=20)

        main_frame = tk.Frame(self, bg="#E9EBFF")
        main_frame.pack(expand=True, fill="both", padx=40, pady=10)

        left_frame = tk.Frame(main_frame, bg="#E9EBFF")
        left_frame.pack(side="left", fill="y", expand=True, padx=20)

        right_frame = tk.Frame(main_frame, bg="#E9EBFF")
        right_frame.pack(side="right", fill="y", expand=True, padx=20)

        separator = tk.Frame(main_frame, bg="#374151", width=2)
        separator.pack(side="left", fill="y", padx=10)

        # Left: Model list
        tk.Label(left_frame, text="Model Seçimi", font=("Arial", 16, "bold"), bg="#E9EBFF").pack(anchor="w", pady=10)

        models_folder = resources_path / "models"
        model_files = [f for f in os.listdir(models_folder) if f.endswith(".pt")]

        if not model_files:
            tk.Label(left_frame, text="No models found.", bg="#E9EBFF", font=("Arial", 14), fg="red").pack()
            return

        self.selected_model = tk.StringVar(value=model_files[0])

        def update_preview(*args):
            model_name = self.selected_model.get().replace(".pt", "")
            image_path = resources_path / "models" / "models_images" / f"{model_name}.jpeg"
            if image_path.exists():
                img = Image.open(image_path)
                img = img.resize((200, 200))
                imgtk = ImageTk.PhotoImage(img)
                preview_label.config(image=imgtk)
                preview_label.imgtk = imgtk
            else:
                preview_label.config(image="", text="(Resim Yok)")

        for model in model_files:
            radio = tk.Radiobutton(
                left_frame,
                text=model,
                variable=self.selected_model,
                value=model,
                bg="#E9EBFF",
                font=("Arial", 12),
                anchor="w",
                command=update_preview
            )
            radio.pack(anchor="w", pady=2)

        # Right: Preview
        tk.Label(right_frame, text="Seçilen Model", font=("Arial", 16, "bold"), bg="#E9EBFF").pack(anchor="center", pady=10)
        preview_label = tk.Label(right_frame, bg="#D3D3D3", width=200, height=200)
        preview_label.pack(pady=10)

        update_preview()

        # Bottom buttons
        bottom_frame = tk.Frame(self, bg="#E9EBFF")
        bottom_frame.pack(fill="x", pady=20, padx=40)

        back_btn = tk.Button(
            bottom_frame,
            text="Geri",
            bg="#F87171",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._build_entrance_screen
        )
        back_btn.pack(side="left")

        next_btn = tk.Button(
            bottom_frame,
            text="Devam",
            bg="#4CD964",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._confirm_model_selection
        )
        next_btn.pack(side="right")

        # Bottom line
        bottom_line = tk.Frame(self, bg="#374151", height=3)
        bottom_line.pack(fill="x", side="bottom", pady=(0, 50))
        
    def _confirm_model_selection(self):
        """Handle model selection confirmation."""
        # Store the selected model path in an instance variable
        self.selected_model_path = resources_path / "models" / self.selected_model.get()
        print(f"Selected model: {self.selected_model_path}")
        # Proceed to the next screen
        #self._build_base_image_screen()
        #self._build_operation_screen()
        self._build_info_before_taking_base_images_screen()

    def _build_operation_screen(self):
        self._prepare_screen_transition()

        operation_frame = tk.Frame(self, width=800, height=600)
        operation_frame.pack(fill="both", expand=True)

        # Use the instance variable for the model path
        self.detection_and_comparison = SessionOperator(
            tkinter_frame=operation_frame,
            end_session_callback=self._end_session,
            model_path=self.selected_model_path,  # Updated to use instance variable
            right_base_image_path=right_base_image_path,
            left_base_image_path=left_base_image_path
        )
        self.detection_and_comparison.run()

    def _end_session(self):
        if hasattr(self, 'detection_and_comparison') and self.detection_and_comparison.is_running:
            self.detection_and_comparison._stop_process()
        self._build_entrance_screen()

    def on_closing(self):
        if hasattr(self, 'datetime_update_id') and self.datetime_update_id is not None:
            try:
                self.after_cancel(self.datetime_update_id)
            except ValueError:
                pass
    
        if self.update_frame_id:
            try:
                self.after_cancel(self.update_frame_id)
            except ValueError:
                pass
    
        if hasattr(self, 'detection_and_comparison') and self.detection_and_comparison.is_running:
            self.detection_and_comparison._stop_process()
    
        if self.cap:
            self.cap.release()
    
        self.destroy()

    def _prepare_screen_transition(self):
        """Prepare for screen transition by cleaning up resources"""
        # Cancel datetime updates
        if hasattr(self, 'datetime_update_id') and self.datetime_update_id is not None:
            try:
                self.after_cancel(self.datetime_update_id)
            except ValueError:
                # Handle case where ID is not valid
                pass
            self.datetime_update_id = None
            
        # Cancel frame updates
        if self.update_frame_id:
            try:
                self.after_cancel(self.update_frame_id)
            except ValueError:
                pass
            self.update_frame_id = None
            
        # Release camera if active
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()

    def _build_info_after_taking_base_images_screen(self, ret, frame):
        """Display success after taking base images."""
        
        taken_frame = frame
        taken_ret = ret

        self._prepare_screen_transition()

        # Set window properties
        self.configure(bg="#E9EBFF")
        self.geometry("1000x1000")
        self.title("BeltzAI Vision Control Yazılımı")

        # Setup datetime display using the modular method
        _, _, _, _ = self._setup_datetime_display()

        # Page indicator (4/5)
        page_indicator = tk.Label(self, text="4/5", font=("Arial", 18, "bold"), bg="#E9EBFF")
        page_indicator.place(relx=1.0, rely=0.0, anchor="ne", x=-20, y=20)

        # Main Content Area
        content_frame = tk.Frame(self, bg="#E9EBFF")
        content_frame.pack(expand=True, fill="both", padx=40, pady=10)
        
        # Separator Line above instructions
        top_line = tk.Frame(content_frame, height=2, bg="#D0D5E3")
        top_line.pack(fill="x", pady=(0, 2))

        # Instruction Box (Green)
        instruction_box = tk.Frame(
            content_frame, bg="#D1FAE5", highlightbackground="#D0D5E3", highlightthickness=1
        )
        instruction_box.pack(fill="x", pady=(0, 20))

        instruction_text_1 = "•  Parçaların fotoğrafları başarıyla çekildi."
        instruction_label_1 = tk.Label(
            instruction_box,
            text=instruction_text_1,
            font=("Arial", 14, "bold"),
            bg="#D1FAE5",
            fg="#065F46",
            justify="left",
        )
        instruction_label_1.pack(anchor="w", padx=20, pady=(20, 5))

        instruction_text_2 = "•  Çekilen görüntü aşağıdadır."
        instruction_label_2 = tk.Label(
            instruction_box,
            text=instruction_text_2,
            font=("Arial", 14, "bold"),
            bg="#D1FAE5",
            fg="#065F46",
            justify="left",
        )
        instruction_label_2.pack(anchor="w", padx=20, pady=(5, 20))

        # Second Instruction Box before buttons
        instruction_box_2 = tk.Frame(
            self, bg="white", highlightbackground="#D0D5E3", highlightthickness=1
        )
        instruction_box_2.pack(fill="x", padx=40, pady=(0, 10))

        # Image Taken Area
        image_area = tk.Label(
            content_frame,
            bg="#E9EBFF",  # A light gray color
            fg="#E9EBFF"   # A darker gray for the text
        )
        image_area.pack(expand=True, fill="both")

        if taken_ret:
            for i, box in enumerate(boxes):
                (x1, y1), (x2, y2) = box
                cv2.rectangle(taken_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Right Box" if i == 0 else "Left Box"
                cv2.putText(taken_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img = cv2.cvtColor(taken_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            image_area.imgtk = imgtk
            image_area.config(image=imgtk)

        instruction_text_3 = "•  Tekrar fotoğraf çekmek için geri butonuna basın. Devam etmek için devam butonuna basın."
        instruction_label_3 = tk.Label(
            instruction_box_2,
            text=instruction_text_3,
            font=("Arial", 13),
            bg="white",
            fg="#1F2937",
            justify="left",
            anchor="w",
        )
        instruction_label_3.pack(anchor="w", padx=20, pady=15)

        # Footer with buttons
        footer_frame = tk.Frame(self, bg="#E9EBFF")
        footer_frame.pack(fill="x", padx=40, pady=(20, 20))

        # 'Geri' Button
        geri_button = tk.Button(
            footer_frame,
            text="Geri",
            bg="#F87171",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._build_taking_base_images_screen
        )
        geri_button.pack(side="left")

        # 'Devam' Button
        devam_button = tk.Button(
            footer_frame,
            text="Devam",
            bg="#28a745",
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._build_operation_screen
        )
        devam_button.pack(side="right")

        # Bottom horizontal line
        bottom_line = tk.Frame(self, bg="#374151", height=3)
        bottom_line.pack(fill="x", side="bottom", pady=(0, 50))


    def _build_info_before_taking_base_images_screen(self):
        """Display instructions before taking base images."""
        self._prepare_screen_transition()
        
        # Set window properties
        self.configure(bg="#E9EBFF")
        self.geometry("1000x800")
        self.title("BeltzAI Vision Control Yazılımı")
        
        # Setup datetime display using the modular method
        _, _, _, _ = self._setup_datetime_display()
        
        # Page indicator (2/5)
        page_indicator = tk.Label(self, text="2/5", font=("Arial", 18, "bold"), bg="#E9EBFF")
        page_indicator.place(relx=0.95, rely=0.1, anchor="center")
        
        # Main content card
        main_card = tk.Frame(self, bg="white", bd=1, relief="solid")
        main_card.pack(fill="both", expand=False, padx=120, pady=(80, 20))          
        
        # Instructions text with bullet points
        bullet1 = tk.Label(
            main_card,
            text="• Sonraki ekranda sağ ve sol kutuların içine parçaları yerleştirin",
            font=("Arial", 16),
            bg="white",
            anchor="w",
            justify="left"
        )
        bullet1.pack(fill="x", padx=40, pady=(60, 20), anchor="w")
        
        bullet2 = tk.Label(
            main_card,
            text="• Sonrasında 'T' tuşuna basarak fotoğraflarını çekin.",
            font=("Arial", 16),
            bg="white",
            anchor="w",
            justify="left"
        )
        bullet2.pack(fill="x", padx=40, pady=(0, 60), anchor="w")
        
        # Button container
        button_frame = tk.Frame(main_card, bg="white")
        button_frame.pack(fill="x", pady=(20, 40))
        
        # Back button
        back_button = tk.Button(
            button_frame,
            text="Geri",
            bg="#FF6B6B",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            relief="raised",
            command=self._build_model_selection_screen
        )
        back_button.pack(side="left", padx=40)
        
        # Continue button
        continue_button = tk.Button(
            button_frame,
            text="Devam",
            bg="#4CD964",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            relief="raised",
            command=self._build_taking_base_images_screen
        )
        continue_button.pack(side="right", padx=40)
        
        # Bottom horizontal line
        bottom_line = tk.Frame(self, bg="#374151", height=3)
        bottom_line.pack(fill="x", side="bottom", pady=(0, 50))

    def _build_taking_base_images_screen(self):
        """
        Builds the UI for the base image capture step (3/5).
        """
        
        # Assumes a _prepare_screen_transition() method exists to clear the window
        self._prepare_screen_transition()

        # Set window properties
        self.configure(bg="#E9EBFF")
        self.geometry("1000x1000")
        self.title("BeltzAI Vision Control Yazılımı")

        # Setup datetime display using the modular method
        _, _, _, _ = self._setup_datetime_display()

        # Page indicator (3/5)
        page_indicator = tk.Label(self, text="3/5", font=("Arial", 18, "bold"), bg="#E9EBFF")
        page_indicator.place(relx=1.0, rely=0.0, anchor="ne", x=-20, y=20)

        # 3. Main Content Area
        # This frame holds the instructions and video input area
        content_frame = tk.Frame(self, bg="#E9EBFF")
        content_frame.pack(expand=True, fill="both", padx=40, pady=10)
        
        # Separator Line above instructions
        top_line = tk.Frame(content_frame, height=2, bg="#D0D5E3")
        top_line.pack(fill="x", pady=(0, 2))

        # Instruction Box
        instruction_box = tk.Frame(
            content_frame, bg="white", highlightbackground="#D0D5E3", highlightthickness=1
        )
        instruction_box.pack(fill="x", pady=(0, 20))

        instruction_text_1 = "•  Sağ ve sol kutuların içine parçaları yerleştirin"
        instruction_label_1 = tk.Label(
            instruction_box,
            text=instruction_text_1,
            font=("Arial", 14, "bold"),
            bg="white",
            justify="left",
        )
        instruction_label_1.pack(anchor="w", padx=20, pady=(20, 5))

        instruction_text_2 = "•  't' tuşuna basarak fotoğraflarını çekin."
        instruction_label_2 = tk.Label(
            instruction_box,
            text=instruction_text_2,
            font=("Arial", 14, "bold"),
            bg="white",
            justify="left",
        )
        instruction_label_2.pack(anchor="w", padx=20, pady=(5, 20))

        # Video Input Area
        self.cap = cv2.VideoCapture(0)
        
        video_area = tk.Label(
            content_frame,
            bg="#E9EBFF",  # A light gray color
            fg="#E9EBFF"   # A darker gray for the text
        )
        video_area.pack(expand=True, fill="both")
        
        # Store reference if you want to update it with a live feed
        self.video_label = video_area 

        footer_frame = tk.Frame(self, bg="#E9EBFF")
        footer_frame.pack(fill="x", padx=40, pady=(20, 20))

        # 'Geri' Button
        # Note: True rounded corners require images or a canvas. 
        # This styles the standard button to match the color scheme.
        geri_button = tk.Button(
            footer_frame,
            text="Geri",
            bg="#F87171",  # A matching red color
            fg="white",
            font=("Arial", 11, "bold"),
            relief="raised",
            bd=2,
            padx=30,
            pady=5,
            command=self._build_info_before_taking_base_images_screen
        )
        geri_button.pack(side="left")

        # Bottom horizontal line
        bottom_line = tk.Frame(self, bg="#374151", height=3)
        bottom_line.pack(fill="x", side="bottom", pady=(0, 50))
        self.bind('<t>', self._capture_base_image)
        self._update_frame()



if __name__ == "__main__":
    app = SequenceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
