import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
import requests
import cv2
from PIL import Image, ImageTk
from pathlib import Path
from comparer_module import Comparer
from session_operator import SessionOperator
import time
import os
from datetime import datetime

SERVER_URL = "http://example.com/validate_session"

MAIN_PATH = Path(__file__).resolve()
resources_path = MAIN_PATH.resolve().parent.parent / "resources"

right_base_image_path = str(resources_path / "base_images/right_base_image.png")
left_base_image_path = str(resources_path / "base_images/left_base_image.png")

boxes = [
    [(35, 120), (165, 250)],  # Right box
    [(450, 120), (580, 250)]   # Left box
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

class SequenceApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")  # Changed to modern dark theme
        self.title("BeltzAI Vision Control")
        self.geometry("1200x900")  # Increased size for better layout
        self.resizable(True, True)  # Made resizable
        self.cap = None
        self.update_frame_id = None
        self.selected_model_path = None
        self.datetime_update_id = None
        
        # Configure modern styling
        self.configure_modern_styles()
        self._build_entrance_screen()

    def configure_modern_styles(self):
        """Configure modern styling for the application"""
        style = ttk.Style()
        
        # Custom card style
        style.configure(
            "Card.TFrame",
            relief="flat",
            borderwidth=0,
            padding=20
        )
        
        # Custom button styles
        style.configure(
            "Modern.TButton",
            padding=(20, 12),
            font=("Segoe UI", 10, "bold")
        )
        
        # Custom label styles
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 24, "bold"),
            foreground="#ffffff"
        )
        
        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 14),
            foreground="#b0b0b0"
        )
        
        style.configure(
            "Header.TLabel",
            font=("Segoe UI", 18, "bold"),
            foreground="#ffffff"
        )

    def _validate_session(self):
        user_input = self.session_entry.get().strip()
        if not user_input or user_input == "Ör: 123123":
            messagebox.showerror("Hata", "Lütfen geçerli bir kullanıcı numarası girin.")
            return
        self._build_model_selection_screen()

    def _setup_modern_header(self, title="BeltzAI Vision Control", step=""):
        """Create modern header with gradient-like effect"""
        header_frame = ttk.Frame(self, style="Card.TFrame")
        header_frame.pack(fill="x", padx=0, pady=0)
        
        # Create gradient effect with multiple frames
        for i in range(3):
            gradient_frame = ttk.Frame(header_frame, height=2)
            gradient_frame.pack(fill="x")
        
        # Main header content
        content_frame = ttk.Frame(header_frame)
        content_frame.pack(fill="x", padx=30, pady=20)
        
        # Left side - Title and datetime
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        
        title_label = ttk.Label(left_frame, text=title, style="Title.TLabel")
        title_label.pack(anchor="w")
        
        # DateTime display
        datetime_frame = ttk.Frame(left_frame)
        datetime_frame.pack(anchor="w", pady=(5, 0))
        
        self.date_label = ttk.Label(datetime_frame, text="", style="Subtitle.TLabel")
        self.date_label.pack(side="left")
        
        separator_label = ttk.Label(datetime_frame, text=" • ", style="Subtitle.TLabel")
        separator_label.pack(side="left")
        
        self.time_label = ttk.Label(datetime_frame, text="", style="Subtitle.TLabel")
        self.time_label.pack(side="left")
        
        # Right side - Step indicator
        if step:
            step_label = ttk.Label(content_frame, text=step, 
                                 font=("Segoe UI", 16, "bold"),
                                 foreground="#4CAF50")
            step_label.pack(side="right", anchor="e")
        
        # Update datetime
        self._update_datetime()
        
        # Separator line
        separator = ttk.Separator(self, orient="horizontal")
        separator.pack(fill="x", padx=20, pady=(0, 20))
        
        return header_frame

    def _update_datetime(self):
        """Update date and time labels"""
        try:
            if hasattr(self, 'date_label') and self.date_label.winfo_exists():
                current_time = datetime.now()
                date_str = current_time.strftime("%d.%m.%Y")
                time_str = current_time.strftime("%H:%M:%S")
                
                self.date_label.config(text=f"📅 {date_str}")
                self.time_label.config(text=f"⏰ {time_str}")
                
                self.datetime_update_id = self.after(1000, self._update_datetime)
        except (tk.TclError, AttributeError):
            pass

    def _build_entrance_screen(self):
        """Modern entrance screen with hero section"""
        self._prepare_screen_transition()
        
        # Header
        self._setup_modern_header()
        
        # Main content with hero section
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=40, pady=20)
        
        # Hero section
        hero_frame = ttk.Frame(main_container, style="Card.TFrame")
        hero_frame.pack(fill="both", expand=True)
        
        # Center content
        center_frame = ttk.Frame(hero_frame)
        center_frame.pack(expand=True, fill="both")
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_rowconfigure(1, weight=0)
        center_frame.grid_rowconfigure(2, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_columnconfigure(1, weight=0)
        center_frame.grid_columnconfigure(2, weight=1)
        
        # Main content card
        content_card = ttk.Frame(center_frame, style="Card.TFrame", 
                               relief="raised", borderwidth=1)
        content_card.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        
        # Company branding
        brand_frame = ttk.Frame(content_card)
        brand_frame.pack(pady=(40, 30))
        
        company_label = ttk.Label(brand_frame, text="GÜNEŞ PLASTİK", 
                                style="Title.TLabel")
        company_label.pack()
        
        subtitle_label = ttk.Label(brand_frame, 
                                 text="Yapay Zeka Destekli Görsel Kontrol Sistemi",
                                 style="Subtitle.TLabel")
        subtitle_label.pack(pady=(10, 0))
        
        # Login section
        login_frame = ttk.Frame(content_card)
        login_frame.pack(pady=30, padx=60, fill="x")
        
        # Input field with modern styling
        input_label = ttk.Label(login_frame, text="Kullanıcı Numarası", 
                              font=("Segoe UI", 12, "bold"))
        input_label.pack(anchor="w", pady=(0, 8))
        
        # Entry with better styling
        entry_frame = ttk.Frame(login_frame)
        entry_frame.pack(fill="x", pady=(0, 20))
        
        self.session_entry = ttk.Entry(entry_frame, font=("Segoe UI", 12), 
                                     width=30)
        self.session_entry.pack(fill="x", pady=5)
        self.session_entry.insert(0, "Kullanıcı numaranızı girin...")
        
        # Bind events for placeholder behavior
        self.session_entry.bind("<FocusIn>", self._on_entry_focus_in)
        self.session_entry.bind("<FocusOut>", self._on_entry_focus_out)
        self.session_entry.bind("<Return>", lambda e: self._validate_session())
        
        # Modern login button
        login_btn = ttk.Button(
            login_frame,
            text="Sisteme Giriş Yap",
            bootstyle="success",
            style="Modern.TButton",
            command=self._validate_session
        )
        login_btn.pack(fill="x", pady=10)
        
        # Status indicators
        status_frame = ttk.Frame(content_card)
        status_frame.pack(pady=(0, 40))
        
        status_items = [
            ("🔗", "Sistem Bağlantısı", "Aktif"),
            ("📷", "Kamera Durumu", "Hazır"),
            ("🤖", "AI Modeli", "Yüklendi")
        ]
        
        for icon, label, status in status_items:
            item_frame = ttk.Frame(status_frame)
            item_frame.pack(fill="x", padx=20, pady=5)
            
            ttk.Label(item_frame, text=f"{icon} {label}:", 
                    font=("Segoe UI", 10)).pack(side="left")
            ttk.Label(item_frame, text=status, 
                    font=("Segoe UI", 10, "bold"),
                    foreground="#4CAF50").pack(side="right")

    def _on_entry_focus_in(self, event):
        if self.session_entry.get() == "Kullanıcı numaranızı girin...":
            self.session_entry.delete(0, tk.END)

    def _on_entry_focus_out(self, event):
        if not self.session_entry.get():
            self.session_entry.insert(0, "Kullanıcı numaranızı girin...")

    def _build_model_selection_screen(self):
        """Modern model selection screen with card-based layout"""
        self._prepare_screen_transition()
        
        # Header
        self._setup_modern_header("Model Seçimi", "Adım 1/5")
        
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=40, pady=20)
        
        # Content grid
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill="both", expand=True)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Model list
        left_panel = ttk.Frame(content_frame, style="Card.TFrame")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        # Right panel - Preview
        right_panel = ttk.Frame(content_frame, style="Card.TFrame")
        right_panel.grid(row=0, column=1, sticky="nsew")
        
        # Left panel content
        left_header = ttk.Label(left_panel, text="Mevcut Modeller", style="Header.TLabel")
        left_header.pack(anchor="w", pady=(0, 20))
        
        # Model list with modern cards
        models_folder = resources_path / "models"
        model_files = [f for f in os.listdir(models_folder) if f.endswith(".pt")]
        
        if not model_files:
            error_label = ttk.Label(left_panel, text="Model bulunamadı!", 
                                  font=("Segoe UI", 14), bootstyle="danger")
            error_label.pack(pady=20)
            return
        
        self.selected_model = tk.StringVar(value=model_files[0])
        self.model_cards = {}
        
        # Create scrollable frame for models
        canvas = tk.Canvas(left_panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Model cards
        for i, model in enumerate(model_files):
            model_card = self._create_model_card(scrollable_frame, model, i == 0)
            model_card.pack(fill="x", pady=5)
            self.model_cards[model] = model_card
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Right panel content - Preview
        right_header = ttk.Label(right_panel, text="Model Önizleme", style="Header.TLabel")
        right_header.pack(pady=(0, 20))
        
        # Preview container
        preview_container = ttk.Frame(right_panel, relief="sunken", borderwidth=2)
        preview_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.preview_label = ttk.Label(preview_container, text="Yükleniyor...", 
                                     font=("Segoe UI", 16))
        self.preview_label.pack(expand=True)
        
        # Model info
        info_frame = ttk.Frame(right_panel, style="Card.TFrame")
        info_frame.pack(fill="x", padx=20, pady=10)
        
        self.model_info_label = ttk.Label(info_frame, text="", 
                                        font=("Segoe UI", 11),
                                        justify="left")
        self.model_info_label.pack(anchor="w")
        
        # Update preview
        self._update_model_preview()
        
        # Bottom navigation
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill="x", padx=40, pady=20)
        
        ttk.Button(nav_frame, text="← Geri", bootstyle="secondary",
                  style="Modern.TButton",
                  command=self._build_entrance_screen).pack(side="left")
        
        ttk.Button(nav_frame, text="Devam Et →", bootstyle="success",
                  style="Modern.TButton",
                  command=self._confirm_model_selection).pack(side="right")

    def _create_model_card(self, parent, model_name, is_selected=False):
        """Create a modern model selection card"""
        card_style = "success" if is_selected else "secondary"
        
        card = ttk.Frame(parent, style="Card.TFrame", relief="raised", borderwidth=1)
        
        # Model info
        info_frame = ttk.Frame(card)
        info_frame.pack(fill="x", padx=15, pady=10)
        
        # Radio button (hidden, we'll use card click)
        radio_var = tk.BooleanVar(value=is_selected)
        
        model_label = ttk.Label(info_frame, text=model_name.replace(".pt", ""),
                              font=("Segoe UI", 12, "bold"))
        model_label.pack(anchor="w")
        
        # Model details
        details_label = ttk.Label(info_frame, text="Yapay Zeka Modeli",
                                font=("Segoe UI", 9),
                                foreground="#888888")
        details_label.pack(anchor="w")
        
        # Click handler
        def select_model():
            self.selected_model.set(model_name)
            self._update_model_cards()
            self._update_model_preview()
        
        card.bind("<Button-1>", lambda e: select_model())
        model_label.bind("<Button-1>", lambda e: select_model())
        details_label.bind("<Button-1>", lambda e: select_model())
        
        return card

    def _update_model_cards(self):
        """Update visual state of model cards"""
        selected_model = self.selected_model.get()
        for model, card in self.model_cards.items():
            if model == selected_model:
                card.configure(relief="raised", style="Card.TFrame")
            else:
                card.configure(relief="flat", style="Card.TFrame")

    def _update_model_preview(self):
        """Update model preview image and info"""
        model_name = self.selected_model.get().replace(".pt", "")
        image_path = resources_path / "models" / "models_images" / f"{model_name}.jpeg"
        
        if image_path.exists():
            try:
                img = Image.open(image_path)
                # Larger preview image
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                self.preview_label.configure(image=imgtk, text="")
                self.preview_label.imgtk = imgtk
            except Exception as e:
                self.preview_label.configure(image="", text="Önizleme yüklenemedi")
        else:
            self.preview_label.configure(image="", text="Önizleme mevcut değil")
        
        # Update model info
        info_text = f"Model: {model_name}\nDosya: {self.selected_model.get()}\nDurum: Hazır"
        self.model_info_label.configure(text=info_text)

    def _confirm_model_selection(self):
        """Confirm model selection and proceed"""
        self.selected_model_path = resources_path / "models" / self.selected_model.get()
        print(f"Selected model: {self.selected_model_path}")
        self._build_info_before_taking_base_images_screen()

    def _build_info_before_taking_base_images_screen(self):
        """Modern instruction screen with visual guides"""
        self._prepare_screen_transition()
        
        # Header
        self._setup_modern_header("Hazırlık Talimatları", "Adım 2/5")
        
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=40, pady=20)
        
        # Instruction cards
        instructions = [
            {
                "icon": "📋",
                "title": "Parça Yerleştirme",
                "desc": "Sonraki ekranda görüntülenen sağ ve sol kutuların içine kontrol edilecek parçaları dikkatli bir şekilde yerleştirin."
            },
            {
                "icon": "📸",
                "title": "Fotoğraf Çekme",
                "desc": "Parçalar doğru konumlandıktan sonra klavyenizden 'T' tuşuna basarak referans fotoğraflarını çekin."
            },
            {
                "icon": "✅",
                "title": "Onaylama",
                "desc": "Çekilen fotoğrafları kontrol edin ve sorun yoksa işlemi onaylayarak devam edin."
            }
        ]
        
        for i, instruction in enumerate(instructions):
            card = ttk.Frame(main_container, style="Card.TFrame", 
                           relief="raised", borderwidth=1)
            card.pack(fill="x", pady=10)
            
            content_frame = ttk.Frame(card)
            content_frame.pack(fill="x", padx=30, pady=25)
            
            # Icon and step number
            header_frame = ttk.Frame(content_frame)
            header_frame.pack(fill="x", pady=(0, 15))
            
            step_label = ttk.Label(header_frame, text=f"Adım {i+1}", 
                                 font=("Segoe UI", 10, "bold"),
                                 foreground="#4CAF50")
            step_label.pack(side="left")
            
            icon_label = ttk.Label(header_frame, text=instruction["icon"], 
                                 font=("Segoe UI", 20))
            icon_label.pack(side="right")
            
            # Title
            title_label = ttk.Label(content_frame, text=instruction["title"],
                                  font=("Segoe UI", 14, "bold"))
            title_label.pack(anchor="w", pady=(0, 8))
            
            # Description
            desc_label = ttk.Label(content_frame, text=instruction["desc"],
                                 font=("Segoe UI", 11),
                                 wraplength=800)
            desc_label.pack(anchor="w")
        
        # Navigation
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill="x", padx=40, pady=20)
        
        ttk.Button(nav_frame, text="← Geri", bootstyle="secondary",
                  style="Modern.TButton",
                  command=self._build_model_selection_screen).pack(side="left")
        
        ttk.Button(nav_frame, text="Hazırım →", bootstyle="success",
                  style="Modern.TButton",
                  command=self._build_taking_base_images_screen).pack(side="right")

    def _build_taking_base_images_screen(self):
        """Modern camera interface with live preview"""
        self._prepare_screen_transition()
        self.geometry("1200x1000")
        
        # Header
        self._setup_modern_header("Referans Fotoğraf Çekimi", "Adım 3/5")
        
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Instructions panel
        instructions_frame = ttk.Frame(main_container, style="Card.TFrame",
                                     relief="raised", borderwidth=1)
        instructions_frame.pack(fill="x", pady=(0, 20))
        
        inst_content = ttk.Frame(instructions_frame)
        inst_content.pack(fill="x", padx=25, pady=20)
        
        ttk.Label(inst_content, text="🎯 Çekim Talimatları", 
                font=("Segoe UI", 14, "bold")).pack(anchor="w")
        
        instructions_text = [
            "• Parçaları yeşil çerçeveler içinde ortalayın",
            "• Parçaların tamamen görünür olduğundan emin olun",
            "• 'T' tuşuna basarak fotoğrafı çekin"
        ]
        
        for inst in instructions_text:
            ttk.Label(inst_content, text=inst, 
                    font=("Segoe UI", 11)).pack(anchor="w", pady=2)
        
        # Camera preview with modern styling
        camera_container = ttk.Frame(main_container, style="Card.TFrame",
                                   relief="sunken", borderwidth=2)
        camera_container.pack(fill="both", expand=True)
        
        # Camera status
        status_frame = ttk.Frame(camera_container)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(status_frame, text="📷 Kamera Görüntüsü", 
                font=("Segoe UI", 12, "bold")).pack(side="left")
        
        self.camera_status = ttk.Label(status_frame, text="● CANLI", 
                                     font=("Segoe UI", 10, "bold"),
                                     foreground="#4CAF50")
        self.camera_status.pack(side="right")
        
        # Video display
        self.video_label = ttk.Label(camera_container, relief="flat")
        self.video_label.pack(expand=True, fill="both", padx=20, pady=(0, 20))
        
        # Initialize camera
        self.cap = cv2.VideoCapture(3)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)  # Fallback to default camera
        
        # Controls
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill="x", pady=20)
        
        ttk.Button(controls_frame, text="← Geri", bootstyle="secondary",
                  style="Modern.TButton",
                  command=self._build_info_before_taking_base_images_screen).pack(side="left")
        
        capture_btn = ttk.Button(controls_frame, text="📸 Fotoğraf Çek (T)", 
                               bootstyle="warning", style="Modern.TButton",
                               command=self._capture_base_image)
        capture_btn.pack(side="right")
        
        # Keyboard binding
        self.bind('<KeyPress-t>', self._capture_base_image)
        self.bind('<KeyPress-T>', self._capture_base_image)
        self.focus_set()  # Ensure window can receive key events
        
        # Start video feed
        self._update_frame()

    def _update_frame(self):
        """Update camera frame with modern styling"""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            # Draw modern boxes
            for i, box in enumerate(boxes):
                (x1, y1), (x2, y2) = box
                # Draw main box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (76, 175, 80), 3)
                # Draw corner markers
                corner_size = 20
                # Top-left corner
                cv2.line(frame, (x1, y1), (x1 + corner_size, y1), (255, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_size), (255, 255, 255), 2)
                # Top-right corner
                cv2.line(frame, (x2, y1), (x2 - corner_size, y1), (255, 255, 255), 2)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_size), (255, 255, 255), 2)
                # Bottom-left corner
                cv2.line(frame, (x1, y2), (x1 + corner_size, y2), (255, 255, 255), 2)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_size), (255, 255, 255), 2)
                # Bottom-right corner
                cv2.line(frame, (x2, y2), (x2 - corner_size, y2), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_size), (255, 255, 2), 2)
                
                # Modern label
                label = "SAĞ PARÇA" if i == 0 else "SOL PARÇA"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), 
                            (76, 175, 80), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Convert and display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            # Scale image to fit display
            display_width = 800
            aspect_ratio = img.height / img.width
            display_height = int(display_width * aspect_ratio)
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.update_frame_id = self.after(33, self._update_frame)  # ~30 FPS

    def _capture_base_image(self, event=None):
        """Capture base images with modern feedback"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Kamera Hatası", "Kamera bağlantısı bulunamadı.")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Çekim Hatası", "Fotoğraf çekilemedi.")
            return

        # Save cropped images
        if len(boxes) == 2:
            crop_and_save(boxes[0], frame, right_base_image_path)
            crop_and_save(boxes[1], frame, left_base_image_path)

        # Show success feedback - check if widget still exists
        try:
            if hasattr(self, 'camera_status') and self.camera_status.winfo_exists():
                self.camera_status.config(text="✅ ÇEKİLDİ", foreground="#4CAF50")
        except tk.TclError:
            pass
        
        # Proceed to next screen immediately (remove the delayed callback)
        self._build_info_after_taking_base_images_screen(ret, frame)

    def _build_info_after_taking_base_images_screen(self, ret, frame):
        """Modern success screen with captured image preview"""
        self._prepare_screen_transition()
        self.geometry("1200x1000")
        
        # Header
        self._setup_modern_header("Fotoğraf Çekimi Tamamlandı", "Adım 4/5")
        
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=40, pady=20)
        
        # Success message
        success_frame = ttk.Frame(main_container, style="Card.TFrame",
                                relief="raised", borderwidth=1)
        success_frame.pack(fill="x", pady=(0, 20))
        
        success_content = ttk.Frame(success_frame)
        success_content.pack(fill="x", padx=30, pady=25)
        
        # Success icon and message
        header_frame = ttk.Frame(success_content)
        header_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(header_frame, text="✅", font=("Segoe UI", 24)).pack(side="left")
        ttk.Label(header_frame, text="Referans Fotoğrafları Başarıyla Çekildi!", 
                font=("Segoe UI", 16, "bold"), 
                foreground="#4CAF50").pack(side="left", padx=(15, 0))
        
        # Details
        details_text = "Parçaların referans görüntüleri kaydedildi. Aşağıda çekilen fotoğrafı görebilirsiniz."
        ttk.Label(success_content, text=details_text, 
                font=("Segoe UI", 11), wraplength=800).pack(anchor="w")
        
        # Image preview container
        preview_container = ttk.Frame(main_container, style="Card.TFrame",
                                    relief="sunken", borderwidth=2)
        preview_container.pack(fill="both", expand=True, pady=(0, 20))
        
        # Preview header
        preview_header = ttk.Frame(preview_container)
        preview_header.pack(fill="x", padx=20, pady=(15, 10))
        
        ttk.Label(preview_header, text="📷 Çekilen Fotoğraf", 
                font=("Segoe UI", 14, "bold")).pack(side="left")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        ttk.Label(preview_header, text=f"Çekim Zamanı: {timestamp}", 
                font=("Segoe UI", 10), 
                foreground="#888888").pack(side="right")
        
        # Image display
        image_label = ttk.Label(preview_container, relief="flat")
        image_label.pack(expand=True, fill="both", padx=20, pady=(0, 20))
        
        if ret:
            # Process frame with annotations
            display_frame = frame.copy()
            for i, box in enumerate(boxes):
                (x1, y1), (x2, y2) = box
                # Draw boxes with modern styling
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (76, 175, 80), 3)
                
                # Add labels
                label = "SAĞ PARÇA" if i == 0 else "SOL PARÇA"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), 
                            (76, 175, 80), -1)
                cv2.putText(display_frame, label, (x1 + 5, y1 - 8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert and resize for display
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            display_width = 700
            aspect_ratio = img.height / img.width
            display_height = int(display_width * aspect_ratio)
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.config(image=imgtk)
        
        # Action instructions
        action_frame = ttk.Frame(main_container, style="Card.TFrame",
                               relief="raised", borderwidth=1)
        action_frame.pack(fill="x")
        
        action_content = ttk.Frame(action_frame)
        action_content.pack(fill="x", padx=30, pady=20)
        
        ttk.Label(action_content, text="📋 Sonraki Adımlar", 
                font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        actions = [
            "• Fotoğrafları kontrol edin ve parçaların doğru konumda olduğundan emin olun",
            "• Tekrar çekmek için 'Geri' butonunu kullanın",
            "• Devam etmek için 'Sistemi Başlat' butonuna tıklayın"
        ]
        
        for action in actions:
            ttk.Label(action_content, text=action, 
                    font=("Segoe UI", 10)).pack(anchor="w", pady=2)
        
        # Navigation
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill="x", padx=40, pady=20)
        
        ttk.Button(nav_frame, text="← Tekrar Çek", bootstyle="warning",
                  style="Modern.TButton",
                  command=self._build_taking_base_images_screen).pack(side="left")
        
        ttk.Button(nav_frame, text="🚀 Sistemi Başlat", bootstyle="success",
                  style="Modern.TButton",
                  command=self._build_operation_screen).pack(side="right")

    def _build_operation_screen(self):
        """Modern operation screen"""
        self._prepare_screen_transition()
        
        # Header
        self._setup_modern_header("Kalite Kontrol Sistemi", "Adım 5/5 - AKTİF")
        
        # Main operation container
        operation_frame = ttk.Frame(self)
        operation_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Status bar
        status_frame = ttk.Frame(operation_frame, style="Card.TFrame",
                               relief="raised", borderwidth=1)
        status_frame.pack(fill="x", pady=(0, 20))
        
        status_content = ttk.Frame(status_frame)
        status_content.pack(fill="x", padx=25, pady=15)
        
        ttk.Label(status_content, text="🔄 Sistem Durumu: AKTİF", 
                font=("Segoe UI", 14, "bold"),
                foreground="#4CAF50").pack(side="left")
        
        # Stop button
        stop_btn = ttk.Button(status_content, text="⏹ Oturumu Sonlandır", 
                            bootstyle="danger", style="Modern.TButton",
                            command=self._end_session)
        stop_btn.pack(side="right")
        
        # Main operation area
        operation_container = ttk.Frame(operation_frame)
        operation_container.pack(fill="both", expand=True)
        
        # Initialize session operator with the selected model
        self.detection_and_comparison = SessionOperator(
            tkinter_frame=operation_container,
            end_session_callback=self._end_session,
            model_path=self.selected_model_path,
            right_base_image_path=right_base_image_path,
            left_base_image_path=left_base_image_path
        )
        self.detection_and_comparison.run()

    def _end_session(self):
        """End current session and return to entrance"""
        if hasattr(self, 'detection_and_comparison') and self.detection_and_comparison.is_running:
            self.detection_and_comparison._stop_process()
        
        # Show confirmation dialog
        result = messagebox.askyesno("Oturum Sonlandır", 
                                   "Kalite kontrol oturumunu sonlandırmak istediğinizden emin misiniz?")
        if result:
            self._build_entrance_screen()

    def _prepare_screen_transition(self):
        """Prepare for screen transition by cleaning up resources"""
        # Cancel datetime updates
        if hasattr(self, 'datetime_update_id') and self.datetime_update_id is not None:
            try:
                self.after_cancel(self.datetime_update_id)
            except ValueError:
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

    def on_closing(self):
        """Handle application closing"""
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

if __name__ == "__main__":
    app = SequenceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()