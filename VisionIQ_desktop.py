import sys
import os
import requests
import uuid
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QSlider,
    QProgressBar, QScrollArea, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

def get_backend_url():
    return os.environ.get('VISIONIQ_BACKEND_URL', 'http://127.0.0.1:8000')

BACKEND_URL = get_backend_url()

def is_backend_available():
    try:
        url = BACKEND_URL + "/ping"
        r = requests.get(url, timeout=1)
        return r.status_code == 200
    except Exception:
        return False

class AnalyzeThread(QThread):
    resultReady = pyqtSignal(dict)
    errorOccurred = pyqtSignal(str)

    def __init__(self, master_path, comparison_paths, algorithm, sensitivity):
        super().__init__()
        self.master_path = master_path
        self.comparison_paths = comparison_paths
        self.algorithm = algorithm
        self.sensitivity = sensitivity

    def run(self):
        try:
            all_files = [('master_image', open(self.master_path, 'rb'))] + [('comparison_images', open(p, 'rb')) for p in self.comparison_paths]
            data = {'algorithm': self.algorithm, 'sensitivity': str(int(self.sensitivity))}
            response = requests.post(f"{BACKEND_URL}/analyze", files=all_files, data=data)
            if response.status_code != 200:
                self.errorOccurred.emit(f"Error: {response.status_code} {response.text}")
                return
            self.resultReady.emit(response.json())
        except Exception as e:
            self.errorOccurred.emit(str(e))

class VisionIQApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionIQ - Desktop Edition")
        self.resize(1280, 900)
        self.master_path = None
        self.comparison_paths = []
        self.pdf_url = None
        self.backend_ok = is_backend_available()
        self.initUI()

    def initUI(self):
        # --- HAL-inspired white & blue theme ---
        self.setStyleSheet("""
            QWidget { background: #f7fafd; color: #003366; font-family: Arial, Helvetica, sans-serif; font-size: 16px; }
            QPushButton { background-color: #0073e6; color: white; border-radius: 10px; padding: 12px 28px; font-weight: 700; font-size: 16px; }
            QPushButton:hover { background-color: #005bb5; }
            QPushButton:pressed { background-color: #003366; }
            QLabel { color: #003366; font-weight: 700; font-family: Arial, Helvetica, sans-serif; }
            QFrame#Card { background-color: #fff; border-radius: 18px; padding: 22px; border: 2px solid #0073e6; }
            QScrollArea { background-color: transparent; border: none; }
            QComboBox, QSlider { background-color: #eaf1fb; color: #003366; border-radius: 8px; font-size: 16px; }
            QComboBox { padding: 8px 18px; min-width: 140px; font-weight: 600; }
            QComboBox QAbstractItemView { background: #fff; border-radius: 8px; font-size: 16px; }
            QSlider::groove:horizontal { height: 8px; background: #e0e7ef; border-radius: 4px; }
            QSlider::handle:horizontal { background: #0073e6; border-radius: 10px; width: 20px; height: 20px; margin: -6px 0; }
            QProgressBar { background: #eaf1fb; color: #003366; border-radius: 8px; font-size: 16px; }
            QToolTip { background-color: #fff; color: #003366; border: 1px solid #0073e6; font-size: 15px; }
        """)
        main_layout = QVBoxLayout(self)
        # --- Title ---
        title = QLabel("VisionIQ: Advanced Image Comparison")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#003366; margin-bottom: 18px; margin-top: 12px; letter-spacing: 0.5px;")
        main_layout.addWidget(title)
        # --- Image upload area ---
        upload_row = QHBoxLayout()
        upload_row.setSpacing(24)
        # Master image card (left)
        master_card = QFrame()
        master_card.setObjectName("Card")
        master_layout = QVBoxLayout(master_card)
        master_layout.setSpacing(10)
        master_label_row = QHBoxLayout()
        master_label = QLabel("Master Image")
        master_label.setFont(QFont("Arial", 16, QFont.Bold))
        master_label_row.addWidget(master_label)
        self.remove_master_btn = QPushButton("✕")
        self.remove_master_btn.setFixedSize(24, 24)
        self.remove_master_btn.setStyleSheet("background:#fff;color:#e60026;border:none;font-size:14px;border-radius:12px;")
        self.remove_master_btn.setToolTip("Remove master image")
        self.remove_master_btn.clicked.connect(self.clear_master)
        self.remove_master_btn.setCursor(Qt.PointingHandCursor)
        master_label_row.addStretch()
        master_label_row.addWidget(self.remove_master_btn)
        master_layout.addLayout(master_label_row)
        self.master_thumb = QLabel()
        self.master_thumb.setFixedSize(220, 220)
        self.master_thumb.setStyleSheet("border:2px solid #0073e6; border-radius:14px; background:#eaf1fb;")
        self.master_thumb.setAlignment(Qt.AlignCenter)
        master_layout.addWidget(self.master_thumb, alignment=Qt.AlignCenter)
        self.upload_btn = QPushButton("Upload Master Image")
        self.upload_btn.setToolTip("Select the master image for comparison")
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setFont(QFont("Arial", 15, QFont.Bold))
        self.upload_btn.clicked.connect(self.upload_master)
        master_layout.addWidget(self.upload_btn)
        upload_row.addWidget(master_card, 2)
        # Comparison images card (right)
        comp_card = QFrame()
        comp_card.setObjectName("Card")
        comp_layout = QVBoxLayout(comp_card)
        comp_layout.setSpacing(10)
        comp_label_row = QHBoxLayout()
        comp_label = QLabel("Comparison Images")
        comp_label.setFont(QFont("Arial", 16, QFont.Bold))
        comp_label_row.addWidget(comp_label)
        comp_label_row.addStretch()
        comp_layout.addLayout(comp_label_row)
        # Horizontally scrollable area for comparison images
        self.comp_thumbs = QHBoxLayout()
        self.comp_thumbs.setSpacing(10)
        self.comp_thumb_widgets = []
        self.refresh_comp_thumbs()
        comp_thumb_widget = QWidget()
        comp_thumb_widget.setLayout(self.comp_thumbs)
        comp_thumb_scroll = QScrollArea()
        comp_thumb_scroll.setWidgetResizable(True)
        comp_thumb_scroll.setFixedHeight(120)
        comp_thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        comp_thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        comp_thumb_scroll.setWidget(comp_thumb_widget)
        comp_layout.addWidget(comp_thumb_scroll)
        self.comp_btn = QPushButton("Upload Comparison Images")
        self.comp_btn.setToolTip("Add more comparison images")
        self.comp_btn.setCursor(Qt.PointingHandCursor)
        self.comp_btn.setFont(QFont("Arial", 15, QFont.Bold))
        self.comp_btn.clicked.connect(self.upload_comparisons)
        comp_layout.addWidget(self.comp_btn)
        upload_row.addWidget(comp_card, 2)
        main_layout.addLayout(upload_row)
        # --- Analysis controls ---
        controls_card = QFrame()
        controls_card.setObjectName("Card")
        controls_layout = QHBoxLayout(controls_card)
        controls_layout.setSpacing(20)
        # Algorithm
        algo_label = QLabel("Algorithm")
        algo_label.setToolTip("Select the comparison algorithm")
        algo_label.setFont(QFont("Arial", 15, QFont.Bold))
        controls_layout.addWidget(algo_label)
        self.algo_combo = QComboBox()
        self.algo_combo.setToolTip("Select the comparison algorithm")
        self.algo_combo.addItems(["SSIM", "MSE", "PSNR", "Histogram", "Feature Matching", "Deep Learning"])
        self.algo_combo.setFont(QFont("Arial", 15, QFont.Bold))
        self.algo_combo.setCursor(Qt.PointingHandCursor)
        controls_layout.addWidget(self.algo_combo)
        # Sensitivity
        sens_label = QLabel("Sensitivity")
        sens_label.setToolTip("Adjust the sensitivity for difference detection")
        sens_label.setFont(QFont("Arial", 15, QFont.Bold))
        controls_layout.addWidget(sens_label)
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setMinimum(0)
        self.sens_slider.setMaximum(100)
        self.sens_slider.setValue(100)
        self.sens_slider.setToolTip("Adjust the sensitivity for difference detection")
        self.sens_slider.setFixedWidth(220)
        self.sens_slider.setCursor(Qt.PointingHandCursor)
        controls_layout.addWidget(self.sens_slider)
        self.sens_val_label = QLabel("100%")
        self.sens_val_label.setToolTip("Current sensitivity value")
        self.sens_val_label.setFont(QFont("Arial", 15, QFont.Bold))
        controls_layout.addWidget(self.sens_val_label)
        self.sens_slider.valueChanged.connect(lambda v: self.sens_val_label.setText(f"{v}%"))
        # Buttons
        self.analyze_btn = QPushButton("Analysis")
        self.analyze_btn.setToolTip("Start the image comparison analysis")
        self.analyze_btn.setFixedWidth(140)
        self.analyze_btn.setFont(QFont("Arial", 15, QFont.Bold))
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.start_analysis)
        controls_layout.addWidget(self.analyze_btn)
        self.export_btn = QPushButton("Download")
        self.export_btn.setToolTip("Download the results as a PDF report")
        self.export_btn.setFixedWidth(140)
        self.export_btn.setFont(QFont("Arial", 15, QFont.Bold))
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.clicked.connect(self.download_pdf)
        self.export_btn.setVisible(False)
        controls_layout.addWidget(self.export_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Clear all images and results")
        self.clear_btn.setFixedWidth(100)
        self.clear_btn.setFont(QFont("Arial", 15, QFont.Bold))
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_all)
        controls_layout.addWidget(self.clear_btn)
        main_layout.addWidget(controls_card)
        # --- Results area ---
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_content = QWidget()
        self.results_layout = QVBoxLayout(self.results_content)
        self.results_area.setWidget(self.results_content)
        main_layout.addWidget(self.results_area, 4)
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)
        # Error label
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color:#e60026;font-weight:bold;")
        self.error_label.setVisible(False)
        main_layout.addWidget(self.error_label)
        # Disable analyze if backend not available
        if not self.backend_ok:
            self.analyze_btn.setEnabled(False)
            self.show_error("Backend server is not running. Please start the backend.")

    def refresh_comp_thumbs(self):
        # Remove all widgets
        for i in reversed(range(self.comp_thumbs.count())):
            item = self.comp_thumbs.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self.comp_thumb_widgets = []
        # Add thumbnails for each comparison image, side by side, square
        for idx, p in enumerate(self.comparison_paths):
            thumb_widget = QWidget()
            thumb_layout = QVBoxLayout(thumb_widget)
            thumb_layout.setContentsMargins(0,0,0,0)
            lbl = QLabel()
            pix = QPixmap(p)
            lbl.setPixmap(pix.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl.setStyleSheet("border-radius:8px;margin-right:6px;background:#eaf1fb; border:2px solid #0073e6; min-width:70px; min-height:70px; max-width:70px; max-height:70px;")
            lbl.setAlignment(Qt.AlignCenter)
            thumb_layout.addWidget(lbl)
            rm_btn = QPushButton("✕")
            rm_btn.setFixedSize(18, 18)
            rm_btn.setStyleSheet("background:#fff;color:#e60026;border:none;font-size:10px;border-radius:9px;")
            rm_btn.setToolTip("Remove this comparison image")
            rm_btn.setCursor(Qt.PointingHandCursor)
            rm_btn.clicked.connect(lambda _, i=idx: self.remove_comparison(i))
            thumb_layout.addWidget(rm_btn, alignment=Qt.AlignCenter)
            self.comp_thumbs.addWidget(thumb_widget)
            self.comp_thumb_widgets.append(thumb_widget)

    def clear_master(self):
        self.master_path = None
        self.master_thumb.clear()

    def remove_comparison(self, idx):
        if 0 <= idx < len(self.comparison_paths):
            self.comparison_paths.pop(idx)
            self.refresh_comp_thumbs()

    def upload_master(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Master Image", filter="Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if path:
            self.master_path = path
            pix = QPixmap(path)
            self.master_thumb.setPixmap(pix.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def upload_comparisons(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Comparison Images", filter="Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if paths:
            self.comparison_paths.extend(paths)
            self.refresh_comp_thumbs()

    def start_analysis(self):
        if not self.master_path or not self.comparison_paths:
            self.show_error("Please select a master image and at least one comparison image.")
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.export_btn.setVisible(False)
        self.error_label.setVisible(False)
        # Clear results
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.thread = AnalyzeThread(self.master_path, self.comparison_paths, self.algo_combo.currentText(), self.sens_slider.value())
        self.thread.resultReady.connect(self.display_results)
        self.thread.errorOccurred.connect(self.show_error)
        self.thread.start()

    def display_results(self, data):
        self.progress.setVisible(False)
        self.export_btn.setVisible(True)
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            widget = self.results_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        # For each comparison, show filename, similarity, differences, and two images side by side
        for comp in data["comparisons"]:
            card = QFrame()
            card.setObjectName("Card")
            card_layout = QVBoxLayout(card)
            # Info row
            info_row = QHBoxLayout()
            info_row.addWidget(QLabel(f"<b>Filename:</b> {comp['filename']}"))
            info_row.addWidget(QLabel(f"<b>Similarity:</b> {comp['similarity_score']}%"))
            info_row.addWidget(QLabel(f"<b>Differences:</b> {comp['num_differences']}"))
            card_layout.addLayout(info_row)
            # Images row
            images_row = QHBoxLayout()
            # Green-bounded image
            left_col = QVBoxLayout()
            left_col.addWidget(QLabel("Differences (Green Boxes):"))
            green_img = QLabel()
            green_img.setFixedSize(200, 200)
            green_img.setAlignment(Qt.AlignCenter)
            green_img.setStyleSheet("border:2px solid #00cc66; border-radius:8px; background:#f7fafd;")
            green_url = comp.get('processed_image_url')
            if green_url:
                try:
                    img_data = requests.get(get_backend_url() + green_url).content
                    pix = QPixmap()
                    if pix.loadFromData(img_data):
                        green_img.setPixmap(pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        green_img.setText("Image not found")
                except Exception:
                    green_img.setText("Image not found")
            else:
                green_img.setText("Image not found")
            left_col.addWidget(green_img)
            images_row.addLayout(left_col)
            # Visual output image
            right_col = QVBoxLayout()
            right_col.addWidget(QLabel(f"{comp.get('visual_label', 'Visual Output')}:"))
            visual_img = QLabel()
            visual_img.setFixedSize(200, 200)
            visual_img.setAlignment(Qt.AlignCenter)
            visual_img.setStyleSheet("border:2px solid #0073e6; border-radius:8px; background:#f7fafd;")
            visual_url = comp.get('visual_output')
            if visual_url:
                try:
                    img_data = requests.get(get_backend_url() + visual_url).content
                    pix = QPixmap()
                    if pix.loadFromData(img_data):
                        visual_img.setPixmap(pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        visual_img.setText("Image not found")
                except Exception:
                    visual_img.setText("Image not found")
            else:
                visual_img.setText("Image not found")
            right_col.addWidget(visual_img)
            images_row.addLayout(right_col)
            card_layout.addLayout(images_row)
            self.results_layout.addWidget(card)
        self.pdf_url = data.get('pdf_url')

    def show_error(self, msg):
        self.progress.setVisible(False)
        self.error_label.setText(str(msg))
        self.error_label.setVisible(True)
        self.export_btn.setVisible(False)

    def download_pdf(self):
        if self.pdf_url:
            from PyQt5.QtWidgets import QFileDialog
            import requests
            save_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "VisionIQ_Report.pdf", "PDF Files (*.pdf)")
            if save_path:
                pdf_data = requests.get(BACKEND_URL + self.pdf_url).content
                with open(save_path, "wb") as f:
                    f.write(pdf_data)

    def clear_all(self):
        self.master_path = None
        self.comparison_paths = []
        self.master_thumb.clear()
        for i in reversed(range(self.comp_thumbs.count())):
            item = self.comp_thumbs.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self.export_btn.setVisible(False)
        self.error_label.setVisible(False)
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionIQApp()
    window.show()
    sys.exit(app.exec_())