import sys
import os
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QSlider,
    QProgressBar, QScrollArea, QFrame, QCheckBox, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

BACKEND_URL = "http://127.0.0.1:8000"

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
            data = {'algorithm': self.algorithm, 'sensitivity': str(self.sensitivity)}
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
        self.initUI()

    def initUI(self):
        self.setStyleSheet("""
            QWidget { background-color: #f0f4f9; font-family: Arial; font-size: 15px; }
            QPushButton { background-color: #0a5cb5; color: white; border-radius: 6px; padding: 8px 18px; font-weight: bold; }
            QPushButton:hover { background-color: #084d99; }
            QLabel { color: #003366; font-weight: bold; }
            QFrame#Card { background-color: #ffffff; border-radius: 14px; padding: 18px; border: 1px solid #c3cbd6; }
            QScrollArea { background-color: transparent; }
        """)
        layout = QVBoxLayout(self)
        # Title
        title = QLabel("VisionIQ: Advanced Image Comparison")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        # Upload and controls
        control_layout = QHBoxLayout()
        # Master image upload
        self.upload_btn = QPushButton("Upload Master Image")
        self.upload_btn.clicked.connect(self.upload_master)
        control_layout.addWidget(self.upload_btn)
        self.master_thumb = QLabel()
        self.master_thumb.setFixedSize(60, 60)
        self.master_thumb.setStyleSheet("border:1px solid #bbb; border-radius:8px; background:#fff;")
        control_layout.addWidget(self.master_thumb)
        # Comparison images upload
        self.comp_btn = QPushButton("Upload Comparison Images")
        self.comp_btn.clicked.connect(self.upload_comparisons)
        control_layout.addWidget(self.comp_btn)
        self.comp_thumbs = QHBoxLayout()
        comp_thumb_widget = QWidget()
        comp_thumb_widget.setLayout(self.comp_thumbs)
        comp_thumb_scroll = QScrollArea()
        comp_thumb_scroll.setWidgetResizable(True)
        comp_thumb_scroll.setFixedHeight(70)
        comp_thumb_scroll.setWidget(comp_thumb_widget)
        control_layout.addWidget(comp_thumb_scroll)
        # Algorithm
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["SSIM", "MSE", "PSNR", "Histogram", "Feature Matching"])
        control_layout.addWidget(QLabel("Algorithm"))
        control_layout.addWidget(self.algo_combo)
        # Sensitivity
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setMinimum(0)
        self.sens_slider.setMaximum(100)
        self.sens_slider.setValue(100)
        control_layout.addWidget(QLabel("Sensitivity"))
        control_layout.addWidget(self.sens_slider)
        self.sens_val_label = QLabel("1.00")
        control_layout.addWidget(self.sens_val_label)
        self.sens_slider.valueChanged.connect(lambda v: self.sens_val_label.setText(f"{v/100:.2f}"))
        # Analyze
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.analyze_btn)
        layout.addLayout(control_layout)
        # Results area
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_content = QWidget()
        self.results_layout = QVBoxLayout(self.results_content)
        self.results_area.setWidget(self.results_content)
        layout.addWidget(self.results_area)
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        # Export button
        self.export_btn = QPushButton("Download PDF Report")
        self.export_btn.clicked.connect(self.download_pdf)
        self.export_btn.setVisible(False)
        layout.addWidget(self.export_btn)
        # Error label
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color:red;font-weight:bold;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    def upload_master(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Master Image")
        if path:
            self.master_path = path
            pix = QPixmap(path)
            self.master_thumb.setPixmap(pix.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def upload_comparisons(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Comparison Images")
        if paths:
            self.comparison_paths = paths
            # Show thumbnails
            for i in reversed(range(self.comp_thumbs.count())):
                item = self.comp_thumbs.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            for p in paths:
                lbl = QLabel()
                pix = QPixmap(p)
                lbl.setPixmap(pix.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                lbl.setStyleSheet("border-radius:8px;margin-right:6px;background:#fff;")
                self.comp_thumbs.addWidget(lbl)

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
        self.thread = AnalyzeThread(self.master_path, self.comparison_paths, self.algo_combo.currentText(), self.sens_slider.value()/100)
        self.thread.resultReady.connect(self.display_results)
        self.thread.errorOccurred.connect(self.show_error)
        self.thread.start()

    def display_results(self, data):
        self.progress.setVisible(False)
        self.export_btn.setVisible(True)
        # Clear previous results
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        # Show overall similarity and processing time
        overall = QLabel(f"<b>Overall Similarity:</b> <span style='color:#4ade80;font-size:1.5em'>{data.get('overall_similarity','')}</span> <span style='color:#aaa'>(Processing: {data.get('processing_time',0):.2f}s)</span>")
        overall.setAlignment(Qt.AlignCenter)
        overall.setStyleSheet("font-size:20px;margin-bottom:18px;")
        overall.setTextFormat(Qt.RichText)
        self.results_layout.addWidget(overall)
        # Show each result
        results = data.get('results', data.get('comparisons', []))
        if not results:
            msg = QLabel("No results found.")
            msg.setAlignment(Qt.AlignCenter)
            self.results_layout.addWidget(msg)
            return
        for comp in results:
            card = QFrame()
            card.setObjectName("Card")
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout = QVBoxLayout(card)
            layout.setSpacing(10)
            # Filename
            name = QLabel(comp.get('filename',''))
            name.setFont(QFont("Arial", 13, QFont.Bold))
            name.setAlignment(Qt.AlignCenter)
            layout.addWidget(name)
            # Green-bounded image
            img = QLabel()
            img.setAlignment(Qt.AlignCenter)
            try:
                img_url = comp.get('processed_image') or comp.get('processed_image_url')
                if img_url:
                    img_data = requests.get(BACKEND_URL + img_url).content
                    pix = QPixmap()
                    pix.loadFromData(img_data)
                    img.setPixmap(pix.scaledToWidth(280, Qt.SmoothTransformation))
                else:
                    img.setText("Image not found")
            except Exception:
                img.setText("Image not found")
            layout.addWidget(img)
            # Similarity
            sim = QLabel(f"Similarity: {comp.get('similarity', comp.get('similarity_score',''))}%")
            sim.setFont(QFont("Arial", 15, QFont.Bold))
            sim.setStyleSheet("color:#4ade80;")
            sim.setAlignment(Qt.AlignCenter)
            layout.addWidget(sim)
            # Differences
            diff = QLabel(f"Differences: {comp.get('differences', comp.get('num_differences',''))}")
            diff.setFont(QFont("Arial", 14, QFont.Bold))
            diff.setStyleSheet("color:#60a5fa;")
            diff.setAlignment(Qt.AlignCenter)
            layout.addWidget(diff)
            # Visual output
            label = QLabel(comp.get('visual_label',''))
            label.setFont(QFont("Arial", 12, QFont.Bold))
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            vis = QLabel()
            vis.setAlignment(Qt.AlignCenter)
            try:
                vis_url = comp.get('visual_output')
                if vis_url:
                    vis_data = requests.get(BACKEND_URL + vis_url).content
                    vis_pix = QPixmap()
                    vis_pix.loadFromData(vis_data)
                    vis.setPixmap(vis_pix.scaledToWidth(240, Qt.SmoothTransformation))
                else:
                    vis.setText("Visual not found")
            except Exception:
                vis.setText("Visual not found")
            layout.addWidget(vis)
            self.results_layout.addWidget(card)
        self.pdf_url = data.get('pdf_url')

    def show_error(self, msg):
        self.progress.setVisible(False)
        self.error_label.setText(str(msg))
        self.error_label.setVisible(True)
        self.export_btn.setVisible(False)

    def download_pdf(self):
        if self.pdf_url:
            import webbrowser
            webbrowser.open(BACKEND_URL + self.pdf_url)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionIQApp()
    window.show()
    sys.exit(app.exec_())
