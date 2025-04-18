import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
                              QPushButton, QHBoxLayout, QFrame, QSizePolicy, QComboBox,
                                QSlider)
from PyQt5.QtCore import Qt




class SIFTFeatureExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIFT Feature Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QStatusBar {
                color: #ffffff;
                background-color: #1e1e1e;
                font-size: 13px;
            }
        """)
        
        # Add minimum size constraint only
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.main_widget.setLayout(self.layout)
        
        # Title label
        self.title_label = QLabel("SIFT Feature Detector")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 24px;
            color: #ffffff;
            font-weight: bold;
            margin-bottom: 20px;
        """)
        self.layout.addWidget(self.title_label)
        
        # Create image frames for both images
        self.images_layout = QHBoxLayout()
        
        # First image frame
        self.image_frame1 = QFrame()
        self.image_frame1.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.image_frame1.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #333333;
                border-radius: 10px;
                min-height: 400px;
            }
        """)
        self.image_frame_layout1 = QVBoxLayout()
        self.image_frame_layout1.setContentsMargins(10, 10, 10, 10)
        self.image_frame1.setLayout(self.image_frame_layout1)
        
        self.image_label1 = QLabel("Image 1")
        self.image_label1.setAlignment(Qt.AlignCenter)
        self.image_label1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label1.setStyleSheet("border: none; color: #ffffff;")
        self.image_frame_layout1.addWidget(self.image_label1)
        
        # Second image frame
        self.image_frame2 = QFrame()
        self.image_frame2.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.image_frame2.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #333333;
                border-radius: 10px;
                min-height: 400px;
            }
        """)
        self.image_frame_layout2 = QVBoxLayout()
        self.image_frame_layout2.setContentsMargins(10, 10, 10, 10)
        self.image_frame2.setLayout(self.image_frame_layout2)
        
        self.image_label2 = QLabel("Image 2")
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label2.setStyleSheet("border: none; color: #ffffff;")
        self.image_frame_layout2.addWidget(self.image_label2)
        
        # Add frames to layout
        self.images_layout.addWidget(self.image_frame1)
        self.images_layout.addWidget(self.image_frame2)
        self.layout.addLayout(self.images_layout)
        
        # Create button and control layout at the bottom
        self.controls_layout = QVBoxLayout()

        # Add toggle for SIFT implementation
        self.sift_toggle_layout = QHBoxLayout()
        self.sift_toggle_label = QLabel("SIFT Implementation:")
        self.sift_toggle_label.setStyleSheet("color: #ffffff;")
        
        self.sift_toggle = QComboBox()
        self.sift_toggle.addItems(["Custom SIFT (High Accuracy)", "OpenCV SIFT"])
        self.sift_toggle.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: white;
                padding: 5px;
                border: 1px solid #333333;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
        """)
        
        self.sift_toggle_layout.addWidget(self.sift_toggle_label)
        self.sift_toggle_layout.addWidget(self.sift_toggle)
        self.sift_toggle_layout.addStretch()
        
        # Add to the beginning of controls_layout
        self.controls_layout.insertLayout(0, self.sift_toggle_layout)

        # First row of buttons
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(15)

        # Create buttons
        self.load_button1 = QPushButton("Load Image 1")
        self.load_button2 = QPushButton("Load Image 2")
        self.extract_button = QPushButton("Extract Features")
        self.clear_button = QPushButton("Clear All")

        # Add buttons to layout
        self.button_layout.addWidget(self.load_button1)
        self.button_layout.addWidget(self.load_button2)
        self.button_layout.addWidget(self.extract_button)
        self.button_layout.addWidget(self.clear_button)

        self.controls_layout.addLayout(self.button_layout)

        # Second row with matching controls
        self.matching_layout = QHBoxLayout()

        # Add matching method combo box
        self.matching_method_label = QLabel("Matching Method:")
        self.matching_method_label.setStyleSheet("color: #ffffff;")
        self.matching_method = QComboBox()
        self.matching_method.addItems(["SSD", "NCC"])
        self.matching_method.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: white;
                padding: 5px;
                border: 1px solid #333333;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
        """)

        # Add number of matches slider
        self.matches_label = QLabel("Number of Matches:")
        self.matches_label.setStyleSheet("color: #ffffff;")
        self.matches_slider = QSlider(Qt.Horizontal)
        self.matches_slider.setMinimum(10)
        self.matches_slider.setMaximum(100)
        self.matches_slider.setValue(50)
        self.matches_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1e1e1e;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d47a1;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)

        self.matches_value_label = QLabel("50")  # Default value
        self.matches_value_label.setStyleSheet("""
            color: #ffffff;
            min-width: 30px;
            text-align: right;
        """)

        # Connect slider value changed signal
        self.matches_slider.valueChanged.connect(lambda value: self.matches_value_label.setText(str(value)))

        # Add the value label to the layout after the slider
        self.matching_layout.addWidget(self.matches_label)
        self.matching_layout.addWidget(self.matches_slider)
        self.matching_layout.addWidget(self.matches_value_label)

        self.match_button = QPushButton("Match Features")
        self.match_button.setEnabled(False)

        self.matching_layout.addWidget(self.matching_method_label)
        self.matching_layout.addWidget(self.matching_method)
        self.matching_layout.addWidget(self.match_button)
        self.matching_layout.addStretch()

        self.controls_layout.addLayout(self.matching_layout)

        # Add controls to main layout
        self.layout.addLayout(self.controls_layout)


        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("padding: 8px;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SIFTFeatureExtractor()
    window.show()
    sys.exit(app.exec_())
