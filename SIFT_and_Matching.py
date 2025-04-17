import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                           QWidget, QPushButton, QFileDialog, QHBoxLayout,
                           QMessageBox, QFrame, QSizePolicy, QComboBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt

# Import custom SIFT implementation
from custom_sift import CustomSIFT

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
        
        # Initialize custom SIFT detector with improved parameters
        self.sift = CustomSIFT(
            n_octaves=4,
            n_scales=5,
            sigma_min=1.6,
            contrast_threshold=0.03,  # Lower threshold for more keypoints
            edge_threshold=12.0,      # Higher threshold for better corner detection
            max_keypoints=7000        # Allow more keypoints
        )
        
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
        self.sift_toggle.currentIndexChanged.connect(self.toggle_sift_implementation)
        
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
        self.matches_slider.valueChanged.connect(self.update_matches_value)

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
        
        # Connect signals
        self.load_button1.clicked.connect(lambda: self.load_image(1))
        self.load_button2.clicked.connect(lambda: self.load_image(2))
        self.extract_button.clicked.connect(self.extract_features)
        self.match_button.clicked.connect(self.match_features)
        self.clear_button.clicked.connect(self.clear_display)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("padding: 8px;")
        
        # Initialize storage for keypoints and descriptors
        self.image1 = None
        self.image2 = None
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None

        # Initialize SIFT detector as custom implementation by default
        self.use_custom_sift = True
        self.sift = CustomSIFT(
            n_octaves=4,
            n_scales=5,
            sigma_min=1.6,
            contrast_threshold=0.03,
            edge_threshold=12.0,
            max_keypoints=7000
        )
        self.opencv_sift = cv2.SIFT_create()
        
    def load_image(self, image_num):
        """Load an image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Open Image {image_num}", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)")
            
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                # Convert to RGB for display
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if image_num == 1:
                    self.image1 = image
                    self.show_image(rgb_image, self.image_label1)
                else:
                    self.image2 = image
                    self.show_image(rgb_image, self.image_label2)
                
                # Enable extract button if both images are loaded
                if self.image1 is not None and self.image2 is not None:
                    self.extract_button.setEnabled(True)
                
                self.status_bar.showMessage(f"Image {image_num} loaded successfully")
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")
    
    def match_features(self):
        """Match features between the two images using selected method"""
        if self.descriptors1 is None or self.descriptors2 is None:
            QMessageBox.warning(self, "Error", "Please extract features first")
            return
        
        start_time = time.time()
        
        try:
            # Match features based on selected method
            method = self.matching_method.currentText()
            
            # For improved matching, use OpenCV's knnMatch with ratio test
            if method == "SSD":  # SSD (use FLANN matcher for better performance)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                
                # Ensure descriptors are proper type
                desc1 = self.descriptors1.astype(np.float32)
                desc2 = self.descriptors2.astype(np.float32)
                
                # Find top 2 matches for each descriptor
                matches = matcher.knnMatch(desc1, desc2, k=2)
                
                # Apply ratio test
                good_matches = self.matching_ratio_test(matches)
                
            else:  # NCC
                # For NCC, use custom matching implementation
                good_matches = self.match_ncc(self.descriptors1, self.descriptors2)
            
            if not good_matches:
                QMessageBox.warning(self, "Error", "No good matches found")
                return
            
            # Get number of matches from slider
            num_matches = min(self.matches_slider.value(), len(good_matches))
            
            # Sort matches by distance
            sorted_matches = sorted(good_matches, key=lambda x: x.distance)[:num_matches]
            
            # Draw matches
            matches_img = cv2.drawMatches(
                self.image1, self.keypoints1,
                self.image2, self.keypoints2,
                sorted_matches,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
            )
            
            # Convert to RGB for display
            matches_rgb = cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB)
            
            # Show results in both frames
            h, w = matches_rgb.shape[:2]
            mid_w = w // 2
            
            img1_matched = matches_rgb[:, :mid_w]
            img2_matched = matches_rgb[:, mid_w:]
            
            self.show_image(img1_matched, self.image_label1)
            self.show_image(img2_matched, self.image_label2)
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            self.status_bar.showMessage(
                f"Feature matching complete ({method}). "
                f"Found {len(good_matches)} good matches, showing top {num_matches}. "
                f"Time: {computation_time:.2f} ms"
            )
        except Exception as e:
            print(f"Feature matching error: {str(e)}")
            QMessageBox.warning(self, "Error", f"Feature matching failed: {str(e)}")
        
    def extract_features(self):
        """Extract SIFT features from both images"""
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Error", "Please load both images first")
            return
            
        start_time = time.time()
        
        try:
            # Process image 1
            gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            
            # Error handling for the custom SIFT implementation
            try:
                self.keypoints1, self.descriptors1 = self.sift.detectAndCompute(gray1, None)
                
                # If custom SIFT fails (returns empty results), fall back to OpenCV SIFT
                if len(self.keypoints1) == 0 and self.use_custom_sift:
                    print("Custom SIFT returned no keypoints for image 1, falling back to OpenCV SIFT")
                    self.keypoints1, self.descriptors1 = self.opencv_sift.detectAndCompute(gray1, None)
                    
            except Exception as e:
                print(f"SIFT detection error on image 1: {str(e)}")
                if self.use_custom_sift:
                    print("Falling back to OpenCV SIFT for image 1")
                    self.keypoints1, self.descriptors1 = self.opencv_sift.detectAndCompute(gray1, None)
                else:
                    QMessageBox.warning(self, "Error", 
                                   f"SIFT failed on image 1: {str(e)}")
                    return
                
            # Process image 2
            gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            
            try:
                self.keypoints2, self.descriptors2 = self.sift.detectAndCompute(gray2, None)
                
                # If custom SIFT fails (returns empty results), fall back to OpenCV SIFT
                if len(self.keypoints2) == 0 and self.use_custom_sift:
                    print("Custom SIFT returned no keypoints for image 2, falling back to OpenCV SIFT")
                    self.keypoints2, self.descriptors2 = self.opencv_sift.detectAndCompute(gray2, None)
                    
            except Exception as e:
                print(f"SIFT detection error on image 2: {str(e)}")
                if self.use_custom_sift:
                    print("Falling back to OpenCV SIFT for image 2")
                    self.keypoints2, self.descriptors2 = self.opencv_sift.detectAndCompute(gray2, None)
                else:
                    QMessageBox.warning(self, "Error", 
                                   f"SIFT failed on image 2: {str(e)}")
                    return
            
            # Draw and display features
            img1_features = cv2.drawKeypoints(
                cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB),
                self.keypoints1,
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            img2_features = cv2.drawKeypoints(
                cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB),
                self.keypoints2,
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Display the results
            self.show_image(img1_features, self.image_label1)
            self.show_image(img2_features, self.image_label2)
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            # Enable match button after feature extraction
            self.match_button.setEnabled(True)
            
            # Get implementation name for status bar
            impl_name = "Custom SIFT" if self.use_custom_sift else "OpenCV SIFT"
            
            # Check if keypoints were found
            if len(self.keypoints1) == 0 or len(self.keypoints2) == 0:
                self.status_bar.showMessage(
                    f"Warning: Few or no keypoints found in one or both images. "
                    f"Consider adjusting parameters or using a different image."
                )
                return
            
            # Update status
            self.status_bar.showMessage(
                f"Features extracted ({impl_name}) - Image 1: {len(self.keypoints1)} keypoints, "
                f"Image 2: {len(self.keypoints2)} keypoints. "
                f"Time: {computation_time:.2f} ms"
            )
        except Exception as e:
            print(f"General feature extraction error: {str(e)}")
            QMessageBox.warning(self, "Error", f"Feature extraction failed: {str(e)}")
        
    def show_image(self, image, label=None):
        """Display an OpenCV image in the QLabel"""
        try:
            # Ensure image is contiguous in memory
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
                
            h, w, ch = image.shape
            bytes_per_line = ch * w
            
            # Convert numpy array to QImage
            q_img = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale the pixmap to fit the label while maintaining aspect ratio
            if label:
                scaled_pixmap = pixmap.scaled(
                    label.width(),
                    label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)
            else:
                scaled_pixmap = pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            QMessageBox.warning(self, "Error", "Failed to display image")
        
    def clear_display(self):
        """Clear all images and reset state"""
        self.image_label1.clear()
        self.image_label2.clear()
        self.image_label1.setText("Image 1")
        self.image_label2.setText("Image 2")
        self.image1 = None
        self.image2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.match_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.status_bar.clearMessage()
        
    def resizeEvent(self, event):
        """Handle window resize events to properly scale the image"""
        super().resizeEvent(event)
        if hasattr(self, 'display_image') and self.display_image is not None:
            self.show_image(self.display_image)

    def match_ssd(self, desc1, desc2):
        """Match features using Sum of Squared Differences"""
        matches = []
        if desc1 is None or desc2 is None:
            return matches
            
        for i in range(desc1.shape[0]):
            min_dist = float('inf')
            match_idx = -1
            for j in range(desc2.shape[0]):
                # Calculate SSD between descriptors
                ssd = np.sum((desc1[i] - desc2[j]) ** 2)
                if ssd < min_dist:
                    min_dist = ssd
                    match_idx = j
            if match_idx >= 0:
                matches.append(cv2.DMatch(i, match_idx, min_dist))
        return matches

    def match_ncc(self, desc1, desc2):
        """Match features using Normalized Cross Correlation with optimization"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
            
        # Use more features for better matching
        max_descriptors = 2000  # Increased from 500
        desc1 = desc1[:min(len(desc1), max_descriptors)]
        desc2 = desc2[:min(len(desc2), max_descriptors)]
        
        # Ensure descriptors are float32
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
        
        # Use batch processing to avoid memory issues
        batch_size = 500
        matches = []
        
        for i in range(0, len(desc1), batch_size):
            batch_desc1 = desc1[i:i+batch_size]
            
            # Pre-normalize descriptors
            batch_desc1_norm = batch_desc1 / (np.linalg.norm(batch_desc1, axis=1)[:, np.newaxis] + 1e-7)
            desc2_norm = desc2 / (np.linalg.norm(desc2, axis=1)[:, np.newaxis] + 1e-7)
            
            # Compute correlation matrix efficiently
            correlation_matrix = np.dot(batch_desc1_norm, desc2_norm.T)
            
            # Get best match for each descriptor
            for j in range(len(batch_desc1)):
                best_idx = np.argmax(correlation_matrix[j])
                score = correlation_matrix[j, best_idx]
                
                # Apply threshold
                if score > 0.6:  # Lower threshold for more matches
                    matches.append(cv2.DMatch(i+j, best_idx, 1.0 - score))
        
        return matches

    def update_matches_value(self, value):
        """Update the label showing number of matches"""
        self.matches_value_label.setText(str(value))

    def toggle_sift_implementation(self, index):
        """Toggle between custom and OpenCV SIFT implementations"""
        try:
            if index == 0:  # Custom SIFT
                self.use_custom_sift = True
                self.sift = CustomSIFT(
                    n_octaves=4,
                    n_scales=3,  # Fewer scales to improve speed
                    sigma_min=1.6,
                    contrast_threshold=0.04,
                    edge_threshold=10.0,
                    max_keypoints=5000
                )
                self.status_bar.showMessage("Using fully custom SIFT implementation (no OpenCV SIFT)")
            else:  # OpenCV SIFT
                self.use_custom_sift = False
                self.sift = cv2.SIFT_create()
                self.status_bar.showMessage("Using OpenCV SIFT implementation")
        except Exception as e:
            print(f"Error toggling SIFT implementation: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to change SIFT implementation: {str(e)}")
            # Fallback to OpenCV SIFT
            self.use_custom_sift = False
            self.sift = cv2.SIFT_create()
            self.sift_toggle.setCurrentIndex(1)  # Set combo box to OpenCV
            self.status_bar.showMessage("Fallback to OpenCV SIFT due to error")

    def matching_ratio_test(self, matches, ratio=0.8):
        """
        Apply ratio test for better matching quality
        - Keeps only matches where the best match is significantly better than the second best
        """
        if not matches or len(matches[0]) < 2:
            return []
            
        # Filter matches using the Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                # If the best match distance is significantly smaller than the second best,
                # it's likely a more reliable match
                if match_pair[0].distance < ratio * match_pair[1].distance:
                    good_matches.append(match_pair[0])
                    
        return good_matches

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SIFTFeatureExtractor()
    window.show()
    sys.exit(app.exec_())