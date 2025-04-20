import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matching import matching_ratio_test, match_ncc, match_ssd, harris_detector
from custom_sift import CustomSIFT
from GUI import SIFTFeatureExtractor
from custom_sift import CustomSIFT

class Main(SIFTFeatureExtractor):
    def __init__(self):
        super().__init__()
        
        # Initialize custom SIFT detector with improved parameters
        self.sift = CustomSIFT(
            n_octaves=4,
            n_scales=5,
            sigma_min=1.6,
            contrast_threshold=0.03,  # Lower threshold for more keypoints
            edge_threshold=12.0,      # Higher threshold for better corner detection
            max_keypoints=7000        # Allow more keypoints
        )

        self.sift_toggle.currentIndexChanged.connect(self.toggle_sift_implementation)
        self.sift_toggle.currentIndexChanged.connect(self.toggle_sift_implementation)

        # Connect signals of SIFT
        self.load_button1.clicked.connect(lambda: self.load_image(1, detector_type="SIFT"))
        self.load_button2.clicked.connect(lambda: self.load_image(2, detector_type="SIFT"))
        self.extract_button.clicked.connect(self.extract_features)
        self.match_button.clicked.connect(self.match_features)
        self.clear_button.clicked.connect(lambda: self.clear_display(detector_type="SIFT"))

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


        #Connect signals of Harris
        self.load_harris_button1.clicked.connect(lambda: self.load_image(1))
        self.load_harris_button2.clicked.connect(lambda: self.load_image(2))
        self.load_harris_button3.clicked.connect(lambda: self.load_image(3))
        self.extract_harris_button.clicked.connect(self.extract_harris_features)
        self.clear_harris_button.clicked.connect(lambda: self.clear_display())

        # Initialize storage for Harris images
        self.image_harris1 = None
        self.image_harris2 = None
        self.image_harris3 = None



    def load_image(self, image_num, detector_type="Harris"):
        """Load an image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Open Image {image_num}", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)")
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                # Convert to RGB for display
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if detector_type == "Harris":
                    if image_num == 1:
                        self.image_harris1 = image
                        self.show_image(rgb_image, self.image_harris_label1)
                    elif image_num == 2:
                        self.image_harris2 = image
                        self.show_image(rgb_image, self.image_harris_label2)
                    else:
                        self.image_harris3 = image
                        self.show_image(rgb_image, self.image_harris_label3)

                    self.extract_harris_button.setEnabled(True)

                else:
                    if image_num == 1:
                        self.image1 = image
                        self.show_image(rgb_image, self.image_label1)
                    else:
                        self.image2 = image
                        self.show_image(rgb_image, self.image_label2)
                
                    # Enable extract button if both images are loaded
                    if self.image1 is not None and self.image2 is not None:
                        self.extract_button.setEnabled(True)
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
            
            if method == "SSD":  # SSD (use FLANN matcher for better performance)
                good_matches = match_ssd(self.descriptors1, self.descriptors2)
            
            else:  # NCC
                # For NCC, use custom matching implementation
                good_matches = match_ncc(self.descriptors1, self.descriptors2)
            
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
            if self.image1.ndim == 3:
                gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = self.image1
            
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

    def extract_harris_features(self):
        """Extract Harris features from the loaded images"""
        
    #     start_time = time.time()
        for image, label in zip([self.image_harris1, self.image_harris2, self.image_harris3],[self.image_harris_label1, self.image_harris_label2, self.image_harris_label3]):
            if image is None:
                continue
            try:
                # Harris corner detection
                harris_response = harris_detector(image)
                harris_response_rgb = cv2.cvtColor(harris_response, cv2.COLOR_BGR2RGB)
                self.show_image(harris_response_rgb, label)

            except Exception as e:
                print(f"Harris feature extraction error: {str(e)}")
                QMessageBox.warning(self, "Error", f"Harris feature extraction failed: {str(e)}")
                return
        
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


    def clear_display(self, detector_type="Harris"):
        """Clear all images and reset state"""
        if detector_type == "Harris":
            self.image_harris_label1.clear()
            self.image_harris_label2.clear()
            self.image_harris_label3.clear()
            self.image_harris_label1.setText("Image 1")
            self.image_harris_label2.setText("Image 2")
            self.image_harris_label3.setText("Image 3")
            self.image_harris1 = None
            self.image_harris2 = None
            self.image_harris3 = None
            self.extract_harris_button.setEnabled(False)
        else:
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




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())