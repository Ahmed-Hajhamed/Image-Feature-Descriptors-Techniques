import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import time

class CustomSIFT:
    """
    A fully custom SIFT implementation without using OpenCV's SIFT function
    """
    
    def __init__(self, n_octaves=4, n_scales=5, sigma_min=1.6, contrast_threshold=0.04, 
                 edge_threshold=10.0, max_keypoints=7000):
        """
        Initialize the Custom SIFT detector
        
        Args:
            n_octaves: Number of octaves in the scale space
            n_scales: Number of scales per octave
            sigma_min: Base sigma for the first scale
            contrast_threshold: Threshold for low contrast keypoints rejection
            edge_threshold: Threshold for edge response (lower values mean stronger filtering)
            max_keypoints: Maximum number of keypoints to return
        """
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma_min = sigma_min
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.max_keypoints = max_keypoints
        
        # Precompute sigma values for each scale
        self.k = 2 ** (1.0 / n_scales)
        self.sigma_diff = np.zeros(n_scales + 3)
        
        # Calculate sigma difference for Gaussian blur
        sigma_prev = self.sigma_min
        for i in range(1, n_scales + 3):
            sigma_total = sigma_prev * self.k
            self.sigma_diff[i] = np.sqrt(sigma_total**2 - sigma_prev**2)
            sigma_prev = sigma_total
    
    def detectAndCompute(self, image, mask=None):
        """
        Detect keypoints and compute descriptors
        
        Args:
            image: Input grayscale image
            mask: Optional mask to restrict feature detection
            
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Numpy array of descriptors (shape: n_keypoints x 128)
        """
        try:
            if mask is not None:
                image = image.copy()
                image[mask == 0] = 0
                
            start_time = time.time()
            
            # Build Gaussian and DoG pyramids
            gaussian_pyramid = self._build_gaussian_pyramid(image)
            dog_pyramid = self._build_dog_pyramid(gaussian_pyramid)
            
            # Detect keypoints in the DoG pyramid
            keypoints = self._detect_keypoints(dog_pyramid)
            
            if not keypoints:
                print("Warning: No keypoints found in initial detection")
                return [], np.array([], dtype=np.float32)
                
            # Compute keypoint orientations
            keypoints = self._compute_orientations(keypoints, gaussian_pyramid)
            
            if not keypoints:
                print("Warning: No keypoints with valid orientations")
                return [], np.array([], dtype=np.float32)
                
            # Compute SIFT descriptors
            keypoints, descriptors = self._compute_descriptors(keypoints, gaussian_pyramid)
            
            # Sort by response strength and limit number of keypoints
            if len(keypoints) > self.max_keypoints:
                keypoints_with_response = [(kp.response, i, kp) for i, kp in enumerate(keypoints)]
                keypoints_with_response.sort(reverse=True)
                
                selected_keypoints = []
                # selected_descriptors = []
                indices = []
                
                # Keep strongest keypoints while maintaining spatial distribution
                for _, i, kp in keypoints_with_response:
                    x, y = kp.pt
                    if not any(((x - existing.pt[0])**2 + (y - existing.pt[1])**2) < 10**2 
                              for existing in selected_keypoints):
                        selected_keypoints.append(kp)
                        indices.append(i)
                    
                    if len(selected_keypoints) >= self.max_keypoints:
                        break
                        
                keypoints = selected_keypoints
                if descriptors.size > 0:  # Check if we have descriptors
                    descriptors = descriptors[indices]
                
            print(f"Pure Python SIFT took {(time.time() - start_time)*1000:.2f} ms, found {len(keypoints)} keypoints")
            
            # If we somehow got no keypoints, return empty arrays
            if not keypoints:
                return [], np.array([], dtype=np.float32)
                
            return keypoints, descriptors
            
        except Exception as e:
            print(f"Error in Custom SIFT: {str(e)}")
            # Return empty results to avoid crashing the application
            return [], np.array([], dtype=np.float32)
    
    def _build_gaussian_pyramid(self, image):
        """Build a pyramid of Gaussian-blurred images"""
        pyramid = []
        
        # Initial image with additional blur
        current_image = gaussian_filter(image.astype(np.float32), sigma=self.sigma_min)
        
        for octave in range(self.n_octaves):
            octave_images = [current_image]
            
            # Generate additional scales for this octave
            for scale in range(1, self.n_scales + 3):
                # Apply incremental Gaussian blur
                blurred = gaussian_filter(octave_images[-1], sigma=self.sigma_diff[scale])
                octave_images.append(blurred)
            
            pyramid.append(octave_images)
            
            # Downsample the image for next octave
            if octave < self.n_octaves - 1:
                current_image = octave_images[self.n_scales]
                current_image = current_image[::2, ::2]  # Downsample by factor of 2
        
        return pyramid
    
    def _build_dog_pyramid(self, gaussian_pyramid):
        """Build a pyramid of Difference-of-Gaussian (DoG) images"""
        dog_pyramid = []
        
        for octave_images in gaussian_pyramid:
            dog_octave = []
            for i in range(1, len(octave_images)):
                # Compute difference between adjacent scales
                dog = octave_images[i] - octave_images[i-1]
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)
            
        return dog_pyramid
    
    def _detect_keypoints(self, dog_pyramid):
        """
        Detect keypoints in the DoG pyramid as local extrema across space and scale
        """
        keypoints = []
        
        # For each octave and scale
        for octave_idx, dog_octave in enumerate(dog_pyramid):
            for scale_idx in range(1, len(dog_octave) - 1):
                # Get current, previous and next DoG images
                prev_dog = dog_octave[scale_idx - 1]
                curr_dog = dog_octave[scale_idx]
                next_dog = dog_octave[scale_idx + 1]
                
                height, width = curr_dog.shape
                
                # Find local maxima/minima
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        # Get the center pixel value
                        center_val = curr_dog[y, x]
                        
                        # Skip low contrast regions early to save computation
                        if abs(center_val) < 0.8 * self.contrast_threshold:
                            continue
                            
                        # Check if it's a local maximum or minimum
                        # Get 3x3 neighborhood in current scale
                        neighborhood = curr_dog[y-1:y+2, x-1:x+2].copy()
                        
                        # Get 3x3 neighborhoods in adjacent scales
                        prev_neighborhood = prev_dog[y-1:y+2, x-1:x+2].copy()
                        next_neighborhood = next_dog[y-1:y+2, x-1:x+2].copy()
                        
                        # Combine all pixels except the center for max/min comparison
                        all_neighbors = np.concatenate([
                            prev_neighborhood.flatten(), 
                            neighborhood.flatten()[0:4],  # Skip center
                            neighborhood.flatten()[5:],   # Skip center
                            next_neighborhood.flatten()
                        ])
                        
                        # Compare with all neighbors
                        is_maximum = (center_val > 0) and np.all(center_val > all_neighbors)
                        is_minimum = (center_val < 0) and np.all(center_val < all_neighbors)
                        
                        if not (is_maximum or is_minimum):
                            continue
                        
                        # Refine keypoint location via quadratic fit
                        try:
                            refined_kp = self._refine_keypoint(dog_octave, scale_idx, x, y, width, height)
                            if refined_kp is None:
                                continue
                            
                            # Get refined location
                            x_refined, y_refined, s_refined, resp = refined_kp
                            
                            # Skip low contrast points after refinement
                            if abs(resp) < self.contrast_threshold:
                                continue
                            
                            # Ensure s_refined is valid index for edge check (as an int)
                            s_int = int(np.round(s_refined))
                            if s_int < 0 or s_int >= len(dog_octave):
                                continue
                                
                            # Skip edge-like points
                            try:
                                if not self._is_strong_corner(dog_octave[s_int], y_refined, x_refined):
                                    continue
                            except (IndexError, ValueError):
                                continue
                            
                            # Calculate scale
                            scale = self.sigma_min * (2 ** octave_idx) * (self.k ** scale_idx)
                            
                            # Create keypoint with default orientation (will be updated later)
                            kp = cv2.KeyPoint()
                            kp.pt = (x_refined * (2 ** octave_idx), y_refined * (2 ** octave_idx))
                            kp.size = scale * 2  # Size is 2*sigma
                            kp.angle = 0  # Temporary value, will be set later
                            kp.response = abs(resp)
                            kp.octave = octave_idx
                            kp.class_id = -1
                            
                            keypoints.append(kp)
                        except Exception as e:
                            # Silently ignore errors in keypoint refinement
                            continue
        
        return keypoints
    
    def _refine_keypoint(self, dog_octave, scale_idx, x, y, width, height):
        """
        Refine keypoint location using quadratic interpolation
        
        Returns:
            (x, y, scale, response) or None if refinement failed
        """
        # Ensure we're not at a scale boundary
        if scale_idx <= 0 or scale_idx >= len(dog_octave) - 1:
            return None
            
        # Convert to float for accurate calculations
        x, y = float(x), float(y)
        scale_idx = float(scale_idx)
        
        max_iterations = 5
        for _ in range(max_iterations):
            # Get current DoG image and adjacent scales
            s_idx = int(np.round(scale_idx))
            if s_idx <= 0 or s_idx >= len(dog_octave) - 1:
                return None
                
            prev_dog = dog_octave[s_idx - 1]
            curr_dog = dog_octave[s_idx]
            next_dog = dog_octave[s_idx + 1]
            
            # Convert to integers for indexing
            y_int, x_int = int(np.round(y)), int(np.round(x))
            
            # Border check
            if y_int <= 0 or y_int >= height-1 or x_int <= 0 or x_int >= width-1:
                return None
            
            try:
                # Compute partial derivatives
                dx = (curr_dog[y_int, x_int+1] - curr_dog[y_int, x_int-1]) / 2.0
                dy = (curr_dog[y_int+1, x_int] - curr_dog[y_int-1, x_int]) / 2.0
                ds = (next_dog[y_int, x_int] - prev_dog[y_int, x_int]) / 2.0
                
                # Compute second derivatives
                dxx = curr_dog[y_int, x_int+1] + curr_dog[y_int, x_int-1] - 2 * curr_dog[y_int, x_int]
                dyy = curr_dog[y_int+1, x_int] + curr_dog[y_int-1, x_int] - 2 * curr_dog[y_int, x_int]
                dss = next_dog[y_int, x_int] + prev_dog[y_int, x_int] - 2 * curr_dog[y_int, x_int]
                
                # Mixed derivatives
                dxy = ((curr_dog[y_int+1, x_int+1] - curr_dog[y_int+1, x_int-1]) -
                       (curr_dog[y_int-1, x_int+1] - curr_dog[y_int-1, x_int-1])) / 4.0
                dxs = ((next_dog[y_int, x_int+1] - next_dog[y_int, x_int-1]) -
                       (prev_dog[y_int, x_int+1] - prev_dog[y_int, x_int-1])) / 4.0
                dys = ((next_dog[y_int+1, x_int] - next_dog[y_int-1, x_int]) -
                       (prev_dog[y_int+1, x_int] - prev_dog[y_int-1, x_int])) / 4.0
            except IndexError:
                return None
                
            # Construct Hessian matrix
            H = np.array([
                [dxx, dxy, dxs],
                [dxy, dyy, dys],
                [dxs, dys, dss]
            ])
            
            # Construct gradient vector
            g = np.array([dx, dy, ds])
            
            # Solve linear system for offset
            try:
                # Add small regularization to avoid singular matrix
                H_reg = H + np.eye(3) * 1e-6
                offset = -np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                return None
            
            # If offset is small enough, we're done
            if max(abs(offset)) < 0.5:
                break
                
            # Update position and scale
            x = x + offset[0]
            y = y + offset[1]
            scale_idx = scale_idx + offset[2]
            
            # Check if we're out of bounds
            if (scale_idx < 1 or scale_idx > len(dog_octave)-2 or
                y < 1 or y > height-2 or x < 1 or x > width-2):
                return None
        
        # Estimate function value at peak
        try:
            value = curr_dog[y_int, x_int] + 0.5 * np.dot(g, offset)
        except (IndexError, UnboundLocalError):
            return None
        
        # Return refined position
        return (x, y, scale_idx, value)
    
    def _is_strong_corner(self, image, y, x):
        """
        Check if the keypoint is a corner, not an edge.
        Uses Hessian matrix eigenvalues ratio.
        """
        y, x = int(round(y)), int(round(x))
        
        # Make sure we're within image bounds
        h, w = image.shape
        if y < 1 or y >= h-1 or x < 1 or x >= w-1:
            return False
            
        # Compute Hessian matrix elements
        dxx = image[y, x+1] + image[y, x-1] - 2*image[y, x]
        dyy = image[y+1, x] + image[y-1, x] - 2*image[y, x]
        dxy = ((image[y+1, x+1] - image[y+1, x-1]) - 
               (image[y-1, x+1] - image[y-1, x-1])) / 4.0
        
        # Compute trace and determinant
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        
        # Avoid division by zero
        if det <= 0:
            return False
            
        # Check eigenvalue ratio
        ratio = trace * trace / det
        threshold = ((self.edge_threshold + 1) ** 2) / self.edge_threshold
        
        return ratio < threshold
    
    def _compute_orientations(self, keypoints, gaussian_pyramid):
        """
        Compute orientations for each keypoint based on local gradient histograms
        """
        oriented_keypoints = []
        
        for kp in keypoints:
            x, y = kp.pt
            octave = kp.octave
            
            # Map back to pyramid coordinates
            scale_factor = 1.0 / (2**octave)
            x_pyr = x * scale_factor
            y_pyr = y * scale_factor
            
            # Get closest scale in the pyramid
            scale_idx = int(round(np.log(kp.size/self.sigma_min/2) / np.log(self.k)))
            scale_idx = np.clip(scale_idx, 0, self.n_scales - 1)
            
            # Get image from pyramid
            img = gaussian_pyramid[octave][scale_idx]
            
            # Calculate window radius based on scale
            sigma = self.k ** scale_idx * self.sigma_min
            radius = int(round(3 * 1.5 * sigma))
            
            # Ensure keypoint is far enough from edges
            h, w = img.shape
            if x_pyr < radius or x_pyr >= w - radius or y_pyr < radius or y_pyr >= h - radius:
                continue
                
            # Create histogram of gradient orientations
            hist = np.zeros(36)
            
            x_pyr_int, y_pyr_int = int(round(x_pyr)), int(round(y_pyr))
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y_bin = y_pyr_int + dy
                    x_bin = x_pyr_int + dx
                    
                    # Skip out-of-bounds pixels
                    if x_bin < 1 or x_bin >= w-1 or y_bin < 1 or y_bin >= h-1:
                        continue
                    
                    # Compute gradient
                    dx_val = img[y_bin, x_bin+1] - img[y_bin, x_bin-1]
                    dy_val = img[y_bin+1, x_bin] - img[y_bin-1, x_bin]
                    
                    magnitude = np.sqrt(dx_val*dx_val + dy_val*dy_val)
                    orientation = np.rad2deg(np.arctan2(dy_val, dx_val)) % 360
                    
                    # Compute weight based on distance from keypoint
                    weight = np.exp(-(dx**2 + dy**2) / (2 * (1.5 * sigma)**2)) * magnitude
                    
                    # Add to histogram
                    bin_idx = int(np.floor(orientation / 10))  # 36 bins = 10 degrees each
                    hist[bin_idx % 36] += weight
            
            # Smooth the histogram
            hist_smoothed = np.copy(hist)
            for _ in range(6):  # Smooth multiple times
                hist_smoothed = np.array([0.25 * hist_smoothed[(i-1)%36] + 
                                          0.5 * hist_smoothed[i] + 
                                          0.25 * hist_smoothed[(i+1)%36] 
                                          for i in range(36)])
            
            # Find peaks in the histogram
            hist_max = np.max(hist_smoothed)
            peaks = []
            
            for i in range(36):
                left = hist_smoothed[(i-1) % 36]
                right = hist_smoothed[(i+1) % 36]
                
                if hist_smoothed[i] > 0.8 * hist_max and hist_smoothed[i] > left and hist_smoothed[i] > right:
                    # Quadratic interpolation for peak position
                    bin_center = i
                    left_val, center_val, right_val = left, hist_smoothed[i], right
                    bin_offset = 0.5 * (left_val - right_val) / (left_val - 2*center_val + right_val)
                    
                    orientation = ((bin_center + bin_offset) * 10) % 360
                    peaks.append(orientation)
            
            # Create a keypoint for each peak
            if not peaks:
                # Use highest bin if no peaks found
                orientation = np.argmax(hist_smoothed) * 10
                kp_oriented = cv2.KeyPoint(x, y, kp.size, orientation, kp.response, kp.octave)
                oriented_keypoints.append(kp_oriented)
            else:
                for orientation in peaks:
                    kp_oriented = cv2.KeyPoint(x, y, kp.size, orientation, kp.response, kp.octave)
                    oriented_keypoints.append(kp_oriented)
                    
        return oriented_keypoints
    
    def _compute_descriptors(self, keypoints, gaussian_pyramid):
        """
        Compute SIFT descriptors for all keypoints
        """
        descriptors = []
        valid_keypoints = []
        
        # Handle empty keypoints case
        if not keypoints:
            return [], np.array([], dtype=np.float32)
        
        for kp in keypoints:
            try:
                x, y = kp.pt
                octave = kp.octave
                angle_rad = np.deg2rad(kp.angle)
                
                # Skip invalid keypoints
                if kp.angle < 0:
                    continue
                
                # Map back to pyramid coordinates
                scale_factor = 1.0 / (2**octave)
                x_pyr = x * scale_factor
                y_pyr = y * scale_factor
                
                # Get closest scale in the pyramid
                scale_idx = int(round(np.log(kp.size/self.sigma_min/2) / np.log(self.k)))
                scale_idx = np.clip(scale_idx, 0, self.n_scales - 1)
                
                # Get image from pyramid
                img = gaussian_pyramid[octave][scale_idx]
                
                # Calculate descriptor parameters
                sigma = self.k ** scale_idx * self.sigma_min
                window_width = int(round(3 * sigma * 4))  # 4x4 histograms, each 3*sigma wide
                hist_width = window_width / 4.0
                
                # Precompute cos and sin for rotation
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                # Ensure keypoint is far enough from edges
                h, w = img.shape
                if (x_pyr - window_width/2 < 0 or x_pyr + window_width/2 >= w or
                    y_pyr - window_width/2 < 0 or y_pyr + window_width/2 >= h):
                    continue
                    
                # Initialize descriptor
                desc = np.zeros(128)  # 4x4 histograms with 8 orientation bins
                
                x_pyr_int, y_pyr_int = int(round(x_pyr)), int(round(y_pyr))
                
                for dy in range(-window_width//2, window_width//2):
                    for dx in range(-window_width//2, window_width//2):
                        # Rotate coordinates
                        rot_dx = cos_angle * dx + sin_angle * dy
                        rot_dy = -sin_angle * dx + cos_angle * dy
                        
                        # Bin coordinates
                        bin_x = (rot_dx / hist_width) + 1.5
                        bin_y = (rot_dy / hist_width) + 1.5
                        
                        # Skip if outside the descriptor
                        if bin_x < 0 or bin_x >= 3 or bin_y < 0 or bin_y >= 3:
                            continue
                            
                        # Get pixel location in unrotated image
                        y_img = y_pyr_int + dy
                        x_img = x_pyr_int + dx
                        
                        # Skip out-of-bounds pixels
                        if x_img < 1 or x_img >= w-1 or y_img < 1 or y_img >= h-1:
                            continue
                        
                        # Compute gradient
                        dx_val = img[y_img, x_img+1] - img[y_img, x_img-1]
                        dy_val = img[y_img+1, x_img] - img[y_img-1, x_img]
                        
                        # Rotate gradient
                        rot_dx_val = cos_angle * dx_val + sin_angle * dy_val
                        rot_dy_val = -sin_angle * dx_val + cos_angle * dy_val
                        
                        magnitude = np.sqrt(rot_dx_val*rot_dx_val + rot_dy_val*rot_dy_val)
                        orientation = np.rad2deg(np.arctan2(rot_dy_val, rot_dx_val)) % 360
                        
                        # Compute Gaussian weight
                        weight = np.exp(-(dx*dx + dy*dy) / (2 * (1.5 * sigma)**2))
                        weighted_mag = weight * magnitude
                        
                        # Compute histogram contributions with trilinear interpolation
                        bin_orientation = orientation * 8 / 360.0
                        
                        # For each of the 8 adjacent bins
                        for bin_y_idx in range(2):
                            y_idx = int(bin_y) + bin_y_idx
                            if y_idx < 0 or y_idx >= 4:
                                continue
                                
                            y_weight = 1.0 - abs(bin_y - y_idx)
                            
                            for bin_x_idx in range(2):
                                x_idx = int(bin_x) + bin_x_idx
                                if x_idx < 0 or x_idx >= 4:
                                    continue
                                    
                                x_weight = 1.0 - abs(bin_x - x_idx)
                                
                                for bin_o_idx in range(2):
                                    o_idx = int(bin_orientation) + bin_o_idx
                                    o_idx = o_idx % 8  # Wrap around for orientation
                                    
                                    o_weight = 1.0 - abs(bin_orientation - int(bin_orientation) - bin_o_idx)
                                    
                                    # Trilinear weight
                                    weight_tri = weighted_mag * x_weight * y_weight * o_weight
                                    
                                    # Update descriptor
                                    desc_idx = 32 * y_idx + 8 * x_idx + o_idx
                                    if 0 <= desc_idx < 128:
                                        desc[desc_idx] += weight_tri
                
                # Normalize descriptor vector
                norm = np.sqrt(np.sum(desc * desc))
                if norm > 1e-10:
                    desc = desc / norm
                    
                # Threshold and renormalize
                desc = np.minimum(desc, 0.2)
                norm = np.sqrt(np.sum(desc * desc))
                if norm > 1e-10:
                    desc = desc / norm
                    
                descriptors.append(desc)
                valid_keypoints.append(kp)
                
            except Exception as e:
                # Skip problematic keypoints
                continue
        
        if not descriptors:
            return [], np.array([], dtype=np.float32)
            
        descriptors = np.array(descriptors, dtype=np.float32)
        return valid_keypoints, descriptors