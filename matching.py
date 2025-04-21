import numpy as np
import cv2
import time 


def harris_detector(image_orignal):
    # convert to grayscale
    image = image_orignal.copy()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = np.float32(gray)  #we must convert it to float for better results

    # start the time
    start_time = time.time()

    ##################### *Harris Corner Detection and lambda_min* ######
    # R= det(H) - k*(Trace(H))^2

    # calc gradients using soble edge detection
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # calc the matrices
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # blurring with gaussian
    blockSize = 5
    Sigma = 1
    Sxx = cv2.GaussianBlur(Ixx, (blockSize, blockSize), sigmaX=Sigma)
    Syy = cv2.GaussianBlur(Iyy, (blockSize, blockSize), sigmaX=Sigma)
    Sxy = cv2.GaussianBlur(Ixy, (blockSize, blockSize), sigmaX=Sigma)

    # harris equation: R= det(H) - k*(Trace(H))^2
    k = 0.04
    detH = Sxx * Syy - Sxy ** 2
    traceH = Sxx + Syy
    corners = detH - k * (traceH ** 2)

    # calc lambda minus: using equation that is big
    term = np.sqrt(((Sxx - Syy) ** 2) / 4 + Sxy ** 2)
    lambda1 = (traceH / 2) + term
    lambda2 = (traceH / 2) - term
    lambda_min = np.minimum(lambda1, lambda2)

    # calc the time and print it
    elapsed_time = time.time() - start_time
    print(f"Computation Time: {elapsed_time:.4f} seconds")

    # for better visualization
    corners = cv2.dilate(corners, None)
    # lambda_min = cv2.dilate(lambda_min, None)

    # threshold and highlight the corners with red dots and lambda_min with green dots
    # image[corners > 0.01 * corners.max()] = [0, 0, 255]
    # image[lambda_min > 0.1 * lambda_min.max()] = [0, 255, 0]
    harris_image = image.copy()
    lambda_min_image = image.copy()
    
    harris_image[corners > 0.01 * corners.max()] = [0, 0, 255]
    lambda_min_image[lambda_min > 0.1 * lambda_min.max()] = [0, 255, 0]

    
    return harris_image, lambda_min_image



def matching_ratio_test(matches, ratio=0.8):
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



def match_ncc( desc1, desc2):
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

def match_ssd(desc1, desc2):
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