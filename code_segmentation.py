import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

# convert PNG to JPEG because ICC embedded is corrupt or incorrectly formatted
# the warning will still pop up because of this transformation but doesn't inder execution of code
def convert_png_to_jpeg(input_dir, output_dir):
    """
    Converts all PNG images in the input directory to JPEG format and saves them in the output directory.
    Args:
        input_dir: Directory containing PNG images.
        output_dir: Directory to save converted JPEG images.
    """
    os.makedirs(output_dir, exist_ok=True)  
    png_files = glob.glob(os.path.join(input_dir, '*.png'))  

    for png_file in png_files:
        image = cv2.imread(png_file)

        base_name = os.path.basename(png_file)
        jpeg_file = os.path.join(output_dir, base_name.replace('.png', '.jpg'))
     
        cv2.imwrite(jpeg_file, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Converted: {png_file} -> {jpeg_file}")
    return glob.glob(os.path.join(output_dir, '*.jpg'))  

# Load a grayscale image
def load_image(image_path):
    """Loads a grayscale image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# K-Means Segmentation Algorithm
def k_means_segmentation(image, k, max_iter=100, tol=1e-4):
    """
    Performs K-means clustering for image segmentation.
    Args:
        image: grayscale image as a 2D NumPy array.
        k: number of clusters.
        max_iter: maximum number of iterations for Lloyd's algorithm.
        tol: convergence tolerance for centroid changes.
    Returns:
        segmented_image: the segmented image with pixel intensities replaced by their cluster means.
    """
    flat_image = image.flatten().astype(float)

    # Randomly initialize k centroids from the pixel intensities
    centroids = np.random.choice(flat_image, k, replace=False)

    for iteration in range(max_iter):
        # Assign each pixel to the nearest centroid
        distances = np.abs(flat_image[:, None] - centroids) 
        labels = np.argmin(distances, axis=1)  

        # Update centroids as the mean of the pixels in each cluster
        new_centroids = np.array([flat_image[labels == i].mean() if np.any(labels == i) else centroids[i] for i in range(k)])
        
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    # pixel values --> cluster mean
    segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)
    return segmented_image

def plot_k_means_results(image_path, original_image, segmented_images, k_values):
    """
    Plots the original and segmented images for comparison.
    Args:
        image_path: path of the image being processed.
        original_image: the original grayscale image.
        segmented_images: a list of segmented images for different k values.
        k_values: a list of k values corresponding to the segmented images.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(segmented_images) + 1, 1)
    plt.title(f"Original Image\n{os.path.basename(image_path)}")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    for i, (segmented_image, k) in enumerate(zip(segmented_images, k_values)):
        plt.subplot(1, len(segmented_images) + 1, i + 2)
        plt.title(f"K-Means (k={k})")
        plt.imshow(segmented_image, cmap='gray')
        plt.axis('off')

    plt.show()

# Single Threshold (2 Classes)
def otsu_thresholding(image):
    """
    Classic Otsu (single threshold => 2 classes).
    Returns binary_image (0 or 255), and the threshold.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0,256))
    total = image.size
    
    sum_total = np.dot(hist, np.arange(256))
    sum_bg, w_bg = 0.0, 0.0
    
    max_var = -1
    best_t = 0
    
    for t in range(256):
        w_bg += hist[t]
        sum_bg += t * hist[t]
        
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        
        between_var = w_bg * w_fg * (mean_bg - mean_fg)**2
        if between_var > max_var:
            max_var = between_var
            best_t = t
    
    bin_img = (image > best_t).astype(np.uint8) * 255
    return bin_img, best_t

# Two Thresholds (3 Classes)
def otsu_thresholding_2(image):
    """
    Two thresholds => 3 classes.
    Returns segmented_image (0, 128, 255) and thresholds (t1, t2).
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0,256))
    total = image.size
    
    p = hist / float(total)
    P = np.cumsum(p)
    S = np.cumsum(np.arange(256) * p)
    mG = S[-1]  
    
    max_var = -1
    best_t1, best_t2 = 0, 0
    
    for t1 in range(256):
        for t2 in range(t1+1, 256):
            w0 = P[t1]                     # class0: [0..t1]
            w1 = P[t2] - P[t1]            # class1: (t1..t2]
            w2 = 1.0 - P[t2]              # class2: (t2..255]
            if w0 < 1e-12 or w1 < 1e-12 or w2 < 1e-12:
                continue
            
            m0 = S[t1] / w0
            m1 = (S[t2] - S[t1]) / w1
            m2 = (S[-1] - S[t2]) / w2
            
            var_between = (w0*(m0 - mG)**2 + 
                           w1*(m1 - mG)**2 + 
                           w2*(m2 - mG)**2)
            if var_between > max_var:
                max_var = var_between
                best_t1, best_t2 = t1, t2
    
    seg_img = np.zeros_like(image)
    seg_img[image <= best_t1] = 0
    seg_img[(image > best_t1) & (image <= best_t2)] = 128
    seg_img[image > best_t2] = 255
    
    return seg_img, best_t1, best_t2

# Three Thresholds (4 Classes) 
def otsu_thresholding_3(image):
    """
    Three thresholds => 4 classes.
    Returns segmented_image (0, 85, 170, 255) and thresholds (t1, t2, t3).
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0,256))
    total = image.size
    
    p = hist / float(total)
    P = np.cumsum(p)
    S = np.cumsum(np.arange(256) * p)
    mG = S[-1]
    
    max_var = -1
    best_t1, best_t2, best_t3 = 0, 0, 0
    
    for t1 in range(256):
        for t2 in range(t1+1, 256):
            for t3 in range(t2+1, 256):
                w0 = P[t1]                   # class0: [0..t1]
                w1 = P[t2] - P[t1]          # class1: (t1..t2]
                w2 = P[t3] - P[t2]          # class2: (t2..t3]
                w3 = 1.0 - P[t3]            # class3: (t3..255]
                
                # Avoid empty classes
                if w0 < 1e-12 or w1 < 1e-12 or w2 < 1e-12 or w3 < 1e-12:
                    continue
                
                m0 = S[t1] / w0
                m1 = (S[t2] - S[t1]) / w1
                m2 = (S[t3] - S[t2]) / w2
                m3 = (S[-1] - S[t3]) / w3
                
                var_between = (w0*(m0 - mG)**2 +
                               w1*(m1 - mG)**2 +
                               w2*(m2 - mG)**2 +
                               w3*(m3 - mG)**2)
                if var_between > max_var:
                    max_var = var_between
                    best_t1, best_t2, best_t3 = t1, t2, t3
    
    seg_img = np.zeros_like(image)
    seg_img[image <= best_t1] = 0
    seg_img[(image > best_t1) & (image <= best_t2)] = 85
    seg_img[(image > best_t2) & (image <= best_t3)] = 170
    seg_img[image > best_t3] = 255
    
    return seg_img, best_t1, best_t2, best_t3

def clean_segmentation(image, neighborhood=4, threshold_ratio=1.0, iterations=1):
    """
    Cleans and denoises the segmented image by removing small holes.
    Args:
        image: segmented image (2D NumPy array) with discrete labels.
        neighborhood: 4 or 8 for the neighborhood system.
        threshold_ratio: ratio of neighbors that need to agree for majority label assignment 
                         (e.g., 1.0 means all neighbors must agree).
        iterations: number of times to apply the cleaning algorithm.
    Returns:
        cleaned_image: denoised segmented image.
    """
    def get_neighbors(x, y, img_shape, neighborhood):
        neighbors = []
        rows, cols = img_shape
        
        # 4-pixel neighborhood
        if neighborhood >= 4:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Top, Bottom, Left, Right
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    neighbors.append((nx, ny))
        
        # 8-pixel neighborhood (add diagonals)
        if neighborhood == 8:
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Diagonals
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    neighbors.append((nx, ny))
        
        return neighbors

    cleaned_image = image.copy()
    rows, cols = image.shape

    for _ in range(iterations):
        new_image = cleaned_image.copy()
        
        for x in range(rows):
            for y in range(cols):
                neighbors = get_neighbors(x, y, (rows, cols), neighborhood)
                neighbor_labels = [cleaned_image[nx, ny] for nx, ny in neighbors]
                
                # Count labels and determine majority
                label_counts = {label: neighbor_labels.count(label) for label in set(neighbor_labels)}
                majority_label = max(label_counts, key=label_counts.get)
                required_votes = int(threshold_ratio * len(neighbors))
                
                # Assign majority label if threshold is met
                if label_counts[majority_label] >= required_votes:
                    new_image[x, y] = majority_label
        
        cleaned_image = new_image

    return cleaned_image


# Main Execution
if __name__ == "__main__":
# ---------------------------------------------------------------------
# 1) Gather Images
# ---------------------------------------------------------------------
    data_dir = "images"  
    converted_dir = "converted_images"  

    image_paths = convert_png_to_jpeg(data_dir, converted_dir)

# ---------------------------------------------------------------------
# 1) K - Means
# ---------------------------------------------------------------------

    for image_path in image_paths:
        original_image = load_image(image_path)

        # Apply K-means segmentation for different k values
        k_values = [2, 3, 4, 5]  # Experiment with different k values
        segmented_images = [k_means_segmentation(original_image, k) for k in k_values]
        
        plot_k_means_results(image_path, original_image, segmented_images, k_values)

# ---------------------------------------------------------------------
# 2) Otsu's Thresholding
# ---------------------------------------------------------------------
    for image_path in image_paths:
        original_image = load_image(image_path)

        # ---- 1) Single-level Otsu
        bin_img_2class, t_1 = otsu_thresholding(original_image)

        # ---- 2) Two thresholds -> 3 classes
        seg_img_3class, t1_2, t2_2 = otsu_thresholding_2(original_image)

        # ---- 3) Three thresholds -> 4 classes
        seg_img_4class, t1_3, t2_3, t3_3 = otsu_thresholding_3(original_image)

        # ---- Print thresholds
        print("\nImage:", os.path.basename(image_path))
        print(f"  Single Otsu threshold = {t_1}")
        print(f"  Two thresholds = ({t1_2}, {t2_2})")
        print(f"  Three thresholds = ({t1_3}, {t2_3}, {t3_3})")


        fig1, axs1 = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        
        axs1[0].imshow(original_image, cmap='gray')
        axs1[0].set_title("Original")
        axs1[0].axis('off')

        axs1[1].imshow(bin_img_2class, cmap='gray')
        axs1[1].set_title(f"2-Class Otsu (T={t_1})")
        axs1[1].axis('off')

        axs1[2].imshow(seg_img_3class, cmap='gray')
        axs1[2].set_title(f"3-Class Otsu (T1={t1_2}, T2={t2_2})")
        axs1[2].axis('off')

        axs1[3].imshow(seg_img_4class, cmap='gray')
        axs1[3].set_title(f"4-Class Otsu\n(T1={t1_3}, T2={t2_3}, T3={t3_3})")
        axs1[3].axis('off')

        plt.suptitle(f"Threshold Comparisons for {os.path.basename(image_path)}", fontsize=16)
        plt.tight_layout()
        plt.show()

        hist, _ = np.histogram(original_image.ravel(), bins=256, range=(0,256))

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Plot the histogram
        ax2.bar(range(256), hist, color='blue', width=1.0)
        
        # Single threshold 
        ax2.axvline(x=t_1, color='red', linestyle='--', linewidth=2, 
                    label=f"Single Otsu: {t_1}")

        # Two thresholds 
        ax2.axvline(x=t1_2, color='green', linestyle='--', linewidth=2, 
                    label=f"2-Threshold: {t1_2}")
        ax2.axvline(x=t2_2, color='green', linestyle='--', linewidth=2, 
                    label=f"2-Threshold: {t2_2}")

        # Three thresholds
        ax2.axvline(x=t1_3, color='orange', linestyle='--', linewidth=2, 
                    label=f"3-Threshold: {t1_3}")
        ax2.axvline(x=t2_3, color='orange', linestyle='--', linewidth=2, 
                    label=f"3-Threshold: {t2_3}")
        ax2.axvline(x=t3_3, color='orange', linestyle='--', linewidth=2, 
                    label=f"3-Threshold: {t3_3}")

        ax2.set_title(f"Histogram + All Otsu Thresholds\n({os.path.basename(image_path)})")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Count")
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    
# ---------------------------------------------------------------------
# 3) Cleaning and Denoising
# ---------------------------------------------------------------------

    for image_path in image_paths:
        original_image = load_image(image_path)

        # Apply K-means segmentation
        k = 2
        segmented_image = k_means_segmentation(original_image, k)

        # Clean the segmented image with different parameters
        cleaned_image_4_50 = clean_segmentation(segmented_image, neighborhood=4, threshold_ratio=0.50, iterations=3)
        cleaned_image_4_75 = clean_segmentation(segmented_image, neighborhood=4, threshold_ratio=0.75, iterations=3)
        cleaned_image_4_1 = clean_segmentation(segmented_image, neighborhood=4, threshold_ratio=1, iterations=3)
        cleaned_image_8_50 = clean_segmentation(segmented_image, neighborhood=8, threshold_ratio=0.5, iterations=3)
        cleaned_image_8_75 = clean_segmentation(segmented_image, neighborhood=8, threshold_ratio=0.75, iterations=3)
        cleaned_image_8_1 = clean_segmentation(segmented_image, neighborhood=8, threshold_ratio=1, iterations=3)

        # Plot results to compare
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows, 4 columns

        # Add the images and titles to each subplot
        titles = [
            "Original", 
            "Segmented (k=2)", 
            "Cleaned (4-pixel, T=0.5)", 
            "Cleaned (4-pixel, T=0.75)", 
            "Cleaned (4-pixel, T=1)", 
            "Cleaned (8-pixel, T=0.5)", 
            "Cleaned (8-pixel, T=0.75)", 
            "Cleaned (8-pixel, T=1)"
        ]

        images = [
            original_image, 
            segmented_image, 
            cleaned_image_4_50, 
            cleaned_image_4_75, 
            cleaned_image_4_1, 
            cleaned_image_8_50, 
            cleaned_image_8_75, 
            cleaned_image_8_1
        ]

        # Loop through each axis and add the corresponding image and title
        for ax, img, title in zip(axes.flat, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12)
            ax.axis('off')

        # Adjust spacing between rows and columns
        plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Add space between plots
        plt.suptitle("Comparison of Cleaning Results", fontsize=16)
        plt.show()


        # Compare different iterations side by side
        iterations = [1, 3, 5, 10]
        cleaned_images = []

        # Generate cleaned images for different iterations
        for it in iterations:
            cleaned_image = clean_segmentation(segmented_image, neighborhood=4, threshold_ratio=0.75, iterations=it)
            cleaned_images.append(cleaned_image)

        # Plot all results in a single figure
        plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
        for i, (it, cleaned_image) in enumerate(zip(iterations, cleaned_images)):
            plt.subplot(1, len(iterations), i + 1)  # Create a subplot for each iteration
            plt.title(f"Iterations: {it}")
            plt.imshow(cleaned_image, cmap='gray')
            plt.axis('off')

        plt.suptitle("Comparison of Cleaning Results Across Iterations", fontsize=16)
        plt.show()


