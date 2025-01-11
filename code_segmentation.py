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
