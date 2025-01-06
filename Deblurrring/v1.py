import cv2
import numpy as np
from scipy.signal import wiener
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.metrics import structural_similarity as ssim


def load_image(filepath):
    """Load an image from the specified file path."""
    return cv2.imread(filepath)

def convert_to_grayscale(image):
    """Convert the image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise_image(image):
    """Apply Gaussian blur to denoise the image."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def histogram_equalization(image):
    """Enhance the contrast of the image using histogram equalization."""
    return cv2.equalizeHist(image)

def edge_detection(image):
    """Detect edges in the image using the Canny edge detector."""
    # Ensure the image is in 8-bit grayscale format
    image_8bit = np.uint8(image) if image.max() <= 1 else image
    return cv2.Canny(image_8bit, 100, 200)

def apply_wiener_filter(image, noise_variance=0.01):
    """Apply Wiener filter for deblurring the image."""
    # Normalize the image to [0, 1] for wiener filter
    image_normalized = image / 255.0
    # Apply Wiener filter
    return wiener(image_normalized, (5, 5), noise_variance)

def sharpen_image(image):
    """Sharpen the image using a sharpening kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def display_images_in_grid(images, titles, cols=3):
    """Display images in a grid."""
    rows = len(images) // cols + int(len(images) % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i])
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused axes

    plt.tight_layout()
    plt.show()

def estimate_noise_variance(image):
    """Estimate the noise variance of an image by comparing to a smoothed version."""
    # Smooth the image using a Gaussian filter
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Calculate the noise by subtracting the smoothed image from the original
    noise = image - smoothed_image
    
    # Estimate the noise variance by calculating the standard deviation of the noise
    noise_variance = np.var(noise)
    
    return noise_variance

def main(filepath):
    # Load the image
    image = load_image(filepath)
    
    # Convert to grayscale
    gray_image = convert_to_grayscale(image)

    # Denoise the image
    denoised_image = denoise_image(gray_image)

    # Enhance contrast
    equalized_image = histogram_equalization(denoised_image)

    # Additional contrast enhancement using skimage
    contrast_enhanced_image = exposure.equalize_adapthist(equalized_image)

    # Detect edges
    edges = edge_detection(contrast_enhanced_image)

    # Apply Wiener filter with modified noise_variance
    wiener_filtered_image = apply_wiener_filter(contrast_enhanced_image, noise_variance=estimate_noise_variance(contrast_enhanced_image))

    # Sharpen the image
    sharpened_image = sharpen_image(wiener_filtered_image)

    # List of images and titles
    images = [gray_image, denoised_image, equalized_image, contrast_enhanced_image, edges, wiener_filtered_image, sharpened_image]
    titles = ['Grayscale Image', 'Denoised Image', 'Histogram Equalized Image', 'Contrast Enhanced Image', 'Edges Detected', 'Wiener Filtered Image', 'Sharpened Image']

    # Display images in grid
    display_images_in_grid(images, titles)

    # List of images and titles for a separate grid
    images = [image, gray_image, sharpened_image]
    titles = ["Original Image",'Grayscale Image', 'Sharpened Image']

    # Display images in grid
    display_images_in_grid(images, titles)


if __name__ == "__main__":
    # Provide the path to your blurred image
    image_path = '../train/train_blur_jpeg/003/00000000.jpg'
    main(image_path)
