import cv2
import numpy as np
from scipy.signal import wiener
from matplotlib import pyplot as plt
from skimage import exposure

def load_image(filepath):
    """Load an image from the specified file path."""
    return cv2.imread(filepath)

def denoise_image(image):
    """Apply Gaussian blur to denoise the image."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def histogram_equalization(image):
    """Enhance the contrast of the image using histogram equalization."""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def edge_detection(image):
    """Detect edges in the image using the Canny edge detector."""
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_wiener_filter(image, noise_variance=0.01):
    """Apply Wiener filter for deblurring the image."""
    channels = []
    for i in range(3):  # Loop through each color channel
        channel_wiener = wiener(image[:,:,i], (5, 5), noise_variance)
        channels.append(channel_wiener)
    return np.stack(channels, axis=-1)

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
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
            ax.set_title(titles[i])
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused axes

    plt.tight_layout()
    plt.show()

def main(filepath):
    # Load the image
    image = load_image(filepath)

    # Denoise the image
    denoised_image = denoise_image(image)

    # Enhance contrast
    equalized_image = histogram_equalization(denoised_image)

    # Additional contrast enhancement using skimage
    contrast_enhanced_image = exposure.equalize_adapthist(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2LAB))
    contrast_enhanced_image = (contrast_enhanced_image * 255).astype(np.uint8)  # Convert back to uint8

    # Detect edges
    edges = edge_detection(contrast_enhanced_image)

    # Apply Wiener filter with modified noise_variance
    wiener_filtered_image = apply_wiener_filter(contrast_enhanced_image, noise_variance=0.05)

    # Sharpen the image
    sharpened_image = sharpen_image(wiener_filtered_image)

    # List of images and titles
    images = [image, denoised_image, equalized_image, contrast_enhanced_image, edges, wiener_filtered_image, sharpened_image]
    titles = ['Original Image', 'Denoised Image', 'Histogram Equalized Image', 'Contrast Enhanced Image', 'Edges Detected', 'Wiener Filtered Image', 'Sharpened Image']

    # Display images in grid
    display_images_in_grid(images, titles)

if __name__ == "__main__":
    # Provide the path to your blurred image
    image_path = 'train/train_blur_jpeg/003/00000000.jpg'
    main(image_path)
