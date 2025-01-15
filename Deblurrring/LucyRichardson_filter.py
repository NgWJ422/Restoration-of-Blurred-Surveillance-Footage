import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, img_as_float


# Constants
Imagepath = '../pic_manually_blurred/motion/hexagon_building_1/blurred_hexagon_building.png'
number_of_iterations = 30  # Number of iterations for Lucy-Richardson deconvolution

# Kernel definitions
def create_kernels():
    # 1. Identity Kernel (No Blur)
    identity_kernel = np.ones((5, 5))  # Simple 5x5 kernel of ones
    identity_kernel /= identity_kernel.sum()  # Normalize the kernel
    
    # 2. Gaussian Kernel
    gaussian_kernel = cv2.getGaussianKernel(5, 1)  # Create a 5x5 Gaussian kernel
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # Make it 2D by matrix multiplication
    
    # 3. Random Kernel (for blind deconvolution)
    random_kernel = np.random.rand(5, 5)  # Random values in a 5x5 matrix
    random_kernel /= random_kernel.sum()  # Normalize the kernel
    
    # 4. Motion Blur Kernel
    kernel_size = 5  # Size of the blur kernel
    motion_blur_kernel = np.zeros((kernel_size, kernel_size))
    motion_blur_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)  # Horizontal motion blur
    motion_blur_kernel /= kernel_size  # Normalize the kernel
    
    return identity_kernel, gaussian_kernel, random_kernel, motion_blur_kernel

def denoise_image(image):
    # Apply Non-Local Means Denoising
    denoised_nlm = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Apply Median Filtering if needed
    denoised_median = cv2.medianBlur(image, ksize=5)

    # Apply Gaussian Blur if needed
    denoised_gaussian = cv2.GaussianBlur(image, (5, 5), sigmaX=1)
    
    # Display results for comparison
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Noisy Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(denoised_nlm, cmap='gray')
    plt.title('Denoised (Non-Local Means)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(denoised_median, cmap='gray')
    plt.title('Denoised (Median Filter)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(denoised_gaussian, cmap='gray')
    plt.title('Denoised (Gaussian Blur)')
    plt.axis('off')

    plt.show()

    chosen_denoised_image = input(
        """Choose the denoised image to use for deconvolution:
        1. Non-Local Means Denoising
        2. Median Filtering
        3. Gaussian Blur
        Enter 1, 2, or 3:
        """)  # You can choose any of the denoised images

    if chosen_denoised_image == '1':
        return denoised_nlm
    elif chosen_denoised_image == '2':
        return denoised_median
    elif chosen_denoised_image == '3':
        return denoised_gaussian
    else:
        print("Invalid choice. Using Non-Local Means Denoising by default.")
        return denoised_nlm

def prepare_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Denoise the image
    denoiseImage = denoise_image(imageGray)

    # Enhance the image using histogram equalization
    imageGray_eq = cv2.equalizeHist(denoiseImage)
    
    # Prepare image for deconvolution
    image_float = img_as_float(imageGray_eq)
    
    return imageGray, image_float

def lucy_richardson_deconvolution(image_float, kernel):
    # Perform Lucy-Richardson deconvolution
    restored_image = restoration.richardson_lucy(image_float, kernel, num_iter=number_of_iterations)
    
    # Convert the restored image back to uint8 for displaying
    restored_image = np.uint8(np.clip(restored_image * 255, 0, 255))
    
    return restored_image

def sharpen_image(image):
    """
    Apply sharpening to the image using an unsharp mask.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])  # Simple sharpening kernel
    return cv2.filter2D(image, -1, kernel)

def smooth_image(image, kernel_size=5):
    """
    Apply smoothing (Gaussian blur) to the image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Main processing function
def main(file_path):

    # Prepare image for deconvolution
    imageGray, image_float = prepare_image(file_path)
    
    # Create different kernels
    identity_kernel, gaussian_kernel, random_kernel, motion_blur_kernel = create_kernels()
    
    # List of kernels
    kernels = {
        "Identity Kernel": identity_kernel,
        "Gaussian Kernel": gaussian_kernel,
        "Motion Blur Kernel": motion_blur_kernel,
        "Random Kernel": random_kernel
    }

    # Perform deblurring with each kernel
    restored_images = {}
    for kernel_name, kernel in kernels.items():
        restored_image = lucy_richardson_deconvolution(image_float, kernel)
        restored_images[kernel_name] = restored_image

    # Display original and deblurred images
    plt.figure(figsize=(10, 10))

    # Subplot for the original image
    plt.subplot(2, 3, 1)
    plt.imshow(imageGray, cmap='gray')
    plt.title('Original Blurred Image')
    plt.axis('off')

    # Subplots for restored images using different kernels
    for i, (kernel_name, restored_image) in enumerate(restored_images.items(), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(restored_image, cmap='gray')
        plt.title(f'Restored Image ({kernel_name})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    sharpen_images = {}
    # Sharpen the restored image
    for i in restored_images:
        sharpen_images[i] = sharpen_image(restored_images[i])
    # Display original and deblurred images
    plt.figure(figsize=(10, 10))

    # Subplot for the original image
    plt.subplot(2, 3, 1)
    plt.imshow(imageGray, cmap='gray')
    plt.title('Original Blurred Image')
    plt.axis('off')

    # Subplots for restored images using different kernels
    for i, (kernel_name, sharpen_images) in enumerate(sharpen_images.items(), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(sharpen_images, cmap='gray')
        plt.title(f'Restored And Sharpen ({kernel_name})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main(Imagepath)
