import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def read_image_and_params(file_path):
    # Check if the directory contains an image and a txt or npy file
    image_file = None
    param_file = None
    for file in os.listdir(file_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_file = os.path.join(file_path, file)
        elif file.endswith('.txt') or file.endswith('.npy'):
            param_file = os.path.join(file_path, file)
    
    if not image_file or not param_file:
        raise FileNotFoundError("Image file or parameter file not found in the directory.")
    
    return image_file, param_file

def extract_blur_type(file_path):
    # Extract the blur type from the file path
    parts = file_path.split('/')
    if len(parts) < 3:
        raise ValueError("Invalid file path structure. Expected structure: ../pic_manually_blurred/blur_type/base_name_index/")
    return parts[-3]  # Assuming the structure is ../pic_manually_blurred/blur_type/base_name_index/

def read_params(param_file, blur_type):
    # Read the parameters from the txt or npy file based on the blur type
    if blur_type in ['motion', 'directional']:
        kernel = np.load(param_file)
        return kernel
    else:
        params = {}
        with open(param_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                if key == 'kernel_size':
                    params[key] = tuple(map(int, value.strip('()').split(',')))
                else:
                    params[key] = int(value) if value.isdigit() else float(value)
        return params

def deblur_image(image, kernel, noise_var=0.01):
    # Normalize the kernel
    kernel /= kernel.sum()

    # Fourier transform of the image and the kernel
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)

    # Wiener filter formula
    kernel_fft_conj = np.conj(kernel_fft)
    wiener_filter_fft = kernel_fft_conj / (np.abs(kernel_fft) ** 2 + noise_var)
    deblurred_fft = image_fft * wiener_filter_fft

    # Inverse Fourier transform to get the deblurred image
    deblurred_image = np.fft.ifft2(deblurred_fft)
    deblurred_image = np.abs(deblurred_image)
    
    # Clip values and convert to uint8
    deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)

    return deblurred_image

# def denoise_image(image):
#     # Apply Non-Local Means Denoising
#     denoised_nlm = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

#     # Apply Median Filtering if needed
#     denoised_median = cv2.medianBlur(image, ksize=5)

#     # Apply Gaussian Blur if needed
#     denoised_gaussian = cv2.GaussianBlur(image, (5, 5), sigmaX=1)
    
#     # Display results for comparison
#     plt.figure(figsize=(20, 10))

#     plt.subplot(2, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Noisy Image')
#     plt.axis('off')

#     plt.subplot(2, 2, 2)
#     plt.imshow(denoised_nlm, cmap='gray')
#     plt.title('Denoised (Non-Local Means)')
#     plt.axis('off')

#     plt.subplot(2, 2, 3)
#     plt.imshow(denoised_median, cmap='gray')
#     plt.title('Denoised (Median Filter)')
#     plt.axis('off')

#     plt.subplot(2, 2, 4)
#     plt.imshow(denoised_gaussian, cmap='gray')
#     plt.title('Denoised (Gaussian Blur)')
#     plt.axis('off')

#     plt.show()

#     chosen_denoised_image = input(
#         """Choose the denoised image to use for deconvolution:
#         1. Non-Local Means Denoising
#         2. Median Filtering
#         3. Gaussian Blur
#         Enter 1, 2, or 3:
#         """)  # You can choose any of the denoised images

#     if chosen_denoised_image == '1':
#         return denoised_nlm
#     elif chosen_denoised_image == '2':
#         return denoised_median
#     elif chosen_denoised_image == '3':
#         return denoised_gaussian
#     else:
#         print("Invalid choice. Using Non-Local Means Denoising by default.")
#         return denoised_nlm

# def sharpen_image(image):
#     """
#     Apply sharpening to the image using an unsharp mask.
#     """
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5,-1],
#                        [0, -1, 0]])  # Simple sharpening kernel
#     return cv2.filter2D(image, -1, kernel)


def main():
    '''
    Performance:
    - Gaussian: good
    - directional: ok, but has artifacts
    - bilateral: not implemented
    - median: not implemented
    - box: good
    - motion: ok, but has artifacts
    '''
    # Ask for the file path
    file_path = "../pic_manually_blurred/box/car1_1/" 
    
    # Read the image and parameters
    image_file, param_file = read_image_and_params(file_path)
    
    # Extract the blur type
    blur_type = extract_blur_type(file_path)
    
    # Read the parameters
    params = read_params(param_file, blur_type)
    
    # Read the image
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    # Denoise the image
    #denoiseImage = denoise_image(image)

    # Enhance the image using histogram equalization
    #imageGray_eq = cv2.equalizeHist(denoiseImage)

    # Deblur the image
    if blur_type in ['motion', 'directional']:
        kernel = params
        deblurred_image = deblur_image(image, kernel)
    else:
        # For other blur types, create a kernel based on the parameters
        if blur_type == 'gaussian':
            kernel_size = params['kernel_size']
            sigma = params['sigma']
            kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
            kernel = np.outer(kernel, kernel)
        elif blur_type == 'bilateral':
            # Bilateral filter parameters are not directly used for deblurring
            raise NotImplementedError("Deblurring for bilateral blur is not implemented.")
        elif blur_type == 'median':
            # Median blur parameters are not directly used for deblurring
            raise NotImplementedError("Deblurring for median blur is not implemented.")
        elif blur_type == 'box':
            kernel_size = params['ksize']
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        deblurred_image = deblur_image(image, kernel)
    
    # Sharpen the deblurred image
    # sharpened_image = sharpen_image(deblurred_image)

    # Display the original and deblurred images
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Blurred Image')
    plt.axis('off')
    
    # plt.subplot(2, 2, 2)
    # plt.imshow(imageGray_eq, cmap='gray')
    # plt.title('Histogram Equalized Image')
    # plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(deblurred_image, cmap='gray')
    plt.title('Deblurred Image')
    plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.imshow(sharpened_image, cmap='gray')
    # plt.title('Deblurred and sharpened Image')
    # plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main()