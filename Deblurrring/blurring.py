import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to apply Gaussian Blur
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Function to apply Median Blur
def apply_median_blur(image, ksize=5):
    return cv2.medianBlur(image, ksize)

# Function to apply Box Blur
def apply_box_blur(image, ksize=5):
    return cv2.blur(image, (ksize, ksize))

# Function to apply Motion Blur
def apply_motion_blur(image, kernel_size=15, angle=0):
    # Create the motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    
    for i in range(kernel_size):
        x = int(center + i * direction[0])
        y = int(center + i * direction[1])
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[x, y] = 1

    kernel = kernel / kernel.sum()  # Normalize the kernel
    print(kernel)
    return cv2.filter2D(image, -1, kernel), kernel

# Function to apply Bilateral Blur
def apply_bilateral_blur(image, d=15, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Function to apply Directional Blur
def apply_directional_blur(image, ksize, angle):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    angle = np.deg2rad(angle)
    direction = np.array([np.cos(angle), np.sin(angle)])

    for i in range(ksize):
        for j in range(ksize):
            offset = np.array([i - center, j - center])
            if np.all(offset == (direction * np.dot(offset, direction)).astype(int)):
                kernel[i, j] = 1

    kernel = kernel / kernel.sum()  # Normalize the kernel
    print(kernel)
    return cv2.filter2D(image, -1, kernel), kernel

def get_next_index(save_dir, base_name):
    existing_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d)) and d.startswith(base_name)]
    indices = [int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()]
    return max(indices, default=0) + 1

# Function to handle image blurring
def blur_image(image_path, blur_type):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if blur_type == "gaussian":
        kernel_size = tuple(map(int, input("Enter kernel size (e.g., 5 5): ").split()))
        sigma = int(input("Enter sigma (0 for default): "))
        blurred_image = apply_gaussian_blur(image, kernel_size, sigma)
    
    elif blur_type == "median":
        ksize = int(input("Enter kernel size (odd number): "))
        blurred_image = apply_median_blur(image, ksize)
    
    elif blur_type == "box":
        ksize = int(input("Enter kernel size (odd number): "))
        blurred_image = apply_box_blur(image, ksize)
    
    elif blur_type == "motion":
        kernel_size = int(input("Enter kernel size for motion blur: "))
        angle = int(input("Enter angle for motion blur (0-360): "))
        blurred_image, kernel = apply_motion_blur(image, kernel_size, angle)
    
    elif blur_type == "bilateral":
        d = int(input("Enter diameter of pixel neighborhood: "))
        sigma_color = int(input("Enter sigma for color: "))
        sigma_space = int(input("Enter sigma for space: "))
        blurred_image = apply_bilateral_blur(image, d, sigma_color, sigma_space)
    
    elif blur_type == "directional":
        ksize = int(input("Enter kernel size for directional blur: "))
        angle = int(input("Enter angle for directional blur (0-360): "))
        blurred_image, kernel = apply_directional_blur(image, ksize, angle)
    
    else:
        print("Unknown blur type!")
        return None
    
    # Display results for comparison
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blurred_image)
    plt.title('Blurred Image')
    plt.axis('off')
    plt.show()

    # Convert the image back to BGR for saving
    blurred_image_bgr = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

    # Save the blurred image
    save_dir = "../pic_manually_blurred"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate the save path with incremented index
    base_name = os.path.basename(image_path).split('.')[0]
    blur_type_dir = os.path.join(save_dir, blur_type)
    if not os.path.exists(blur_type_dir):
        os.makedirs(blur_type_dir)
    index = get_next_index(blur_type_dir, base_name)
    save_path = os.path.join(blur_type_dir, f"{base_name}_{index}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the image in BGR format
    image_save_path = os.path.join(save_path, f"blurred_{base_name}.png")
    cv2.imwrite(image_save_path, blurred_image_bgr)
    print(f"Blurred image saved at: {image_save_path}")

    # Save the kernel or parameters
    if blur_type in ['motion', 'directional']:
        kernel_save_path = os.path.join(save_path, "kernel.npy")
        np.save(kernel_save_path, kernel)
        print(f"Kernel saved at: {kernel_save_path}")
    else:
        params_save_path = os.path.join(save_path, "params.txt")
        with open(params_save_path, 'w') as f:
            if blur_type == 'bilateral':
                f.write(f"d: {d}\nsigma_color: {sigma_color}\nsigma_space: {sigma_space}\n")
            elif blur_type == 'gaussian':
                f.write(f"kernel_size: {kernel_size}\nsigma: {sigma}\n")
            elif blur_type == 'median':
                f.write(f"ksize: {ksize}\n")
            elif blur_type == 'box':
                f.write(f"ksize: {ksize}\n")
        print(f"Parameters saved at: {params_save_path}")

# Main function
def main():
    # Ask for the image path
    image_path = "../assets/hexagon_building.jpg"
    
    # Ask for the type of blur to apply
    blur_type = input("Enter blur type (gaussian, median, box, motion, bilateral, directional): ").lower()
    
    blur_image(image_path, blur_type)

if __name__ == "__main__":
    main()
