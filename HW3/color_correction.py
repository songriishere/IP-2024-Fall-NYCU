import cv2
import numpy as np
import os


"""
TODO White patch algorithm
"""
def white_patch_algorithm(img):
    img = img.astype(np.float32)  
    max_vals = img.max(axis=(0, 1))  
    white_patch_img = (img / max_vals) * 255  
    white_patch_img = np.clip(white_patch_img, 0, 255)  
    return white_patch_img.astype(np.uint8)
    raise NotImplementedError


"""
TODO Gray-world algorithm
"""
def gray_world_algorithm(img):
    img = img.astype(np.float32)  
    mean_vals = img.mean(axis=(0, 1))  
    mean_gray = mean_vals.mean()  
    scale_factors = mean_gray / mean_vals  
    gray_world_img = img * scale_factors 
    gray_world_img = np.clip(gray_world_img, 0, 255)  
    return gray_world_img.astype(np.uint8)    
    raise NotImplementedError


"""
Bonus 
"""
def other_white_balance_algorithm():
    raise NotImplementedError


"""
Main function
"""
def main():

    os.makedirs("result/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread("data/color_correction/input{}.bmp".format(i + 1))

        # TODO White-balance algorithm
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)

        cv2.imwrite("result/color_correction/white_patch_input{}.bmp".format(i + 1), white_patch_img)
        cv2.imwrite("result/color_correction/gray_world_input{}.bmp".format(i + 1), gray_world_img)

if __name__ == "__main__":
    main()