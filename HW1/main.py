import numpy as np
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args

def padding(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############

    H,W,C = input_img.shape
    pad_size = kernel_size // 2
    output_img = np.zeros((H+pad_size*2 , W+pad_size*2 , C) , dtype=np.float64)
    output_img[pad_size : pad_size+H , pad_size : pad_size+W] = input_img.copy().astype(np.float64)

    ############### YOUR CODE ENDS HERE #################

    return output_img

def convolution(input_img, padding_img , kernel ,kernel_size):
    ############### YOUR CODE STARTS HERE ###############
    H,W,C = input_img.shape
    temp = padding_img.copy()
    pad = kernel_size // 2

    for y in range(H):
        for x in range(W):
            for c in range(C):
                padding_img[pad+y , pad+x , c] = np.sum(kernel * temp[y:y+kernel_size , x:x+kernel_size , c])
    #padding_img = np.clip(padding_img,0,255)
    padding_img = padding_img[pad: pad + H, pad: pad + W].astype(np.uint8)

    ############### YOUR CODE ENDS HERE #################
    return padding_img
    
def gaussian_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############
    sigma = 1
    kernel_size = 3

    padding_img = padding(input_img , kernel_size)

    pad = kernel_size // 2
    kernel = np.zeros((kernel_size,kernel_size) , dtype=np.float64)
    for x in range(-pad , -pad+kernel_size):
        for y in range(-pad , -pad+kernel_size):
            kernel[y+pad , x+pad] = np.exp( -(x**2 + y**2) / (2*(sigma**2)))
    kernel /= (2*np.pi*sigma*sigma)
    kernel /= kernel.sum()
    
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, padding_img, kernel , kernel_size),sigma,kernel_size

def median_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############
    H,W,C = input_img.shape
    kernel_size = 3
    padding_img = padding(input_img , kernel_size)
    pad = kernel_size // 2

    output_img = np.zeros_like(input_img)
    for y in range(pad, H + pad):
        for x in range(pad, W + pad):
            for c in range(C):
                window = padding_img[y-pad:y+pad+1, x-pad:x+pad+1, c]
                median_value = np.median(window)
                output_img[y-pad,x-pad,c] = median_value

    ############### YOUR CODE ENDS HERE #################
    return output_img,kernel_size

def laplacian_sharpening(input_img):
    ############### YOUR CODE STARTS HERE ###############
    kernel_size = 3
    padding_img = padding(input_img , kernel_size)

    kernel = np.array([ [0 , -1, 0],
                        [-1, 5, -1],
                        [0 , -1, 0] ],dtype=np.float64)
    
    # kernel = np.array([ [-1, -1, -1],
    #                     [-1,  9, -1],
    #                     [-1, -1, -1]], dtype=np.float64)

    pad = kernel_size // 2

    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img,padding_img, kernel ,kernel_size)

if __name__ == "__main__":
    args = parse_args()
    
    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img ,sigma , kernel_size = gaussian_filter(input_img)
        cv2.imwrite(f"output_sigma={sigma}_kernel={kernel_size}.jpg", output_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img , kernel_size = median_filter(input_img)
        cv2.imwrite(f"output_kernel={kernel_size}.jpg", output_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)
        cv2.imwrite("output.jpg", output_img)

    
    