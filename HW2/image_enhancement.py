import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
TODO Part 1: Gamma correction
"""
def gamma_correction(gamma, img):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    img_corrected = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3): 
                img_corrected[i, j, k] = table[img[i, j, k]]
    
    return img_corrected
"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]
    histo,_ = np.histogram(y_channel.flatten(), 256, [0, 256])

    cdf = histo.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')

    y_equalized = cdf_normalized[y_channel]
    img_yuv[:, :, 0] = y_equalized

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) , y_channel , y_equalized

    raise NotImplementedError

def plot_histograms(original_img, equalized_img):
    original_y = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    equalized_y = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(original_y.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title("Original Histogram")

    plt.subplot(1, 2, 2)
    plt.hist(equalized_y.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title("Equalized Histogram")

    plt.show()
"""
Bonus
"""
def other_enhancement_algorithm():
    raise NotImplementedError


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # TODO: modify the hyperparameter
    gamma_list = [0.6, 1.0, 2.5] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(gamma , img)

        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.waitKey(0)

    #TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img,y_channel ,y_equalized = histogram_equalization(img)

    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)
    plot_histograms(img, histogram_equalization_img)


if __name__ == "__main__":
    main()
