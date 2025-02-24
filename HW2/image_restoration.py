import cv2
import numpy as np


"""
TODO Part 1: Motion blur PSF generation
"""
def generate_motion_blur_psf(length,angle):

    if length % 2 == 0:
        length += 1

    psf = np.zeros((length, length))
    center = length // 2

    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    for i in range(-center, center + 1):
        x = int(round(i * cos_theta))
        y = int(round(i * sin_theta))
        
        if abs(x) <= center and abs(y) <= center:
            psf[center + y, center + x] = 1
    
    psf /= psf.sum() 
    
    return psf

    raise NotImplementedError


"""
TODO Part 2: Wiener filtering
"""
def wiener_filtering(img_blurred, psf, K):

    img_blurred_float = img_blurred.astype(np.float32) / 255.0
    
    h, w = img_blurred_float.shape[:2]
    
    psf_pad = np.zeros((h, w))
    psf_h, psf_w = psf.shape
    psf_pad[(h-psf_h)//2:(h+psf_h)//2, (w-psf_w)//2:(w+psf_w)//2] = psf

    H = np.fft.fft2(np.fft.fftshift(psf_pad))
    
    wiener_img = np.zeros_like(img_blurred_float)
    
    for channel in range(3):
        G = np.fft.fft2(img_blurred_float[:,:,channel])
        
        H_mag_sqr = np.abs(H) ** 2
        F_hat = (1/H) * (H_mag_sqr/(H_mag_sqr + K)) * G

        img_deconv = np.real(np.fft.ifft2(F_hat))

        wiener_img[:,:,channel] = img_deconv

    wiener_img = np.clip(wiener_img * 255, 0, 255).astype(np.uint8)
    
    return wiener_img

    raise NotImplementedError


"""
TODO Part 3: Constrained least squares filtering
"""
def constrained_least_square_filtering(blurred_img, psf, gamma=0.1):
   
    laplacian = np.zeros(blurred_img.shape[:2])
    laplacian[0, 0] = 4
    laplacian[0, 1] = laplacian[1, 0] = laplacian[0, -1] = laplacian[-1, 0] = -1
    laplacian_fft = np.fft.fft2(laplacian)

    restored_channels = []
    for channel in cv2.split(blurred_img):
        blurred_img_fft = np.fft.fft2(channel)
        psf_fft = np.fft.fft2(psf, s=channel.shape)

        H_conj = np.conj(psf_fft)
        denominator = H_conj * psf_fft + gamma * laplacian_fft
        restored_img_fft = H_conj * blurred_img_fft / denominator

        restored_channel = np.fft.ifft2(restored_img_fft).real
        restored_channels.append(np.clip(restored_channel, 0, 255).astype(np.uint8))

    return cv2.merge(restored_channels)


"""
Bouns
"""
def other_restoration_algorithm():
    raise NotImplementedError


def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr


"""
Main function
"""
def main():
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))

        # TODO Part 1: Motion blur PSF generation
        psf = generate_motion_blur_psf(length=41 , angle=-45)

        # TODO Part 2: Wiener filtering
        if(i==0):
            wiener_img = wiener_filtering(img_blurred, psf, K=0.008)
        else :
            wiener_img = wiener_filtering(img_blurred, psf, K=0.01)

        # TODO Part 3: Constrained least squares filtering
        #task 1 , Gamma = 0.9  . task 2 , Gamma = 0.9
        if(i == 0):
            constrained_least_square_img = constrained_least_square_filtering(img_blurred, psf, gamma=0.9)
        else:
            constrained_least_square_img = constrained_least_square_filtering(img_blurred, psf, gamma=0.9)
        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, wiener_img)))

        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, constrained_least_square_img)))#

        cv2.imshow("window of Wiener", np.hstack([img_blurred, wiener_img]))
        cv2.imshow("window of constrained", np.hstack([img_blurred, constrained_least_square_img]))
        #cv2.imshow("window", np.hstack([img_blurred, wiener_img, constrained_least_square_img]))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
