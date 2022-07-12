from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import  numpy as np

def compute_psnr(img_batch1, img_batch2):
    p_total = 0
    p_num = img_batch1.size(0)
    img_batch1 = img_batch1.cpu().detach().numpy()
    img_batch2 = img_batch2.cpu().detach().numpy()

    for i in range(p_num):
        img1, img2 = img_batch1[i], img_batch2[i]
        img1 = img1.transpose((1,2,0))
        img2 = img2.transpose((1,2,0))
        img1 = np.resize(img1, img2.shape)
        p = psnr(img2, img1)
        p_total += p
    return p_total / p_num


def compute_ssim(img_batch1, img_batch2):
    s_total = 0
    s_num = img_batch1.size(0)
    img_batch1 = img_batch1.cpu().detach().numpy()
    img_batch2 = img_batch2.cpu().detach().numpy()

    for i in range(s_num):
        img1, img2 = img_batch1[i], img_batch2[i]
        img1 = np.resize(img1, img2.shape)
        img1 = img1.transpose((1,2,0))
        img2 = img2.transpose((1,2,0))
        isRGB = len(img1.shape) == 3 and img1.shape[-1] == 3
        s = ssim(img1, img2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                 multichannel=isRGB)
        s_total += s
    return s_total / s_num