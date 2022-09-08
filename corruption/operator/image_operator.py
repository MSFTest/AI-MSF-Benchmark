''''
@Project: fusion_rewrite   
@Description: Please add Description       
@Time:2022/9/6 13:31       
@Author:NianGao    
 
'''

import numpy as np
import skimage as sk
import cv2
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary


class ImageOperator(object):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    @staticmethod
    def operator_map():
        operator_map = {
            "loss_partial": ImageOperator.loss_partial,
            "distortion": ImageOperator.distortion,
            "loss_complete": ImageOperator.loss_complete,
            "motion_blur": ImageOperator.motion_blur,
            "defocus_blur": ImageOperator.defocus_blur,
            "gaussian_noise": ImageOperator.gaussian_noise,
            "impulse_noise": ImageOperator.impulse_noise,
            "brightness": ImageOperator.brightness,
            "darkness": ImageOperator.darkness,
        }
        return operator_map

    @staticmethod
    def brightness(image, severity=1):
        x = image.copy()
        c = [.1, .2, .3, .4, .5][severity - 1]
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return np.uint8(np.clip(x, 0, 1) * 255)

    @staticmethod
    def darkness(image, severity=1):
        x = image.copy()
        c = [.1, .2, .3, .4, .5][severity - 1]
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] - c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return np.uint8(np.clip(x, 0, 1) * 255)

    @staticmethod
    def gaussian_noise(image, severity=1):
        x = image.copy()
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        x = np.array(x) / 255.
        return np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255)

    @staticmethod
    def impulse_noise(image, severity=1):
        x = image.copy()
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]
        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.uint8(np.clip(x, 0, 1) * 255)

    @staticmethod
    def defocus_blur(image, severity=1):

        def disk(radius, alias_blur=0.1, dtype=np.float32):
            if radius <= 8:
                L = np.arange(-8, 8 + 1)
                ksize = (3, 3)
            else:
                L = np.arange(-radius, radius + 1)
                ksize = (5, 5)
            X, Y = np.meshgrid(L, L)
            aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
            aliased_disk /= np.sum(aliased_disk)

            # supersample disk to antialias
            return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

        x = image.copy()
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return np.uint8(np.clip(channels, 0, 1) * 255)

    @staticmethod
    def motion_blur(image, severity=1):
        from PIL import Image
        class MotionImage(WandImage):
            def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
                wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

        print(image.size)
        # x = image.copy()
        x = Image.fromarray(image)
        w, h = x.size
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())
        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)
        if x.shape != (h, w):
            return np.uint8(np.clip(x[..., [2, 1, 0]], 0, 255))  # BGR to RGB
        else:  # greyscale to RGB
            return np.uint8(np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255))

    @staticmethod
    def loss_partial(image, severity=1):
        params = [0.1, 0.25, 0.5, 0.75, 0.9][severity - 1]
        img_shape = image.shape
        arr = image.copy().flatten()
        loss_num = int(len(arr) * params)
        index = np.random.permutation(len(arr))[:loss_num]
        arr[index] = 0
        image = arr.reshape(img_shape).astype("uint8")
        return image

    @staticmethod
    def loss_complete(image=None, severity=1):
        image = np.zeros_like(image)
        return image.astype("uint8")

    @staticmethod
    def distortion(image, severity=1):
        params = [(0.5, 0.5), (0.75, 0.75), (1, 1), (1.25, 1.25), (1.5, 1.5)][severity - 1]
        import albumentations as A
        return A.OpticalDistortion(distort_limit=params[0], shift_limit=params[1], p=1)(image=image)["image"]
