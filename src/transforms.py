from torchvision.transforms import v2
from random import uniform

class RandomBrightness(object):
    """Adjusts the image brightness based on random values.

    Args:
        brightness_factor (float or tuple): Desired brightness adjustment. If float, 
            output is scaled according to the brightness factor. If tuple, brightness
            factor will be randomly sampled between the values.
        contrast_factor (float or tuple): Desired contrast adjustment. If float, 
            output is scaled according to the contrast factor. If tuple, contrast
            factor will be randomly sampled between the values.
        gamma (float or tuple): Desired gamma adjustment. If float, 
            output is scaled according to the gamma value. If tuple, gamma value
            will be randomly sampled between the values.
    """
    def __init__(self, brightness_factor=1., contrast_factor=1., gamma=1.):
        assert isinstance(brightness_factor, (float, tuple))
        assert isinstance(contrast_factor, (float, tuple))
        assert isinstance(gamma, (float, tuple))

        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.gamma = gamma

    def __call__(self, sample):
        # Function call
        if (isinstance(self.brightness_factor, float)):
            img = v2.functional.adjust_brightness(sample, brightness_factor=self.brightness_factor)
        else:
            rand_val = uniform(self.brightness_factor[0], self.brightness_factor[1])
            img = v2.functional.adjust_brightness(sample, brightness_factor=rand_val)

        if (isinstance(self.contrast_factor, float)):
            img = v2.functional.adjust_contrast(img, contrast_factor=self.contrast_factor)
        else:
            rand_val = uniform(self.contrast_factor[0], self.contrast_factor[1])
            img = v2.functional.adjust_contrast(sample, contrast_factor=rand_val)

        if(isinstance(self.gamma, float)):
            img = v2.functional.adjust_gamma(img, gamma=self.gamma)
        else:
            rand_val = uniform(self.gamma[0], self.gamma[1])
            img = v2.functional.adjust_gamma(img, gamma=rand_val)

        return img

        