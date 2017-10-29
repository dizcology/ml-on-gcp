from gym.envs import atari
import numpy as np

class AtariEnvOverlay(atari.AtariEnv):
    def __init__(self, *args, **kwargs):
        print('Loading AtariEnv with overlay')
        super(AtariEnvOverlay, self).__init__(*args, **kwargs)

        self.overlay = np.zeros(210 * 160 * 3).astype(np.uint8).reshape(210, 160, 3)


    def set_overlay(self, data=None):
        self.overlay = np.zeros(210 * 160 * 3).astype(np.uint8).reshape(210, 160, 3)

        if data is None:
            return

        # Note: data.shape = (1, 6400)
        cutoff = np.percentile(data, 95, axis=1)
        data = data.reshape(80, 80)

        data[data < cutoff] = 0
        data[data >= cutoff] = 50

        #data = data.astype(np.uint8)

        inflated = np.kron(data, np.ones((2, 2), dtype=np.uint8))
        
        self.overlay[35:195, :, 0] = inflated


    def _get_image(self):
        # shape: (210, 160, 3)
        # image[35:195] has shape (160, 160, 3)
        image =  self.ale.getScreenRGB2()

        # overlay here
        image += self.overlay

        return image

