# utils/frame_sampler.py

class FrameSampler:
    """
    Controls how often expensive models (Qwen) are run
    """

    def __init__(self, every_n_frames=1):
        self.every_n_frames = every_n_frames
        self.counter = 0

    def should_sample(self):
        self.counter += 1
        if self.counter >= self.every_n_frames:
            self.counter = 0
            return True
        return False
