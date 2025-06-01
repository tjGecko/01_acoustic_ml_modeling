class AngleNormalizer:
    """Normalize angles to [-1, 1] range for training."""
    def __init__(self, az_range=(-90, 90), el_range=(-90, 90)):
        self.az_range = az_range
        self.el_range = el_range

    def normalize(self, az, el):
        """Normalize angles to [-1, 1] range."""
        az_norm = 2 * (az - self.az_range[0]) / (self.az_range[1] - self.az_range[0]) - 1
        el_norm = 2 * (el - self.el_range[0]) / (self.el_range[1] - self.el_range[0]) - 1
        return az_norm, el_norm

    def denormalize(self, az_norm, el_norm):
        """Convert normalized angles back to original range."""
        az = (az_norm + 1) * (self.az_range[1] - self.az_range[0]) / 2 + self.az_range[0]
        el = (el_norm + 1) * (self.el_range[1] - self.el_range[0]) / 2 + self.el_range[0]
        return az, el
