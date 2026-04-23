import numpy as np
import scipy.ndimage as ndimage
from BatSpec.QtUp.PolyROI import ROIData

class ROIProcessor:
    """Класс для математической обработки выделенных областей (без UI)."""
    def __init__(self, full_data: np.ndarray) -> None:
        self.full_data = full_data

    def extract_crop(self, roi: ROIData, padding: int = 0) -> np.ndarray:
        max_y, max_x = self.full_data.shape
        x1 = max(0, int(roi.x) - padding)
        y1 = max(0, int(roi.y) - padding)
        x2 = min(max_x, int(roi.x + roi.w) + padding)
        y2 = min(max_y, int(roi.y + roi.h) + padding)
        return self.full_data[y1:y2, x1:x2].copy()

    def apply_gaussian_blur(self, crop: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        return ndimage.gaussian_filter(crop, sigma=sigma)

    def apply_laplacian(self, crop: np.ndarray) -> np.ndarray:
        return np.abs(ndimage.laplace(crop))
    
