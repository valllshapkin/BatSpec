import cv2
import numpy as np
import MultiArray as ma
from BatSpec.Core.Functions import SpecFunc
from BatSpec.Core.Physical.Units import UREG

def _apply_cv2_morph(spec: SpecFunc, op: int, radius: int, shape: int = cv2.MORPH_ELLIPSE) -> SpecFunc:
    ctx = spec.context
    mat_np = ma.to_numpy(spec._matrx_a)
    
    # Для CV2 бинаризуем
    binary = (mat_np > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(shape, (2 * radius + 1, 2 * radius + 1))
    
    if op == -1: # Erode
        result = cv2.erode(binary, kernel)
    elif op == -2: # Dilate
        result = cv2.dilate(binary, kernel)
    else:
        result = cv2.morphologyEx(binary, op, kernel)
        
    # Восстанавливаем амплитуду
    max_val = np.max(mat_np)
    final_mat = (result * max_val).astype(mat_np.dtype)
    
    return SpecFunc(ma.convert_to(final_mat, ctx), spec._first_a, spec._sec___a, spec._matrx_u)

def morphErode(spec: SpecFunc, radius: int = 2) -> SpecFunc:
    return _apply_cv2_morph(spec, -1, radius)

def morphDilate(spec: SpecFunc, radius: int = 2) -> SpecFunc:
    return _apply_cv2_morph(spec, -2, radius)

def morphOpen(spec: SpecFunc, radius: int = 2) -> SpecFunc:
    return _apply_cv2_morph(spec, cv2.MORPH_OPEN, radius)

def morphClose(spec: SpecFunc, radius: int = 2) -> SpecFunc:
    return _apply_cv2_morph(spec, cv2.MORPH_CLOSE, radius)

def morphHorizontalConnect(spec: SpecFunc, width: int = 10) -> SpecFunc:
    ctx = spec.context
    mat_np = ma.to_numpy(spec._matrx_a)
    binary = (mat_np > 0).astype(np.uint8)
    # Прямоугольное ядро: высота 1, ширина width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    final_mat = (result * np.max(mat_np)).astype(mat_np.dtype)
    return SpecFunc(ma.convert_to(final_mat, ctx), spec._first_a, spec._sec___a, spec._matrx_u)

def morphVerticalConnect(spec: SpecFunc, height: int = 10) -> SpecFunc:
    ctx = spec.context
    mat_np = ma.to_numpy(spec._matrx_a)
    binary = (mat_np > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    final_mat = (result * np.max(mat_np)).astype(mat_np.dtype)
    return SpecFunc(ma.convert_to(final_mat, ctx), spec._first_a, spec._sec___a, spec._matrx_u)

def makeLabelsWatershed(spec: SpecFunc, threshold: float = 0.3, min_distance: int = 5) -> SpecFunc:
    """Маркерный водораздел (Watershed). Разделяет слипшиеся сигналы строго по седлу между пиками."""
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    ctx = spec.context
    mat_np = ma.to_numpy(spec._matrx_a).astype(np.float32)
    
    mask = mat_np >= threshold
    peaks_coords = peak_local_max(mat_np, min_distance=min_distance, labels=mask)
    
    if len(peaks_coords) == 0:
        return SpecFunc(ma.zeros_like(spec._matrx_a), spec._first_a, spec._sec___a, UREG.dimensionless)
        
    markers = np.zeros_like(mat_np, dtype=np.int32)
    for i, (r, c) in enumerate(peaks_coords):
        markers[r, c] = i + 1
        
    labels = watershed(-mat_np, markers, mask=mask).astype(np.int32)
    
    return SpecFunc(ma.convert_to(labels, ctx), spec._first_a, spec._sec___a, UREG.dimensionless)
