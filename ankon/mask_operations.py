import cv2
import numpy as np


# ==============================================================================
# FUNCTION 1: extract_white_mask
# ==============================================================================
def extract_white_mask(image_path: str) -> dict:
    """
    Reads an image and extracts the white masked object.

    How it works:
    - Loads the image in grayscale (white=255, black=0).
    - Applies a binary threshold: pixels >= 200 become white (255), rest black (0).
      This cleanly isolates the white blob from the black background.
    - Finds connected components (blobs) using connectedComponentsWithStats.
      Label 0 is always the background, so we skip it and find the largest
      remaining component — that's our white object.
    - Returns a dictionary containing:
        * 'binary_mask'    : uint8 array, same size as image, white object = 255
        * 'original_image' : the original grayscale image (used as base for output)
        * 'canvas_shape'   : (H, W) of the original image
        * 'bbox'           : (x, y, w, h) bounding box of the object
        * 'centroid'       : (cx, cy) center of mass of the object

    Parameters
    ----------
    image_path : str  →  path to the input image

    Returns
    -------
    dict with keys: binary_mask, original_image, canvas_shape, bbox, centroid
    """
    # --- Load image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    H, W = img.shape

    # --- Threshold: keep only near-white pixels ---
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # --- Find connected components ---
    # num_labels: total labels (including background=0)
    # stats: [x, y, w, h, area] per label
    # centroids: (cx, cy) per label
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels < 2:
        raise ValueError("No white object found in the image.")

    # Ignore label 0 (background). Find the largest foreground blob.
    foreground_areas = stats[1:, cv2.CC_STAT_AREA]   # areas of labels 1..N
    largest_label = int(np.argmax(foreground_areas)) + 1  # +1 to offset skip of 0

    # Build a clean mask containing only the largest white object
    clean_mask = np.zeros((H, W), dtype=np.uint8)
    clean_mask[labels == largest_label] = 255

    # Extract bounding box and centroid
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[largest_label]

    return {
        "binary_mask": clean_mask,
        "original_image": img,
        "canvas_shape": (H, W),
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
    }


# ==============================================================================
# FUNCTION 2: scale_mask
# ==============================================================================
def scale_mask(mask_info: dict, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Scales the white object and composites it onto the original image.

    How it works:
    - Starts with a copy of the original image as the base canvas.
    - Crops the object out using its bounding box.
    - Resizes (scales) the cropped region by (scale_x, scale_y).
      scale_x / scale_y > 1 → grow,  < 1 → shrink.
    - Erases the original object location from the canvas by filling it black.
    - Paints the scaled object onto the canvas at the same top-left anchor (x, y).
      Overflow beyond canvas boundaries is clipped automatically.

    Parameters
    ----------
    mask_info : dict   →  output of extract_white_mask()
    scale_x   : float  →  horizontal scale factor  (e.g. 1.5 = 150 % width)
    scale_y   : float  →  vertical   scale factor  (e.g. 0.5 = 50 % height)

    Returns
    -------
    np.ndarray  →  uint8 grayscale image with scaled object on original background
    """
    mask   = mask_info["binary_mask"]
    H, W   = mask_info["canvas_shape"]
    x, y, w, h = mask_info["bbox"]

    # --- Start from a copy of the original image ---
    canvas = mask_info["original_image"].copy()

    # --- Erase the original object from the canvas ---
    # Where the mask is white, set pixels to 0 (black) to remove the old object
    canvas[mask == 255] = 0

    # --- Crop the object region from the mask ---
    obj_crop = mask[y : y + h, x : x + w]

    # --- Compute new dimensions (at least 1 pixel) ---
    new_w = max(1, int(round(w * scale_x)))
    new_h = max(1, int(round(h * scale_y)))

    # --- Resize with INTER_NEAREST to keep it a clean binary mask ---
    scaled_obj = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- Paste scaled object onto the canvas ---
    # Destination region (clamped to canvas bounds)
    dst_x1, dst_y1 = x, y
    dst_x2 = min(W, x + new_w)
    dst_y2 = min(H, y + new_h)

    # Corresponding source region in scaled_obj
    src_x2 = dst_x2 - dst_x1
    src_y2 = dst_y2 - dst_y1

    # Only paint where the scaled mask is white (OR logic preserves background)
    roi = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
    src_roi = scaled_obj[0:src_y2, 0:src_x2]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.where(src_roi == 255, 255, roi)

    return canvas


# ==============================================================================
# FUNCTION 3: translate_mask
# ==============================================================================
def translate_mask(mask_info: dict, tx: int, ty: int) -> np.ndarray:
    """
    Translates (shifts) the white object and composites it onto the original image.

    How it works:
    - Starts with a copy of the original image as the base canvas.
    - Erases the original object from that copy (sets its pixels to black).
    - Uses cv2.warpAffine with a pure translation matrix on the binary mask:
          M = | 1  0  tx |
              | 0  1  ty |
      This shifts every white pixel by (tx, ty).
    - Paints the shifted mask onto the canvas using OR logic so the background
      is preserved everywhere except where the moved object now sits.
    - Positive tx → move right,  negative tx → move left.
    - Positive ty → move down,   negative ty → move up.
    - Pixels that shift outside the canvas boundaries are discarded.

    Parameters
    ----------
    mask_info : dict  →  output of extract_white_mask()
    tx        : int   →  horizontal translation in pixels
    ty        : int   →  vertical   translation in pixels

    Returns
    -------
    np.ndarray  →  uint8 grayscale image with translated object on original background
    """
    mask = mask_info["binary_mask"]
    H, W = mask_info["canvas_shape"]

    # --- Start from a copy of the original image ---
    canvas = mask_info["original_image"].copy()

    # --- Erase the original object from the canvas ---
    canvas[mask == 255] = 0

    # --- Build the 2×3 affine translation matrix ---
    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    # --- Shift the binary mask ---
    # BORDER_CONSTANT + borderValue=0 fills vacated areas with black
    shifted_mask = cv2.warpAffine(
        mask, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # --- Paint the shifted object onto the canvas ---
    # Where the shifted mask is white, overwrite canvas with 255
    canvas[shifted_mask == 255] = 255

    return canvas


# ==============================================================================
# DEMO / TEST  (edit the path and parameters as needed)
# ==============================================================================
if __name__ == "__main__":
    IMAGE_PATH = "t1.jpg"   # <-- change to your image path

    # --- Step 1: Extract the white mask ---
    mask_info = extract_white_mask(IMAGE_PATH)
    print(f"Canvas shape : {mask_info['canvas_shape']}")
    print(f"Bounding box : {mask_info['bbox']}")
    print(f"Centroid     : {mask_info['centroid']}")

    # Save the extracted mask
    cv2.imwrite("output_extracted_mask.png", mask_info["binary_mask"])
    print("Saved: output_extracted_mask.png")

    # --- Step 2: Scale (1.5x wide, 1.5x tall) ---
    scaled = scale_mask(mask_info, scale_x=.5, scale_y=.5)
    cv2.imwrite("output_scaled.png", scaled)
    print("Saved: output_scaled.png")

    # --- Step 3: Translate (50 px right, 30 px down) ---
    translated = translate_mask(mask_info, tx=50, ty=30)
    cv2.imwrite("output_translated.png", translated)
    print("Saved: output_translated.png")
