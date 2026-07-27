import os
import cv2

from label import detector


def _face_center(img):
    #find the face so we can crop around it instead of just the image's center
    faces = detector(img)
    height, width = img.shape[:2]
    if not faces:
        #no face found, fall back to just the middle of the image
        return width // 2, height // 2
    face = faces[0]
    return (face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2


def crop_to_aspect(img, target_ratio=1.0):
    height, width = img.shape[:2]
    cx, cy = _face_center(img)

    #wider than target ratio -> chop off the sides, keeping the face centered
    if width / height > target_ratio:
        new_width = int(height * target_ratio)
        x0 = min(max(cx - new_width // 2, 0), width - new_width)
        return img[:, x0:x0 + new_width]

    #taller than target ratio -> same deal, chop top/bottom instead
    new_height = int(width / target_ratio)
    y0 = min(max(cy - new_height // 2, 0), height - new_height)
    return img[y0:y0 + new_height, :]


def crop_images(src_dir, dst_dir, target_ratio=1.0):
    os.makedirs(dst_dir, exist_ok=True)

    src_filenames = set(os.listdir(src_dir))

    #wipe out any leftover crop whose source image got deleted (otherwise dst_dir
    #always matches whats currently in src_dir instead of accumulating stale faces
    for filename in os.listdir(dst_dir):
        if filename not in src_filenames:
            os.remove(os.path.join(dst_dir, filename))

    for filename in src_filenames:
        img = cv2.imread(os.path.join(src_dir, filename))
        if img is None:
            continue
        cropped = crop_to_aspect(img, target_ratio)
        cv2.imwrite(os.path.join(dst_dir, filename), cropped)