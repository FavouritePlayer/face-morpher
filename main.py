import os

import cv2

from label import label_faces


def main():
    for filename in os.listdir("assets"):
        image_path = os.path.join("assets", filename)
        img = cv2.imread(image_path)
        if img is None:
            continue

        all_points = label_faces(image_path)
        for points in all_points:
            for (x, y) in points:
                cv2.circle(img, (x, y), 10, (0, 0, 255), -1)

        cv2.imshow(os.path.basename(image_path), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()