import os

import cv2
import numpy as np
from scipy.spatial import Delaunay

from affine import solve_affine, warp_affine
from crop import crop_images
from label import label_faces

CROPPED_DIR = "assets_cropped"


def normalize_coords(points, img):
    #Divide by this image's own width/height so points from different
    #resolution photos land in same [0,1] space so they can be averaged
    height, width = img.shape[:2]
    return points / [width, height]


def triangulate(points):
    #Triangles as index triples into points so the same triangulation
    #can be reapplied to any other point set with matching point order
    return Delaunay(points).simplices


def warp_triangle(img_src, img_dst, tri_src, tri_dst):
    #tri_src/tri_dst are the same traingle (same 3 indices), just read out of
    #two different point sets, warp src's pixels so they land inside tri_dst
    tri_src = tri_src.astype(np.float32)
    tri_dst = tri_dst.astype(np.float32)

    #getAffineTransform/warpAffine only work on rectangles, so crop down to the
    #smallest rect containing each triangle instead of touching the full image
    #boundingRect returns (x, y, w, h) top-left corner + size, not the triangle itself
    rect_src = cv2.boundingRect(tri_src)
    rect_dst = cv2.boundingRect(tri_dst)

    src_x, src_y, src_w, src_h = rect_src
    dst_x, dst_y, dst_w, dst_h = rect_dst

    #shift the vertices so theyre relative to their own rect's top-left corner
    #since thats the coord space the crop below lives in
    #float32 - list upcasts to float64, and getAffineTransform wants float32 so recast
    tri_src_rect = (tri_src - [src_x, src_y]).astype(np.float32)
    tri_dst_rect = (tri_dst - [dst_x, dst_y]).astype(np.float32)

    src_crop = img_src[src_y:src_y + src_h, src_x:src_x + src_w]

    #find M st tri_src_rect @ M = tri_dst_rect (only need 3 pts
    M = cv2.getAffineTransform(tri_src_rect, tri_dst_rect)

    #apply M to every pixel in src_crop, output sized to fit the dst rect
    warped = cv2.warpAffine(
        src_crop,
        M,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    #warped fills the whole rect, but we only want the triangle part of it,
    #the rest belongs to whatever triangle sits next door so mask that out
    mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    tri_dst_rect_int = tri_dst_rect.astype(np.int32)
    cv2.fillConvexPoly(mask, tri_dst_rect_int, 255)

    #paste just the masked in pixels into img_dst at the rect's location
    dst_slice = img_dst[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w]
    dst_slice[mask == 255] = warped[mask == 255]


def warp_triangle_scratch(img_src, img_dst, tri_src, tri_dst):
    #same as warp_triangle above, but using our own solve_affine/warp_affine from affine.py instead of cv2.getAffineTransform/cv2.warpAffine
    tri_src = tri_src.astype(np.float32)
    tri_dst = tri_dst.astype(np.float32)

    #boundingRect still needs float32/int32 specifically, cv2 thing not ours
    rect_src = cv2.boundingRect(tri_src)
    rect_dst = cv2.boundingRect(tri_dst)

    src_x, src_y, src_w, src_h = rect_src
    dst_x, dst_y, dst_w, dst_h = rect_dst

    #our solve_affine/warp_affine dont care about float32 specifically, so no
    #need to recast back down to float32 after this subtraction like the cv2
    #version has to
    tri_src_rect = tri_src - [src_x, src_y]
    tri_dst_rect = tri_dst - [dst_x, dst_y]

    src_crop = img_src[src_y:src_y + src_h, src_x:src_x + src_w]

    M = solve_affine(tri_src_rect, tri_dst_rect)
    warped = warp_affine(src_crop, M, (dst_w, dst_h))

    mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    tri_dst_rect_int = tri_dst_rect.astype(np.int32)
    cv2.fillConvexPoly(mask, tri_dst_rect_int, 255)

    dst_slice = img_dst[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w]
    dst_slice[mask == 255] = warped[mask == 255]


def warp_to_shape(img, points, target_points, triangles, canvas_size):
    #blank canvas sized to the midway shape, not this image's own size
    #everyone needs to land in the same coords so we can average them after
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    #same triangle indices work on any point set, so grab this face's 3 pts
    #and the matching 3 pts in the target shape, warp that one piece over
    for tri in triangles:
        warp_triangle_scratch(img, canvas, points[tri], target_points[tri])

    return canvas


def render_points(img, points, triangles, window_name):
    canvas = img.copy()

    for tri in triangles:
        cv2.polylines(canvas, [points[tri]], isClosed=True, color=(0, 255, 0), thickness=2)

    for i, (x, y) in enumerate(points):
        cv2.circle(canvas, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(canvas, str(i), (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_images(image_dir):
    #list out whats in there so we can pick by number instead of typing filenames
    filenames = sorted(os.listdir(image_dir))

    print(f"Images in {image_dir}:")
    for i, filename in enumerate(filenames, start=1):
        print(f"  {i}) {filename}")

    selection = input("Select images to merge (e.g. 1,3), or press enter for all: ")

    #empty input just means "gimme everything"
    if not selection.strip():
        return filenames

    picked_indices = [int(piece.strip()) for piece in selection.split(",")]
    return [filenames[i - 1] for i in picked_indices]


def main():
    crop_images("assets", CROPPED_DIR)

    selected_filenames = select_images(CROPPED_DIR)

    #for easier access
    images_data = []
    cur_points = []
    for filename in selected_filenames:
        image_path = os.path.join(CROPPED_DIR, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue

        all_points = label_faces(image_path)
        images_data.append((filename, img, all_points))
        for points in all_points:
            cur_points.append(normalize_coords(points, img))

    #Average every normalized point set together to get the midway shape
    midway_shape = np.mean(cur_points, axis=0)
    triangles = triangulate(midway_shape)

    #Same triangle pattern, drawn on each original face's own point positions
    for filename, img, all_points in images_data:
        for points in all_points:
            render_points(img, points, triangles, filename)

    #Cropping makes every image square, so any of them's side length works as the canvas size
    canvas_size = images_data[0][1].shape[0]
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    #Denormalized, to new canvas
    midway_pixels = (midway_shape * canvas_size).astype(int)
    render_points(canvas, midway_pixels, triangles, "Midway Shape")

    #warp every face onto the midway shape, then average all of em together
    warped_faces = []
    for filename, img, all_points in images_data:
        for points in all_points:
            warped = warp_to_shape(img, points, midway_pixels, triangles, canvas_size)
            warped_faces.append(warped)

            #show this one face already morphed onto the midway shape, before
            #it gets blended with the others
            cv2.imshow(f"Warped: {filename}", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    average_face = np.mean(warped_faces, axis=0).astype(np.uint8)
    cv2.imshow("Average Face", average_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()