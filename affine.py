import numpy as np

#Two separate jobs:
#solve_affine: given 3 point pairs, solve for the 2x3 affine matrix M
#  warp_affine: given M, actually move the pixels from src into a dst sized image
#https://learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/


def solve_affine(tri_src, tri_dst):
    #An affine transform maps any point (x, y) to a new point (x', y') using 6 numbers total:
    #x' = a*x + b*y + c
    #y' = d*x + e*y + f
    #Use to define M = [[a, b, c], [d, e, f]]
    #Can find coeff using 3 vertices of 2 respective traingles

    #Build the coeff matrix A. Each row is one vertex, written as [x, y, 1]
    #np.linalg.solve below needs A invertible, aka tri_src cant be 3 collinear points
    #or a zero area traingle - delunay should factor for that.
    ones_column = np.ones((3, 1))
    A = np.hstack([tri_src, ones_column])

    #Solve for [a, b, c] st A @ [a, b, c] = (the 3 destination x' values).
    dst_x_values = tri_dst[:, 0]
    x_coeffs = np.linalg.solve(A, dst_x_values)

    #Same idea again, separately, for [d, e, f] using the destination y'
    #doesnt interact with abc so they are diff
    dst_y_values = tri_dst[:, 1]
    y_coeffs = np.linalg.solve(A, dst_y_values)

    #Copy output of cv2.getAffineTransform
    M = np.vstack([x_coeffs, y_coeffs])
    return M


def warp_affine(src, M, dsize):
    #M tells us how a point in src maps FORWARD into dst: dst_point = M @ src_point.
    #But forward-mapping pixel by pixel breaks on a discrete grid: if M magnifies,
    #some dst pixels never get hit by any src pixel at all. If M shrinks, multiple src pixels 
    #round onto the same dst pixel (collisions). So instead we go backward: for every 
    #dst pixel ask where it came from in src.

    #Need to Invert M
    #Use homogenous matrix. Just tag on [0, 0, 1] and represent every point as (x, y, 1)
    #Thus 1 = 0x + 0y + 1, so trivially true.
    #Typical for affine tranformations
    #The M_homogenous determinant is not zero iff tri_src or tri_dst is 3 collinear points/a zero-area triangle
    #That is given by properties of delaunay triangulation.
    extra_row = np.array([[0, 0, 1]])
    M_homogeneous = np.vstack([M, extra_row])
    M_inv = np.linalg.inv(M_homogeneous)

    dst_w, dst_h = dsize
    src_h, src_w = src.shape[:2]

    #Build a list of every single (x, y) coordinate in the destination image.
    #np.meshgrid gives us two grids: one where each entry is that pixel's x,
    #and one where each entry is that pixel's y.
    y_grid, x_grid = np.meshgrid(np.arange(dst_h), np.arange(dst_w), indexing="ij")

    #Flatten those grids and stack them into homogeneous coordinates, so we
    #end up with one big array of shape (3, num_pixels): row0 = all the x's,
    #row1 = all the y's, row2 = all 1's (for homogenous condition and math).
    ones_row = np.ones(x_grid.size)
    dst_coords = np.stack([x_grid.ravel(), y_grid.ravel(), ones_row])

    #Doing it once with dst_coords.
    src_coords = M_inv @ dst_coords

    #Pull the x's and y's back out and reshape them back into a grid matching the destination image's height/width, so src_x[row, col] 
    #is the source x-coordinate that destination pixel (row, col) should be sampled from
    src_x = src_coords[0].reshape(dst_h, dst_w)
    src_y = src_coords[1].reshape(dst_h, dst_w)

    #Since they are not whole numbers, use bilinear interpolation.
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1

    #Do bounds checking to ensure it falls within triangle/bounding box area.
    x0 = np.clip(x0, 0, src_w - 1)
    x1 = np.clip(x1, 0, src_w - 1)
    y0 = np.clip(y0, 0, src_h - 1)
    y1 = np.clip(y1, 0, src_h - 1)

    #How far off the true point is from its floor, aka the blend weight.
    #eg 42.7 -> frac_x = 0.7, so mostly weighted toward x1 not x0.
    #For unit square.
    frac_x = src_x - np.floor(src_x)
    frac_y = src_y - np.floor(src_y)
    wx = np.clip(frac_x, 0, 1)
    wy = np.clip(frac_y, 0, 1)

    #Add a channel axis so this broadcasts against the 3 (B,G,R) values per pixel in src.
    wx = wx[..., None]
    wy = wy[..., None]

    #4 neighbor pixels: (x0,y0)(x1,y0) top row, (x0,y1)(x1,y1) bottom row.
    #Blend left/right first, top row and bottom row separately.
    #Bilinear interp https://en.wikipedia.org/wiki/Bilinear_interpolation
    top_row = src[y0, x0] * (1 - wx) + src[y0, x1] * wx
    bottom_row = src[y1, x0] * (1 - wx) + src[y1, x1] * wx

    #Then blend those two together, up/down this time, for the final color.
    warped = top_row * (1 - wy) + bottom_row * wy

    #Cast back down to uint8 since all the blending above made it floats.
    return warped.astype(src.dtype)