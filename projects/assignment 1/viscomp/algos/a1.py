# CSC320 Fall 2022
# Assignment 1
# (c) Kyros Kutulakos, Towaki Takikawa, Esther Lin
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

import numpy as np
import cv2
import viscomp.ops.image as img_ops


def run_a1_algo(source_image, destination_image, source_coords,
                destination_coords, homography=None):
    """Run the entire A1 algorithm.

    Args:
        source_image (np.ndarray): The source image of shape [Hs, Ws, 4]
        destination_image (np.ndarray): The destination image of shape [Hd, Wd, 4]
        source_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the source image.
        destination_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the destination image.
        homography (np.ndarray): (Optional) [3, 3] homography matrix. If passed in, will use this
                                 instead of calculating it.

    Returns:
        (np.ndarray): Written out image of shape [Hd, Wd, 4]
    """
    if homography is None:
        print("Calculating homography...")
        np.set_printoptions(formatter={'float': '{:.4f}'.format})
        homography = calculate_homography(destination_coords, source_coords)
    else:
        print("Using preset homography matrix...")
    print("")
    print("Homography matrix:")
    print(homography)
    print("")
    print("Performing backward mapping...")
    points = homography @ np.concatenate([destination_coords, np.ones([4, 1])],
                                         axis=-1).T
    backprojected_coords = (points.T)[:4, :2] / (points.T)[:4, 2:]
    print(source_coords)
    print(destination_coords)
    print(backprojected_coords)
    output_buffer = backward_mapping(homography, source_image,
                                     destination_image, destination_coords)
    print("Algorithm has succesfully finished running!")
    return output_buffer


def convex_polygon(poly_coords, image_coords):
    """From coords that define a convex hull, find which image coordinates are inside the hull.

     Args:
         poly_coords (np.ndarray): [N, 2] list of 2D coordinates that define a convex polygon.
                              Each nth index point is connected to the (n-1)th and (n+1)th
                              point, and the connectivity wraps around (i.e. the first and last
                              points are connected to each other)
         image_coords (np.ndarray): [H, W, 2] array of coordinates on the image. Using this,
                                 the goal is to find which of these coordinates are inside
                                 the convex hull of the polygon.
         Returns:
             (np.ndarray): [H, W] boolean mask where True means the coords is inside the hull.
     """
    mask = np.ones_like(image_coords[..., 0]).astype(np.bool)
    N = poly_coords.shape[0]
    for i in range(N):
        dv = poly_coords[(i + 1) % N] - poly_coords[i]
        winding = (image_coords - poly_coords[i][None]) * (
            np.flip(dv[None], axis=-1))
        winding = winding[..., 0] - winding[..., 1]
        mask = np.logical_and(mask, (winding > 0))
    return mask


# student_implementation

def calculate_homography(source, destination):
    """Calculate the homography matrix based on source and desination coordinates.

     Args:
         source (np.ndarray): [4, 2] matrix of 2D coordinates in the source image.
         destination (np.ndarray): [4, 2] matrix of 2D coordinates in the destination image.

     Returns:
         (np.ndarray): [3, 3] homography matrix.
    """
    A = np.zeros((8, 8))
    B = np.zeros(8)
    h = 0

    for i in range(4):
        for j in range(2):
            if j == 0:
                A[h, 0] = -source[i][0]
                A[h, 1] = -source[i][1]
                A[h, 2] = -1
                A[h, 6] = destination[i][0] * source[i][0]
                A[h, 7] = destination[i][0] * source[i][1]
                B[h] = -destination[i][0]
                h += 1

            else:
                A[h, 3] = -source[i][0]
                A[h, 4] = -source[i][1]
                A[h, 5] = -1
                A[h, 6] = destination[i][1] * source[i][0]
                A[h, 7] = destination[i][1] * source[i][1]
                B[h] = -destination[i][1]
                h += 1

    ans = np.linalg.solve(A, B)
    # ans = np.matmul(B,  np.linalg.inv(A))

    arr = np.array([[ans[0], ans[1], ans[2]],
                    [ans[3], ans[4], ans[5]],
                    [ans[6], ans[7], 1]])

    return arr


def backward_mapping(transform, source_image, destination_image,
                     destination_coords):
    """Perform backward mapping onto the destination image.

     The goal of this function is to map each destination image pixel which is within the polygon defined
     by destination_coords to a corresponding image pixel in source_image.

     Hints: Start by iterating through the destination image pixels using a nested for loop. For each pixel,
     use the convex_polygon function to find whether they are inside the polygon. If they are, figure out
     how to use the homography matrix to find the corresponding pixel in source_image.

     Args:
         transform (np.ndarray): [3, 3] homogeneous transformation matrix.
         source_image (np.ndarray): The source image of shape [Hs, Ws, 4]
         destination_image (np.ndarray): The destination image of shape [Hd, Wd, 4]
         source_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the source image.
         destination_coords (np.ndarray): [4, 2] matrix of normalized 2D coordinates in the destination image.

     Returns:
         (np.ndarray): [Hd, Wd, 4] image with the source image projected onto the destination image.
     """
    for i in range(4):
        destination_coords[i][0] = (destination_coords[i][0] + 1) / 2 * \
                                   destination_image.shape[1]
        destination_coords[i][1] = (destination_coords[i][1] + 1) / 2 * \
                                   destination_image.shape[0]
    # print((destination_coords))
    h, w = destination_image.shape[:2]
    for x in range(w):
        for y in range(h):
            coord_int = np.array([x, y])
            if convex_polygon(destination_coords, coord_int):
                coord = img_ops.normalize_coordinates(coord_int, h, w)
                coord[1] *= -1
                newxy = np.dot(transform, np.array([*coord, 1]))
                newxy = newxy[:2] / newxy[2]
                newxy[1] *= -1
            # print(newxy)
                newx = round(((newxy[0] + 1) / 2) * w)

                newy = round(((newxy[1] + 1) / 2) * h)
            #
            # print([newx, newy])
            # print(convex_polygon(destination_coords, np.array([newx, newy])))

                coord[1] *= -1

            # if 0 <= newx < destination_image.shape[1] and newy >= 0 \
            #         and newy < destination_image.shape[0]:

                destination_image[y][x] = source_image[newy][newx]
    #                 print(destination_image[newy][newx])
    #
    # print(destination_image)
    output_buffer = destination_image
    return output_buffer
