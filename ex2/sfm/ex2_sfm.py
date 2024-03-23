import cv2
import json

import numpy as np
import matplotlib.pyplot as plt


###########################
##### Helper Function #####
###########################
def get_keypoints(img1, img2):
    '''
    Finds correspondence points between two images by searching for salient
    points via SIFT and then searching for matches via k-nearest neighours.
    Performs a ratio test to filter for outliers.

    Args:
        img1 (np.ndarray): image, first view
        img2 (np.ndarray): image, second view
    Returns:
        p_source(np.ndarray): Nx3 array of correspondence points in first view
            in homogenous image coordinates.
        p_source(np.ndarray): Nx3 array of correspondence points in second view
            in homogenous image coordinates.
    '''
    # Initialize feature description algorithm
    # We are going to use SIFT but OpenCV provides
    # implementations for many more - feel free to try them!
    descriptor = cv2.SIFT_create(nfeatures=10000)

    keypoints1, features1 = descriptor.detectAndCompute(img1, None)
    keypoints2, features2 = descriptor.detectAndCompute(img2, None)

    # Initialize matching algorithm
    # We are going to use k-nearest neighbours with L2 as the distance measure
    bf = cv2.BFMatcher_create(cv2.NORM_L2)

    # Find matching points
    matches = bf.knnMatch(features1, features2, k=2)

    # Remove outliers via ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    # Select matched points
    keypoints1 = np.float32(
        [keypoints1[good_match.queryIdx].pt for good_match in good]
    ).reshape(-1, 2)
    keypoints2 = np.float32(
        [keypoints2[good_match.trainIdx].pt for good_match in good]
    ).reshape(-1, 2)

    # Augment point vectors
    N = keypoints1.shape[0]
    keypoints1 = np.concatenate([keypoints1, np.ones((N, 1))], axis=-1)
    keypoints2 = np.concatenate([keypoints2, np.ones((N, 1))], axis=-1)

    return keypoints1, keypoints2


def draw_matches(img1, img2, keypoints1, keypoints2):
    '''
    Returns a visualization of correspondences accross two images.

    Args:
        img1 (np.ndarray): image, first view
        img2 (np.ndarray): image, second view
        keypoints1 (np.ndarray): Nx3 array of correspondence points in first
            view in homogenous image coordinates.
        keypoints2 (np.ndarray): Nx3 array of correspondence points in second
            view in homogenous image coordinates.

    Returns:
        output_img (np.ndarray): image, horizontally stacked first and sedond
        view with correspondence points overlaid.
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # stack views horizontally
    output_img = np.zeros((max([h1, h2]), w1 + w2, 3), dtype='uint8')
    output_img[:h1, :w1, :] = np.dstack([img1])
    output_img[:h2, w1:w1 + w2, :] = np.dstack([img2])

    # draw correspondences
    # we only visualize only a subset for clarity
    for p1, p2 in zip(keypoints1[::4], keypoints2[::4]):
        (x1, y1) = p1[:2]
        (x2, y2) = p2[:2]

        cv2.circle(output_img, (int(x1), int(y1)), 10, (0, 255, 255), 10)
        cv2.circle(output_img, (int(x2) + w1, int(y2)), 10, (0, 255, 255), 10)

        cv2.line(output_img, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (0, 255, 255), 5)

    return output_img


def draw_epipolar_lines(img1, img2, els1, els2, kps1, kps2):
    '''
    Returns an image with epipolar lines drawn onto the images.

    Args:
        img1 (np.ndarray): image, first view
        img2 (np.ndarray): image, second view
        els1 (np.ndarray): Nx3 array of epipolar lines in the first view
            induced by correspondences in the second view.
        els2 (np.ndarray): Nx3 array of epipolar lines in the second view
            induced by correspondences in the first view.
        keypoints1 (np.ndarray): Nx3 array of correspondence points in first
            view in homogenous image coordinates.
        keypoints2 (np.ndarray): Nx3 array of correspondence points in second
            view in homogenous image coordinates.

    Returns:
        output_img (np.ndarray): image, horizontally stacked first and sedond
        view with correspondence points overlaid.
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    output_img = np.zeros((max([h1, h2]), w1 + w2, 3), dtype='uint8')
    output_img[:h1, :w1, :] = np.dstack([img1])
    output_img[:h2, w1:w1 + w2, :] = np.dstack([img2])

    for p1, el1, p2, el2 in zip(kps1, els1, kps2, els2):
        (x1, y1) = p1[:2]
        (x2, y2) = p2[:2]

        el1_x_0, el1_y_0 = map(int, [0, -el1[2] / el1[1]])
        el1_x_1, el1_y_1 = map(
            int, [w1, (-el1[0] / el1[1]) * w1 - el1[2] / el1[1]]
        )
        el2_x_0, el2_y_0 = map(int, [0, -el2[2] / el2[1]])
        el2_x_1, el2_y_1 = map(
            int, [w2, (-el2[0] / el2[1]) * w2 - el2[2] / el2[1]]
        )

        cv2.circle(output_img, (int(x1), int(y1)), 10, (255, 0, 255), 10)
        cv2.circle(output_img, (int(x2) + w1, int(y2)), 10, (0, 255, 255), 10)

        cv2.line(output_img, (el1_x_0, el1_y_0), (el1_x_1, el1_y_1), (0, 255, 255), 5)
        cv2.line(output_img, (el2_x_0 + w1, el2_y_0), (el2_x_1 + w1, el2_y_1), (255, 0, 255), 5)

    return output_img


def assemble_pose_matrix(R, t):
    '''
    Builds 4x4 pose matrix (extrinsics) from 3x3 rotation matrix and
    3-d translation vector. See also lecture two.

    Args:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3-d translation vector.

    Returns:
        pose (np.ndarray): 4x4 pose matrix (extrinsics).
    '''
    # augment R
    R = np.concatenate([R, np.zeros([1, 3])], axis=0)

    # augment T
    t = np.concatenate([t, np.ones([1, 1])])

    # assemble and return pose matrix
    return np.concatenate([R, t], axis=1)


def assemble_projection_matrix(K, R, t):
    '''
    Builds 3x4 projection matrix from 3x3, calibration matrix, 3x3 rotation
    matrix and 3-d translation vector. See also lecture two.

    Args:
        K (np.ndarray): 3x3 calibration matrix.
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3-d translation vector.

    Returns:
        P (np.ndarray): 4x4 pose matrix.
    '''
    # TODO: use assemble pose
    # augment K
    K = np.concatenate([K, np.zeros([3, 1])], axis=1)

    # augment R
    R = np.concatenate([R, np.zeros([1, 3])], axis=0)

    # augment T
    t = np.concatenate([t, np.ones([1, 1])])

    # assemble and return camera matrix P
    return K @ np.concatenate([R, t], axis=1)


def chirality_check(keypoints1, keypoint2, K1, K2, R1, R2, t):
    '''
    Triangulates world coordinates given correspondences from two views with
    relative extrinsics R and T.

    Args:
        keypoints1 (np.ndarray): Nx3 array of correspondence points in first
            view in homogenous image coordinates.
        keypoints2 (np.ndarray): Nx3 array of correspondence points in second
            view in homogenous image coordinates.
        K1 (np.ndarray): The 3x3 calibration matrix K for the first
            view/camera.
        K2 (np.ndarray): The 3x3 calibration matrix K for the second
            view/camera.
        R1 (np.ndarray): first possible 3x3 rotation matrix from first to
            second view.
        R2 (np.ndarray): second possible 3x3 rotation matrix from first to
            second view.
        T (np.ndarray): 3-d translation vector from first to second view.

    Returns:
        pose (tuple of np.ndarrays): 3x3 rotation matrix and 3-d translation
            vector from first to second view.
        x_w (np.ndarray): Nx4 array of 3-d points in homogenous world
            coordinates.
    '''
    solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    valid = [0, 0, 0, 0]
    all_triangulations = []
    for i, solution in enumerate(solutions):
        triangulations = []
        for kp1, kp2 in zip(keypoints1, keypoints2):
            # triangulate points with respect to reference camera
            x_w = triangulate_point(
                kp1, kp2,
                K1, K2,
                solution[0], solution[1],
            )
            triangulations.append(x_w)

            # perform chirality check
            visible = False
            # visibility in reference view
            if x_w[2] > 0 and x_w[2] < 50:
                visible = True

            # visibility in other view
            x_w = assemble_projection_matrix(K1, solution[0], solution[1]) @ x_w

            if x_w[2] > 0 and x_w[2] < 50:
                visible = True
            else:
                visible = False

            # increment number of valid points if visible in both
            if visible:
                valid[i] += 1

        # collect triangulations for all solutions
        # so we don't have to recompute them
        all_triangulations.append(
            np.array(triangulations, np.float32)
        )

    # return the solution for which the most points are visible in both cameras
    return solutions[np.argmax(valid)], all_triangulations[np.argmax(valid)]


def draw_camera(ax, pose, K):
    '''
    Draws a camera coordinate frame in 3d plot.

    Args:
        ax (matplotlib axes): The figure in which to draw the camera.
        pose (np.ndarray): 4x4 pose matrix (extrinsics).
        K (np.ndarray): 3x3 calibration matrix (intrinsics).

    Returns:
        ax (matplotlib): Figure with camera added.
    '''
    # set up unit vectors
    scale = K[0, 0] / 10000
    o = np.array([0, 0, 0, 1])
    x = np.array([1 * scale, 0, 0, 1])
    y = np.array([0, 1 * scale, 0, 1])
    z = np.array([0, 0, 1 * scale, 1])

    pose = np.linalg.inv(pose)
    o_prime = pose @ o
    x_prime = pose @ x
    y_prime = pose @ y
    z_prime = pose @ z

    ax.plot(
        [o_prime[0], x_prime[0]], [o_prime[2], x_prime[2]], [-o_prime[1], -x_prime[1]], c='r'
    )
    ax.plot(
        [o_prime[0], y_prime[0]], [o_prime[2], y_prime[2]], [-o_prime[1], -y_prime[1]], c='b'
    )
    ax.plot(
        [o_prime[0], z_prime[0]], [o_prime[2], z_prime[2]], [-o_prime[1], -z_prime[1]], c='g'
    )
    return ax


###########################
#### Exercise Function ####
###########################
def compute_fundamental_matrix(keypoints1, keypoints2):
    '''
    Computes the fundamental matrix from image coordinates using the 8-point
    algorithm by constructing and solving the corresponding linear system.

    Args:
        keypoints1 (np.ndarray): Nx3 array of correspondence points in first
            view in homogenous image coordinates.
        keypoints2 (np.ndarray): Nx3 array of correspondence points in second
            view in homogenous image coordinates.

    Returns:
        F (np.ndarray): 3x3 fundamental matrix.
    '''

    N = keypoints1.shape[0]
    assert N >= 8, "至少需要8对点"

    # 构建矩阵 A
    A = np.zeros((N, 9))
    for i in range(N):
        u, v, w = keypoints1[i]  # 齐次坐标 z，w = 0
        x, y, z = keypoints2[i]
        A[i] = [u * x, v * x, x, u * y, v * y, y, u, v, 1]
    # 使用SVD解基础矩阵
    _, _, Vt = np.linalg.svd(A)

    # 提取Vt的最后一列作为扁平基础矩阵
    f = Vt[-1]

    F = f.reshape((3, 3))

    # 对F进行SVD分解
    U, S, Vt = np.linalg.svd(F)

    # 将奇异值矩阵Sigma调整为仅保留前两个奇异值（第三个设为0）
    S[2] = 0

    # 重构基础矩阵F
    F = np.dot(U, np.dot(np.diag(S), Vt))
    return F / F[2, 2]


def compute_fundamental_matrix_normalized(keypoints1, keypoints2):
    '''
    Computes the fundamental matrix from image coordinates using the normalized
    8-point algorithm by first normalizing the keypoint coordinates to zero-mean
    and unit variance, then constructing and solving the corresponding linear
    system and finally undoing the normaliziation by back-transforming the
    resulting matrix.

    Args:
        keypoints1 (np.ndarray): Nx3 array of correspondence points in first
            view in homogenous image coordinates.
        keypoints2 (np.ndarray): Nx3 array of correspondence points in second
            view in homogenous image coordinates.

    Returns:
        F (np.ndarray): 3x3 fundamental matrix.
    '''
    # Normalize keypoints
    mean1 = np.mean(keypoints1, axis=0)
    mean2 = np.mean(keypoints2, axis=0)
    std1 = np.std(keypoints1, axis=0)
    std2 = np.std(keypoints2, axis=0)

    # 防止除0，由于齐次坐标，标准差std的最后一项为0
    std1[2] = 1
    std2[2] = 1

    nomalized_points1 = (keypoints1 - mean1) / std1
    nomalized_points2 = (keypoints2 - mean2) / std2

    F = compute_fundamental_matrix(nomalized_points1, nomalized_points2)

    T1 = np.array([
        [1 / std1[0], 0, -mean1[0] / std1[0]],
        [0, 1 / std1[1], -mean1[1] / std1[1]],
        [0, 0, 1]
    ])
    T2 = np.array([
        [1 / std2[0], 0, -mean2[0] / std2[0]],
        [0, 1 / std2[1], -mean2[1] / std2[1]],
        [0, 0, 1]
    ])

    # Undo normalization
    F_denormalized = np.dot(T2.T, np.dot(F, T1))

    return F_denormalized / F_denormalized[2, 2]


def compute_essential_matrix(F, K1, K2):
    '''
    Computes the essential from the fundamental matrix given known intrinsics.

    Args:
        F (np.ndarray): 3x3 fundamental matrix.
        K1 (np.ndarray): The 3x3 calibration matrix K for the first
            view/camera.
        K2 (np.ndarray): The 3x3 calibration matrix K for the second
            view/camera.

    Returns:
        E (np.ndarray): 3x3 essential matrix.
    '''
    E = K2.T @ F @ K1
    return E


def triangulate_point(keypoint1, keypoint2, K1, K2, R, t):
    '''
    Triangulates world coordinates given correspondences from two views with
    relative extrinsics R and t.

    Args:
        keypoints1 (np.ndarray): Nx3 array of correspondence points in first
            view in homogenous image coordinates.
        keypoints2 (np.ndarray): Nx3 array of correspondence points in second
            view in homogenous image coordinates.
        K1 (np.ndarray): The 3x3 calibration matrix K for the first
            view/camera.
        K2 (np.ndarray): The 3x3 calibration matrix K for the second
            view/camera.
        R (np.ndarray): 3x3 rotation matrix from first to second view.
        t (np.ndarray): 3-d translation vector from first to second view.

    Returns:
        x_w (np.ndarray): Nx4 array of 3-d points in homogenous world
            coordinates.
    '''
    P1 = assemble_projection_matrix(K1, np.eye(3), np.zeros((3, 1)))
    P2 = assemble_projection_matrix(K1, R, t)
    # print(keypoint1[0] * P1[2] - P1[0])
    A = np.array(
        [keypoint1[0] * P1[2] - P1[0],
         keypoint1[1] * P1[2] - P1[1],
         keypoint2[0] * P2[2] - P2[0],
         keypoint2[1] * P2[2] - P2[1]]
    )

    _, _, Vt = np.linalg.svd(A)
    x_w = Vt[-1]

    return x_w / x_w[3]


if __name__ == '__main__':
    # Load images
    img1 = cv2.cvtColor(cv2.imread('./assets/img1.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('./assets/img2.jpg'), cv2.COLOR_BGR2RGB)

    # Load cameras file
    camera_dict = np.load('./assets/cameras.npz')
    # camera1
    scale_mat1 = camera_dict['scale_mat_%d' % 45].astype(np.float32)
    world_mat1 = camera_dict['world_mat_%d' % 45].astype(np.float32)
    proj_mat1 = (world_mat1 @ scale_mat1)[:3, :4]
    K1 = cv2.decomposeProjectionMatrix(proj_mat1)[0]
    K1 = K1 / K1[2, 2]
    # camera2
    scale_mat2 = camera_dict['scale_mat_%d' % 41].astype(np.float32)
    world_mat2 = camera_dict['world_mat_%d' % 41].astype(np.float32)
    proj_mat2 = (world_mat2 @ scale_mat2)[:3, :4]
    K2 = cv2.decomposeProjectionMatrix(proj_mat2)[0]
    K2 = K2 / K2[2, 2]

    # # Let's visualize the images
    # f = plt.figure(figsize=(15, 5))
    # ax1 = f.add_subplot(121)
    # ax2 = f.add_subplot(122)
    # ax1.imshow(img1)
    # ax2.imshow(img2)
    # plt.show()

    keypoints1, keypoints2 = get_keypoints(img1, img2)
    correspondence_vis = draw_matches(img1, img2, keypoints1, keypoints2)
    # fig = plt.figure(figsize=(15, 10))
    # plt.imshow(correspondence_vis)
    # plt.show()

    # add noise to correspondences
    keypoints_noisy1 = np.copy(keypoints1)
    keypoints_noisy2 = np.copy(keypoints2)
    keypoints_noisy1[..., :2] += np.random.normal(0, 2, size=keypoints_noisy1[..., :2].shape)
    keypoints_noisy2[..., :2] += np.random.normal(0, 2, size=keypoints_noisy2[..., :2].shape)

    # # Compute F from noisy correspondences using 8-point algorithm
    # F = compute_fundamental_matrix(
    #     keypoints_noisy1,
    #     keypoints_noisy2
    # )
    #
    # # Compute epipolar lines
    # el2 = np.swapaxes(F @ keypoints1.swapaxes(0, 1), 0, 1)
    # el1 = np.swapaxes(F.transpose() @ keypoints2.swapaxes(0, 1), 0, 1)
    #
    # # Plot epipolar lines
    # el_vis = draw_epipolar_lines(
    #     img1, img2, el1[0:5], el2[0:5], keypoints1[0:5], keypoints2[0:5]
    # )
    #
    # # Compute F from noisy correspondences using normalized 8-point algorithm
    # F_normalized = compute_fundamental_matrix_normalized(
    #     keypoints_noisy1,
    #     keypoints_noisy2
    # )
    #
    # el2_normalized = np.swapaxes(F_normalized @ keypoints1.swapaxes(0, 1), 0, 1)
    # el1_normalized = np.swapaxes(F_normalized.transpose() @ keypoints2.swapaxes(0, 1), 0, 1)
    #
    # el_vis_normalized = draw_epipolar_lines(
    #     img1, img2,
    #     el1_normalized[0:5], el2_normalized[0:5],
    #     keypoints1[0:5], keypoints2[0:5]
    # )
    # fig = plt.figure(figsize=(15, 10))
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    # ax1.imshow(el_vis)
    # ax2.imshow(el_vis_normalized)
    # plt.show()

    # For the following sections we recompute F without artificial noise
    F_normalized = compute_fundamental_matrix_normalized(keypoints1, keypoints2)

    # From the essential matrix we can recover the relative rotation and translation between views
    E = compute_essential_matrix(F_normalized, K1, K2)
    R1, R2, t = cv2.decomposeEssentialMat(E)

    pose2, triangulations = chirality_check(keypoints1, keypoints2, K1, K2, R1, R2, t)

    # Draw results
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(211)
    ax1.imshow(draw_matches(img1, img2, [], []))
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.scatter(
        triangulations[:, 0],
        triangulations[:, 2],
        - triangulations[:, 1],
        c='r', marker='o',
    )

    P1 = assemble_pose_matrix(np.eye(3), np.zeros([3, 1]))
    ax2 = draw_camera(ax2, P1, K1)
    P2 = assemble_pose_matrix(pose2[0], pose2[1])
    ax2 = draw_camera(ax2, P2, K2)
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Z Axis')
    ax2.set_zlabel('Y Axis')
    plt.show()
