''' Look-up of local information as explained in description '''

import numpy as np

def get_local_info(mesh_nr, pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]

    region = 0
    normals = np.zeros(9)
    normal_emb = np.zeros(3)
    wall_corners = np.zeros([3, 2])
    distances = np.zeros([1, 7])

    if mesh_nr == 1:
        if x <= 5000 and y <= 1750:
            region = 1
            normals = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1])
            normal_emb = np.array([0, 2, 4])

            wall_corners[0, :] = (0, 0)
            wall_corners[1, :] = (10000, 0)
            wall_corners[2, :] = (0, 5000)

        elif x > 5000 and y <= 1750:
            region = 2
            normals = np.array([0, 1, 0, -1, 0, 0, 0, 0, 1])
            normal_emb = np.array([0, 3, 4])

            wall_corners[0, :] = (0, 0)
            wall_corners[1, :] = (10000, 0)
            wall_corners[2, :] = (10000, 5000)

        elif (x <= 4000 and y >= 1750 and y <= 3000) or (x <= 2000 and y >= 3000):
            region = 3
            normals = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1])
            normal_emb = np.array([1, 2, 4])

            wall_corners[0, :] = (0, 5000)
            wall_corners[1, :] = (4000, 5000)
            wall_corners[2, :] = (0, 0)

        elif (x >= 6000 and y >= 1750 and y <= 3000) or (x >= 8000 and y >= 3000):
            region = 4
            normals = np.array([0, -1, 0, -1, 0, 0, 0, 0, 1])
            normal_emb = np.array([1, 3, 4])

            wall_corners[0, :] = (6000, 5000)
            wall_corners[1, :] = (10000, 5000)
            wall_corners[2, :] = (10000, 0)

        elif x >= 4000 and x <= 5000 and y >= 1750 and y <= 3000:
            region = 5
            normals = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1])
            normal_emb = np.array([1, 2, 4])

            wall_corners[0, :] = (4000, 3000)
            wall_corners[1, :] = (6000, 3000)
            wall_corners[2, :] = (0, 0)

        elif x >= 5000 and x <= 6000 and y >= 1750 and y <= 3000:
            region = 6
            normals = np.array([0, -1, 0, -1, 0, 0, 0, 0, 1])
            normal_emb = np.array([1, 3, 4])

            wall_corners[0, :] = (4000, 3000)
            wall_corners[1, :] = (6000, 3000)
            wall_corners[2, :] = (10000, 0)

        elif x >= 2000 and x <= 4000 and y >= 3000:
            region = 7
            normals = np.array([0, -1, 0, -1, 0, 0, 0, 0, 1])
            normal_emb = np.array([1, 3, 4])

            wall_corners[0, :] = (0, 5000)
            wall_corners[1, :] = (4000, 5000)
            wall_corners[2, :] = (4000, 3000)

        elif x >= 6000 and x <= 8000 and y >= 3000:
            region = 8
            normals = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1])
            normal_emb = np.array([1, 2, 4])

            wall_corners[0, :] = (6000, 5000)
            wall_corners[1, :] = (10000, 5000)
            wall_corners[2, :] = (6000, 3000)

        distances[:, 0:2] = abs(pos[0:2] - wall_corners[0, :])
        distances[:, 2:4] = abs(pos[0:2] - wall_corners[1, :])
        distances[:, 4:6] = abs(pos[0:2] - wall_corners[2, :])
        distances[:, 6] = abs(pos[2] - 5000) # 5000 = zmax

        # if above the half of the height, consider normal from ceiling
        if z > 2500:
            normal_emb[-1] = 5


    elif mesh_nr == 2:
        if x <= 1000 and y <= 1000:
            region = 1
            normal_emb = np.array([0, 6, 4])

            wall_corners[0, :] = (0, 0)
            wall_corners[1, :] = (1000, 0)
            wall_corners[2, :] = (1000, 1000)

        elif x >= 4000 and y <= 1000:
            region = 2
            normal_emb = np.array([0, 7, 4])

            wall_corners[0, :] = (5000, 0)
            wall_corners[1, :] = (4000, 0)
            wall_corners[2, :] = (4000, 1000)

        elif x <= 1000 and y >= 6000:
            region = 3
            normal_emb = np.array([1, 8, 4])

            wall_corners[0, :] = (0, 7000)
            wall_corners[1, :] = (1000, 7000)
            wall_corners[2, :] = (1000, 6000)

        elif x >= 4000 and y >= 6000:
            region = 4
            normal_emb = np.array([1, 9, 4])

            wall_corners[0, :] = (5000, 7000)
            wall_corners[1, :] = (4000, 7000)
            wall_corners[2, :] = (4000, 6000)

        elif 1000 <= x <= 2000 and 1000 <= y <= 3000:
            region = 5
            normal_emb = np.array([1, 2, 4])

            wall_corners[0, :] = (1000, 3000)
            wall_corners[1, :] = (2000, 3000)
            wall_corners[2, :] = (1000, 1000)

        elif 3000 <= x <= 4000 and 1000 <= y <= 3000:
            region = 6
            normal_emb = np.array([1, 3, 4])

            wall_corners[0, :] = (4000, 3000)
            wall_corners[1, :] = (3000, 3000)
            wall_corners[2, :] = (4000, 1000)

        elif 1000 <= x <= 2000 and 4000 <= y <= 6000:
            region = 7
            normal_emb = np.array([0, 2, 4])

            wall_corners[0, :] = (1000, 4000)
            wall_corners[1, :] = (2000, 4000)
            wall_corners[2, :] = (1000, 6000)

        elif 3000 <= x <= 4000 and 4000 <= y <= 6000:
            region = 8
            normal_emb = np.array([0, 3, 4])

            wall_corners[0, :] = (4000, 4000)
            wall_corners[1, :] = (3000, 4000)
            wall_corners[2, :] = (4000, 6000)

        distances[:, 0:2] = abs(pos[0:2] - wall_corners[0, :])
        distances[:, 2:4] = abs(pos[0:2] - wall_corners[1, :])
        distances[:, 4:6] = abs(pos[0:2] - wall_corners[2, :])
        distances[:, 6] = abs(pos[2] - 5000) # 5000 = zmax

        if z > 2500:
            normal_emb[-1] = 5


    elif mesh_nr == 3:
        if 3000 <= x <= 4500 and y <= 1700:
            region = 1
            normal_emb = np.array([0, 3, 4])

            wall_corners[0, :] = (4500, 0)
            wall_corners[1, :] = (3000, 0)
            wall_corners[2, :] = (4500, 1700)

        elif 6000 <= x <= 7500 and y <= 1700:
            region = 2
            normal_emb = np.array([0, 2, 4])

            wall_corners[0, :] = (6000, 0)
            wall_corners[1, :] = (7500, 0)
            wall_corners[2, :] = (6000, 1700)

        elif 3000 <= x <= 4500 and y >= 14000:
            region = 3
            normal_emb = np.array([1, 3, 4])

            wall_corners[0, :] = (4500, 15900)
            wall_corners[1, :] = (3000, 15900)
            wall_corners[2, :] = (4500, 14000)

        elif 6000 <= x <= 7500 and y >= 14000:
            region = 4
            normal_emb = np.array([1, 2, 4])

            wall_corners[0, :] = (6000, 15900)
            wall_corners[1, :] = (7500, 15900)
            wall_corners[2, :] = (6000, 14000)

        elif x <= 3000 and y <= 1700:
            region = 5
            normal_emb = np.array([1, 7, 4])

            wall_corners[0, :] = (0, 1700)
            wall_corners[1, :] = (3000, 1700)
            wall_corners[2, :] = (3000, 0)

        elif x >= 7500 and y <= 1700:
            region = 6
            normal_emb = np.array([1, 6, 4])

            wall_corners[0, :] = (10500, 1700)
            wall_corners[1, :] = (7500, 1700)
            wall_corners[2, :] = (7500, 0)

        elif x <= 3000 and 1700 <= y <= 6900:
            region = 7
            normal_emb = np.array([1, 11, 4])

            wall_corners[0, :] = (0, 6900)
            wall_corners[1, :] = (3000, 6900)
            wall_corners[2, :] = (3000, 1700)

        elif x >= 7500 and 1700 <= y <= 6900:
            region = 8
            normal_emb = np.array([1, 10, 4])

            wall_corners[0, :] = (10500, 6900)
            wall_corners[1, :] = (7500, 6900)
            wall_corners[2, :] = (7500, 1700)

        elif x <= 3000 and 9000 <= y <= 14000:
            region = 9
            normal_emb = np.array([0, 13, 4])

            wall_corners[0, :] = (0, 9000)
            wall_corners[1, :] = (3000, 9000)
            wall_corners[2, :] = (3000, 14000)

        elif x >= 7500 and 9000 <= y <= 14000:
            region = 10
            normal_emb = np.array([0, 12, 4])

            wall_corners[0, :] = (10500, 9000)
            wall_corners[1, :] = (7500, 9000)
            wall_corners[2, :] = (7500, 14000)

        elif x <= 3000 and y >= 14000:
            region = 11
            normal_emb = np.array([0, 9, 4])

            wall_corners[0, :] = (0, 14000)
            wall_corners[1, :] = (3000, 14000)
            wall_corners[2, :] = (3000, 15900)

        elif x >= 7500 and y >= 14000:
            region = 12
            normal_emb = np.array([0, 8, 4])

            wall_corners[0, :] = (10500, 14000)
            wall_corners[1, :] = (7500, 14000)
            wall_corners[2, :] = (7500, 15900)

        distances[:, 0:2] = abs(pos[0:2] - wall_corners[0, :])
        distances[:, 2:4] = abs(pos[0:2] - wall_corners[1, :])
        distances[:, 4:6] = abs(pos[0:2] - wall_corners[2, :])
        distances[:, 6] = abs(pos[2] - 3000) # 3000 = zmax

        if z > 1500:
            normal_emb[-1] = 5

    return region, normals, normal_emb, distances


def correct_rotations_m1(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    region = 0
    rotations = 0

    # everything on floor
    if x <= 5000 and y <= 1750 and z <= 2500:
        region = 1
        rotations = np.array([[0, 0, -0.5], [0, 0, 1], [0, 1, 0]])
    if x > 5000 and y <= 1750 and z <= 2500:
        region = 2
        rotations = np.array([[0, 0, -0.5], [0, 0, 0], [0, 1, 0]])
    if (x <= 4000 and y >= 1750 and y <= 3000) or (x <= 2000 and y >= 3000) and z <= 2500:
        region = 3
        rotations = np.array([[0, 0, 0.5], [0, 0, 1], [0, 1, 0]])
    if (x >= 6000 and y >= 1750 and y <= 3000) or (x >= 8000 and y >= 3000) and z <= 2500:
        region = 4
        rotations = np.array([[0, 0, 0.5], [0, 0, 0], [0, 1, 0]])
    if x >= 4000 and x <= 5000 and y >= 1750 and y <= 3000 and z <= 2500:
        region = 5
        rotations = np.array([[0, 0, 0.5], [0, 0, 1], [0, 1, 0]])
    if x >= 5000 and x <= 6000 and y >= 1750 and y <= 3000 and z <= 2500:
        region = 6
        rotations = np.array([[0, 0, 0.5], [0, 0, 0], [0, 1, 0]])
    if x >= 2000 and x <= 4000 and y >= 3000 and z <= 2500:
        region = 7
        rotations = np.array([[0, 0, 0.5], [0, 0, 0], [0, 1, 0]])
    if x >= 6000 and x <= 8000 and y >= 3000 and z <= 2500:
        region = 8
        rotations = np.array([[0, 0, 0.5], [0, 0, 1], [0, 1, 0]])

    # everything on top
    if x <= 5000 and y <= 1750 and z > 2500:
        region = 1
        rotations = np.array([[0, 0, -0.5], [0, 0, 1], [0, -1, 0]])
    if x > 5000 and y <= 1750 and z > 2500:
        region = 2
        rotations = np.array([[0, 0, -0.5], [0, 0, 0], [0, -1, 0]])
    if (x <= 4000 and y >= 1750 and y <= 3000) or (x <= 2000 and y >= 3000) and z > 2500:
        region = 3
        rotations = np.array([[0, 0, 0.5], [0, 0, 1], [0, -1, 0]])
    if (x >= 6000 and y >= 1750 and y <= 3000) or (x >= 8000 and y >= 3000) and z > 2500:
        region = 4
        rotations = np.array([[0, 0, 0.5], [0, 0, 0], [0, -1, 0]])
    if x >= 4000 and x <= 5000 and y >= 1750 and y <= 3000 and z > 2500:
        region = 5
        rotations = np.array([[0, 0, 0.5], [0, 0, 1], [0, -1, 0]])
    if x >= 5000 and x <= 6000 and y >= 1750 and y <= 3000 and z > 2500:
        region = 6
        rotations = np.array([[0, 0, 0.5], [0, 0, 0], [0, -1, 0]])
    if x >= 2000 and x <= 4000 and y >= 3000 and z > 2500:
        region = 7
        rotations = np.array([[0, 0, 0.5], [0, 0, 0], [0, -1, 0]])
    if x >= 6000 and x <= 8000 and y >= 3000 and z > 2500:
        region = 8
        rotations = np.array([[0, 0, 0.5], [0, 0, 1], [0, -1, 0]])

    return region, rotations


def local_info_robot(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    region = 0
    normals = 0
    normal_emb = 0
    wall_corners = np.zeros([3, 2])
    distances = np.zeros([1, 7])

    ## right side of J floor
    if 15.5 <= x <= 17.5 and 5.5 <= y <= 8.4:
        region = 1
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([1, 2, 4])

        wall_corners[0, :] = (15.5, 8.4)
        wall_corners[1, :] = (17.5, 8.4)
        wall_corners[2, :] = (15.5, 5.5)

    elif 15.5 <= x <= 17.5 and 2.6 <= y <= 5.5:
        region = 2
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([0, 2, 4])

        wall_corners[0, :] = (15.5, 2.6)
        wall_corners[1, :] = (17.5, 2.6)
        wall_corners[2, :] = (15.5, 5.5)

    elif x >= 15.5 and y >= 4.5:
        region = 3
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([1, 3, 4])

        wall_corners[0, :] = (23.3, 9.7)
        wall_corners[1, :] = (15.5, 9.7)
        wall_corners[2, :] = (23.3, 5.5)

    elif x >= 15.5 and y < 5.5:
        region = 4
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([0, 3, 4])

        wall_corners[0, :] = (23.3, 0)
        wall_corners[1, :] = (15.5, 0)
        wall_corners[2, :] = (23.3, 5.5)

    ## left side of J floor
    elif -18.5 <= x <= -17 and 5.5 <= y <= 8.5:
        region = 1
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([1, 3, 4])

        wall_corners[0, :] = (-17, 8.5)
        wall_corners[1, :] = (-18.5, 8.5)
        wall_corners[2, :] = (-17, 5.5)

    elif -18.5 <= x <= -17 and 2.5 <= y <= 5.5:
        region = 2
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([0, 3, 4])

        wall_corners[0, :] = (-17, 5.5)
        wall_corners[1, :] = (-18.5, 5.5)
        wall_corners[2, :] = (-17, 2.5)

    elif x <= -17 and y >= 5.5:
        region = 3
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([1, 2, 4])

        wall_corners[0, :] = (-23.5, 9.7)
        wall_corners[1, :] = (-17, 9.7)
        wall_corners[2, :] = (-23.3, 5.5)

    elif x <= -17 and y <= 5.5:
        region = 4
        normals = ([0, 1, 0, 1, 0, 0, 0, 0, 1])
        normal_emb = np.array([0, 2, 4])

        wall_corners[0, :] = (-23.5, 0)
        wall_corners[1, :] = (-17, 0)
        wall_corners[2, :] = (-23.3, 5.5)

    distances[:, 0:2] = abs(pos[0:2] - wall_corners[0, :])
    distances[:, 2:4] = abs(pos[0:2] - wall_corners[1, :])
    distances[:, 4:6] = abs(pos[0:2] - wall_corners[2, :])
    distances[:, 6] = abs(pos[2] - 3)  # 3m = height of floor

    if z > 1.5:
        normal_emb[-1] = 5

    return region, normals, normal_emb, distances
