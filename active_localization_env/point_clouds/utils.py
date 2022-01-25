import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R


def do_lidar_scan(position, bsp_tree, num_points=1024):
    # Perform intersection
    r = R.from_euler('z', 360 / num_points, degrees=True).as_matrix()
    ray = np.array([5000, 0, 0])
    tol = 1
    p1 = position
    point_cloud = None
    points = vtk.vtkPoints()
    for _ in range(num_points):
        p2 = p1 + ray
        code = bsp_tree.IntersectWithLine(p1, p2, tol, points, None)
        if code is not 0:
            new_point = np.array(points.GetData().GetTuple3(0))[:2]
        else:
            new_point = p2[:2]
        if point_cloud is None:
            point_cloud = new_point
        else:
            point_cloud = np.vstack([point_cloud, new_point])
        ray = ray.dot(r)
    point_cloud = point_cloud - point_cloud.min()
    point_cloud = point_cloud / point_cloud.max()
    point_cloud = np.expand_dims(point_cloud, 0)
    return point_cloud
