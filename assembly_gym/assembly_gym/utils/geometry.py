import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from compas.geometry import Plane, Point, Rotation, distance_point_point


def merge_coplanar_faces(mesh):
    faces = [*mesh.faces()]
    while len(faces) > 0:
        face = faces.pop()
        for face2 in mesh.face_neighborhood(face):
            points = mesh.face_coordinates(face) + mesh.face_coordinates(face2)
            points = [Point(*p) for p in points]

            if is_coplanar(points):
                new_face = mesh.merge_faces([face, face2])
                faces.remove(face2)
                faces.append(new_face)
                break


def contains_point(plane, point, tol=1e-6):
    # compute distance from point to plane
    return np.abs(np.inner(point - plane.point, plane.normal)) <= tol


def is_coplanar(points):
    if len(points) < 4:
        return True
    plane = Plane.from_three_points(*points[:3])
    for p in points[3:]:
        if not contains_point(plane, p):
            return False
    return True


def align_frames_2d(frame1, frame2, frame1_coordinates=None):
    """
    Compute the offset and rotation to align frame1 with frame2 in 2D, i.e. rotate around the y axis.
    """
    if frame1_coordinates is None:
        frame1_coordinates = [0, 0, 0]
    n1 = frame1.normal
    n2 = frame2.normal
    rotation = Rotation.from_axis_and_angle(np.cross(n1, n2) + np.array([0, 1e-6, 0]),
                                            np.arccos(np.clip(-np.dot(n1, n2), -1.0, 1.0))).rotation
    offset = frame1.to_world_coordinates(frame1_coordinates) - frame2.point.transformed(rotation)
    return offset, rotation


def affine_transform_vertices(vertices, shift, rotation):
    rot = Rotation.from_euler("xyz", rotation)
    return shift + rot.apply(vertices)



def check_collision2D(shape1, shape2):  # only works in 2D: the block must be aligned wrt to the y axis
    vertices1_2d = np.delete(shape1[np.where(shape1[:, 1] > 0)], 1, 1)
    vertices2_2d = np.delete(shape2[np.where(shape2[:, 1] > 0)], 1, 1)

    polygon1 = Polygon(vertices1_2d)
    polygon2 = Polygon(vertices2_2d)

    return polygon1.intersects(polygon2)


def quaternion_distance(q1, q2):
    """
    Computes the angle between two quaternions
    """
    return np.arccos(2 * np.dot(q1.unitized().xyzw, q2.unitized().xyzw) ** 2 - 1)

def distance_box_point(box, point):
    """
    Computes the distance between a box and a point
    """
    if box.contains(point):
        return 0.

    return distance_point_point(point, project_point_on_box(box, point))


def project_point_on_box(box, point):
    """
    Projects a point on a box
    """
    return Point(min(max(point[0], box.xmin), box.xmax),
                 min(max(point[1], box.ymin), box.ymax),
                 min(max(point[2], box.zmin), box.zmax))


# Works when only using horizontal rectangular blocks (with file block.urdf)
def collision_rectangles(pos, state):
    if len(state['blocks']) > 0 and ((abs(np.array(state['blocks'])[:,0] - pos[0]) < 0.099) & (abs(np.array(state['blocks'])[:,2] - pos[2]) < 0.049)).any():
        return True
    if len(state['obstacles']) > 0 and ((abs(np.array(state['obstacles'])[:,0] - pos[0]) < 0.074) & (abs(np.array(state['obstacles'])[:,2] - pos[2]) < 0.049)).any():
        return True
    return False