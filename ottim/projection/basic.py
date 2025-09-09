# This file is used to define the class Trajectory and its related functions

import dataclasses
import math
from scipy.interpolate import CubicSpline
import numpy as np


class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Point):
            if (self.x == other.x) and (self.y == other.y):
                return True
        return False
    
    def __str__(self, ):
        return '[%f, %f]' % (self.x, self.y)
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def distance_to_point(self, other_point):
        assert isinstance(other_point, Point)
        return math.sqrt((self.x - other_point.x)**2+(self.y - other_point.y)**2)

    def move(self, direction, distance):
        x = self.x + distance * math.cos(direction)
        y = self.y + distance * math.sin(direction)
        return Point(x, y)

    def is_in_boundary(self, boundary):
        if self.x < boundary.x_left or self.x > boundary.x_right or \
           self.y < boundary.y_left or self.y > boundary.y_right:
            return False
        else:
            return True
    
    def move_by_distance(self, distance, theta):
        x_diff = distance * math.cos(theta)
        y_diff = distance * math.sin(theta)
        self.x += x_diff
        self.y += y_diff

    def is_out_boundary(self, boundary):
        return not self.is_in_boundary(boundary)
    
    def clone(self,):
        return Point(self.x, self.y)


class Boundary:
    def __init__(self, x_right, y_right, x_left=0.0, y_left=0.0) -> None:
        self.x_left, self.x_right = x_left, x_right
        self.y_left, self.y_right = y_left, y_right
    
    def __repr__(self):
        return f"Boundary(x_left={self.x_left:.1f}, x_right={self.x_right:.1f}, y_left={self.y_left:.1f}, y_right={self.y_right:.1f})"

    def is_point_inside(self, point: Point):
        return point.is_in_boundary(self)
    
    def get_union_boundary(self, target_boundary):
        if target_boundary is None:
            print("[ERROR] target_boundary è None → impossibile calcolare unione")
            return None

        x_left = max(self.x_left, target_boundary.x_left)
        x_right = min(self.x_right, target_boundary.x_right)
        y_left = max(self.y_left, target_boundary.y_left)
        y_right = min(self.y_right, target_boundary.y_right)
        
        # return None if union is empty
        if x_left > x_right or y_left > y_right:
            print(f"[DEBUG] No overlapping area: self={self}, target={target_boundary}")
            return None
        else:
            return Boundary(x_right, y_right, x_left, y_left)



# return theta between two points in (-pi, pi]
def get_theta_from_points(start_point: Point, end_point: Point):
    assert not start_point == end_point

    if start_point.x == end_point.x:
        return math.pi / 2 if end_point.y > start_point.y else - math.pi / 2
    elif start_point.y == end_point.y:
        return 0 if end_point.x > start_point.x else math.pi
    else:
        y_diff = end_point.y - start_point.y
        x_diff = end_point.x - start_point.x
        theta = math.atan(y_diff / x_diff)
        if x_diff < 0:
            return -math.pi + theta if y_diff < 0 else math.pi + theta
        else:
            return theta