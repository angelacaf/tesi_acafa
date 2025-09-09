from projection.basic import *



# Defination of the class for a radiation line
# Note the theta belongs to (-pi, pi]
class ProjectionRay:
    def __init__(self, center: Point, theta: float, length: float=0):
        self.center = center
        self.theta = theta
        self.length = length
    
    def __str__(self, ):
        return '[%f, %f, %f, %f]' % (self.center.x, self.center.y, self.theta, self.length)
    
    def set_length(self, length):
        self.length = length
    
    @classmethod
    def from_numpy(cls, np_array):
        if len(np_array) == 3:
            center_x, center_y, theta = np_array
            length = 0.0
        else:
            center_x, center_y, theta, length = np_array

        center = Point(center_x, center_y)
        return cls(center, theta, length)
    

    @classmethod
    def from_ends(cls, start_point: Point, end_point: Point, has_length=False):
        center = Point((start_point.x+end_point.x)/2, (start_point.y+end_point.y)/2)
        theta = get_theta_from_points(start_point, end_point)
        length = start_point.distance_to_point(end_point) if has_length else 0.0
        return cls(center, theta, length)
    
    def to_np_array(self):
        numpy_data = np.float32([self.center.x, self.center.y, self.theta, self.length])
        return numpy_data

    def is_along_y(self):
        return self.is_up() or self.is_down()

    def is_along_x(self):
        return self.is_right() or self.is_left()

    def is_up(self):
        return self.theta == math.pi / 2
    
    def is_down(self):
        return self.theta == -math.pi / 2

    def is_right(self):
        return self.theta == 0
    
    def is_left(self):
        return (self.theta == math.pi) or (self.theta == -math.pi)

    # Get the end points of the projection line given the length
    def get_ends_by_length(self, length):
        start_point = self.center.clone().move_by_distance(-length/2, self.theta)
        end_point = self.center.clone().move_by_distance(length/2, self.theta)
        return start_point, end_point
    
    # Check whether the point is on projection line
    def is_point_on_line(self, point):
        if point == self.center:
            return True
        else:
            theta = get_theta_from_points(self.center, point)
            return theta == self.theta
    
    
    # Get x based on y on projection line, return None if the line is vertical
    def get_x_by_y(self, y):
        if self.is_along_y():
            return None
        elif self.is_along_x():
            return self.center.x
        else:
            return (y - self.center.y) / math.tan(self.theta) + self.center.x

    # Get y based on x on projection line, return None if the line is horizontal
    def get_y_by_x(self, x):
        if self.is_along_x():
            return None
        else:
            return (x-self.center.x) * math.tan(self.theta) + self.center.y

    # Get the ends of a line that intersects with the boundary, return None if there are no intersections
    def get_ends_on_boundary(self, boundary: Boundary):
        if self.is_along_x():
            if self.center.x < boundary.x_left or self.center.x > boundary.x_right:
                return None
            else:
                bottom_intersection = Point(self.center.x, boundary.y_left)
                top_intersection = Point(self.center.x, boundary.y_right)
                return bottom_intersection, top_intersection if self.is_up() else top_intersection, bottom_intersection
        elif self.is_along_y():
            if self.center.y < boundary.y_left or self.center.y > boundary.y_right:
                return None
            else:
                left_intersection = Point(boundary.x_left, self.center.y)
                right_intersection = Point(boundary.x_right, self.center.y)
                return left_intersection, right_intersection if self.is_right() else right_intersection, left_intersection
        else:
            left_intersection = Point(boundary.x_left, self.get_y_by_x(boundary.x_left))
            right_intersection = Point(boundary.x_right, self.get_y_by_x(boundary.x_right))

            bottom_intersection = Point(self.get_x_by_y(boundary.y_left), boundary.y_left)
            top_intersection = Point(self.get_x_by_y(boundary.y_right), boundary.y_right)
            
            # Find left end
            #print(f'Left Intersection: {left_intersection.x}, {left_intersection.y}')
            #print(f'Boundary: {boundary.left}, {boundary.right}, {boundary.bottom}, {boundary.top}')
            if left_intersection.is_in_boundary(boundary):
                left_point = left_intersection
            else:
                
                left_point = bottom_intersection if bottom_intersection.x < top_intersection.x else top_intersection
                if left_point.is_out_boundary(boundary):
                    return None

            # Find right end
            if right_intersection.is_in_boundary(boundary):
                right_point = right_intersection
            else:
                right_point = top_intersection if bottom_intersection.x <= top_intersection.x else bottom_intersection
                if right_point.is_out_boundary(boundary):
                    return None
                
            # print(left_point, right_point)
            if self.theta < math.pi / 2 and self.theta > - math.pi / 2:
                return left_point, right_point
            else:
               return right_point, left_point
    
    
    # Get sample points along ProjectionLine given resolution or number of points

    
    def get_sample_points_within_boundary(self, boundary, resolution=None, point_num=576):
        result = self.get_ends_on_boundary(boundary)
        if result is None:
            print(f"[WARNING] Ray theta={self.theta:.2f} at center=({self.center.x:.1f}, {self.center.y:.1f}) does not intersect the boundary.")
            return []  # Nessuna intersezione → ignora il raggio
        start_point, end_point = result
        return self.get_sample_points_within_ends(start_point, end_point, resolution, point_num)

    # Get sample points along ProjectionLine within a distance
    def get_sample_points_within_distance(self, distance=None, resolution=None, point_num=None):
        distance = self.length if distance is None else distance
        start_point = Point(x = self.center.x - distance / 2 * math.cos(self.theta), 
                            y = self.center.y - distance / 2 * math.sin(self.theta))
        end_point = Point(x = self.center.x + distance / 2 * math.cos(self.theta), 
                          y = self.center.y + distance / 2 * math.sin(self.theta))
        return self.get_sample_points_within_ends(start_point, end_point, resolution, point_num)
    
    
    # Get sample points along ProjectionLine within both distance and boundary
    def get_sample_points_within_distance_and_boundary(self, distance, boundary, resolution=None, point_num=None):
        distance = self.length if distance is None else distance

        left = self.center.x - abs(distance / 2 * math.cos(self.theta))
        right = self.center.x + abs(distance / 2 * math.cos(self.theta))
        bottom = self.center.y - abs(distance / 2 * math.sin(self.theta))
        top = self.center.y + abs(distance / 2 * math.sin(self.theta))
        
        boundary_from_distance = Boundary(right, top, left, bottom)
        
        # DEBUG: stampa i due boundary prima dell'unione
       # print(f"[DEBUG] boundary_input: {boundary}")
       # print(f"[DEBUG] boundary_from_distance: {boundary_from_distance}")

        union_boundary = boundary.get_union_boundary(boundary_from_distance)

        if union_boundary is None:
            print(f"[WARNING] Ray theta={self.theta:.2f} at center=({self.center.x:.1f}, {self.center.y:.1f}) → NO union")
            return []

        #print(f"[DEBUG] union_boundary: {union_boundary}")
        
        return self.get_sample_points_within_boundary(union_boundary, resolution, point_num)

    # Get sample points between two ends
    @staticmethod
    def get_sample_points_within_ends(start_point, end_point, resolution=None, point_num=None):
        assert (resolution is None) != (point_num is None)
        assert start_point != end_point
        
        distance = start_point.distance_to_point(end_point)
        if resolution is not None:
            point_num = math.floor(distance/resolution) + 1
        else:
            resolution = distance/(point_num - 1)
        
        sample_points = []
        segment_length = start_point.distance_to_point(end_point)
        for point_id in range(point_num):
            ratio = point_id * resolution / segment_length
            x = (end_point.x - start_point.x) * ratio +  start_point.x
            y = (end_point.y - start_point.y) * ratio +  start_point.y
            sample_points.append(Point(x, y))
        return sample_points