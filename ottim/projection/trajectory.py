from projection.basic import *
from projection.projection_ray import ProjectionRay
from core.tools import norm_theta

# The class of Trajectory is defined to simluate the movement of a projector, we assume 
# the trajectory moves within a 2D plane.
class Trajectory():
    def __init__(self, start_point, end_point, curve_function) -> None:
        self.start_point = start_point
        self.end_point = end_point
        self.curve_function = curve_function
        self.xs, self.ys = None, None
    
    def get_ys_on_curve(self, xs):
        ys = [self.curve_function(x) for x in xs]
        return np.array(ys)
    
    # Return samples on trajectory with equal distance
    def get_points_array_on_trajectory(self, resolution):
        if (self.xs is None) or (self.ys is None):
            self.xs = np.arange(self.start_point.x, self.end_point.x + resolution, resolution)
            self.ys = self.get_ys_on_curve(self.xs)
        return self.xs, self.ys
    
    # Return samples on trajectory with equal distance
    def get_points_on_trajectory_in_equal_curve(self, sample_num, resolution=0.01, fast_mode=False):
        xs = np.arange(self.start_point.x, self.end_point.x + resolution, resolution)
        ys = self.get_ys_on_curve(xs)
        curve_distance_array = np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)
        
        # get accumulate distance list
        accumulate_length_list = [0.0]
        total_length = 0
        for curve_distance in curve_distance_array:
            total_length += curve_distance
            accumulate_length_list.append(total_length)
        
        curve_length = accumulate_length_list[-1]
        sample_points = []
        point_num = len(xs)
        if fast_mode:
            arch_resolution = curve_length / (sample_num + 1)
            target_arch_length = arch_resolution
            for point_id in range(point_num - 2):

                if accumulate_length_list[point_id] == target_arch_length:
                    sample_points.append(Point(xs[point_id], ys[point_id]))
                    target_arch_length += arch_resolution
                    
                elif accumulate_length_list[point_id] < target_arch_length and accumulate_length_list[point_id+1] > target_arch_length:
                    x = (xs[point_id] + xs[point_id + 1]) / 2
                    y = self.curve_function(x)
                    sample_points.append(Point(x, y))
                    target_arch_length += arch_resolution
                else:
                    pass
        else:
            arch_resolution = curve_length / (sample_num+1)
            for i in range(1, sample_num + 1):
                distance_array = accumulate_length_list - arch_resolution*i
                closest_index = np.argmin(np.abs(distance_array))
                sample_points.append(Point(xs[closest_index], ys[closest_index]))
                
        if sample_num != len(sample_points):
                print('Get %d out of %d sample points with resolution: %f under mode: %s' % 
                      (len(sample_points), sample_num, resolution, 'fast' if fast_mode else 'normal'))
                
        return sample_points
    
    def sample_projection_rays(self, sample_n, ray_length=0, delta_theta=0):
        sample_points_on_trajectory = self.get_points_on_trajectory_in_equal_curve(sample_n)
        projection_rays = []
        for point in sample_points_on_trajectory:
            theta = self.get_normal_at_point(point)
            theta += delta_theta  # aggiungi rotazione globale
            projection_rays.append(ProjectionRay(point, theta, ray_length))
        return projection_rays

        
    
    def get_tangent_at_point(self, point, resolution=0.01):
        left_neighbour = Point(point.x-resolution, self.curve_function(point.x-resolution))
        right_neighbour_point = Point(point.x+resolution, self.curve_function(point.x+resolution))
        tangent_theta = get_theta_from_points(left_neighbour, right_neighbour_point)
        return tangent_theta
    
    def get_normal_at_point(self, point, resolution=0.01):
        tangent_theta = self.get_tangent_at_point(point, resolution)
        normal_theta = norm_theta(tangent_theta - math.pi / 2)
        return normal_theta
    
    def get_distance_to_point(self, point: Point, resolution=0.5):
        if self.xs is None:
            self.xs, self.ys = self.get_points_array_on_trajectory(resolution)
        distance_list = [math.sqrt((x-point.x) ** 2 + (y-point.y) ** 2) for x, y in zip(self.xs, self.ys)]
        return min(distance_list)
    
    def get_distance_to_points(self, points, resolution=0.5):
        if self.xs is None:
            self.xs, self.ys = self.get_points_array_on_trajectory(resolution)
        points_x_array = np.reshape([point.x for point in points], (-1, 1))
        points_y_array = np.reshape([point.y for point in points], (-1, 1))
        
        trajctory_points_x = np.reshape([x for x in self.xs], (1, -1))
        trajctory_points_y = np.reshape([y for y in self.ys], (1, -1))
        
        distance_map = np.sqrt((points_x_array - trajctory_points_x) ** 2 + (points_y_array - trajctory_points_y) ** 2)
        min_distance = np.min(distance_map, axis=1)
        return min_distance



