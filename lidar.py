import numpy as np
from pybullet_tools.utils import get_aabb, aabb_contains_point
import pybullet as p
from utils import dist

class Lidar:
    def __init__(self, step_length=0.1, lidar_range=1, slices=90):
        self.slices = slices
        self.max_step_size = np.sqrt(step_length**2 + step_length**2)
        self.lidar_range = lidar_range
        self.step_size = self.max_step_size/2
        self.z = 0.1
        self.rot_step = (2 * np.pi)/self.slices

    # returns list of distances where object is first detected
    # state is an np array of the robots pose [x, y, theta]
    def getLidarRaw(self, state, obstacles):
        dists = [np.inf for i in range(0, self.slices)]
        start_point = np.asarray([state[0], state[1], self.z])
        # iterate through each slice and get distance to obstacle
        for i in range(0, self.slices):
            # find unit vector where the slice is facing
            rotation = state[2] + (i * self.rot_step)
            direction = np.asarray([1 * np.cos(rotation), 1 * np.sin(rotation), 0]) * (1/(np.sqrt((1 * np.cos(rotation))**2 + (1 * np.sin(rotation))**2)))

            curr_point = np.asarray([state[0], state[1], self.z])
            # step towards obstacle or max_distance
            isTerminated = False
            while not isTerminated:
                # Step in direction of slice
                step = direction * self.step_size
                curr_point = curr_point + step

                # check to see if state is within obstacles
                for ob_idx in obstacles.keys():
                    obstacle_params = get_aabb(obstacles[ob_idx])
                    if aabb_contains_point(curr_point, obstacle_params):
                        dists[i] = dist(start_point, curr_point)
                        isTerminated = True
                        break

                # check to see if it hit the range limit on lidar
                if dist(start_point, curr_point) >= self.lidar_range:
                    isTerminated = True
            ##### END OF stepping WHILE-LOOP #####
        ##### END OF slice FOR-LOOP #####
        return dists


    # returns a list of points where an obstacle was seen
    def getLidarScan(self, state, obstacles):
        dists = self.getLidarRaw(state, obstacles)

        # data structure to return
        obstacle_points = []

        robot_x = state[0]
        robot_y = state[1]
        robot_theta = state[2]

        # loop through all distances
        for i, distance in enumerate(dists):
            # check to see if an obstacle was seen
            if distance == np.inf: # no obstacle seen
                continue

            # convert from polar to rectangular coordinates
            theta = robot_theta + (i * self.rot_step)
            obstacle_points.append([(distance * np.cos(theta)) + robot_x, (distance * np.sin(theta)) + robot_y])

        return obstacle_points

    def getRange(self):
        return self.lidar_range
