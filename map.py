import numpy as np
from queue import Queue

"""
 This file is a map object that will hold our complete map and update it based on lidar scans
 The cells are held within a map to allow for dynamic expansion of map
"""
class Map:
    def __init__(self, resolution, robot):
        self.myrobot = robot
        self.trans_step_size = self.myrobot.get_trans_step_size()
        self.rot_step_size = self.myrobot.get_rot_step_size()
        self.resolution = resolution
        self.res_place = 1000
        self.max_range = 10
        self.bfs_depth = 2
        self.obstacles = {}
        self.newObstacles = []

    ############## Public member functions ##################
    # takes in point and finds cell that contains the point -- returns if there is obstacle there or not
    def isObstacle(self, point):
        #hash = self.hash_coord(point)

        if self.obstacles.get((point[0], point[1])) is None:
            # Check area around for containing obstacles -- creates a buffer
            # breadth first search with a depth of 2
            bq = Queue()
            exploredSet = {}

            # put first element in queue -- (point, depth)
            bq.put(([point[0], point[1]], 0))
            exploredSet[tuple([point[0], point[1]])] = True

            while not bq.empty():
                top_pt = bq.get()

                # Add children
                if top_pt[1] < self.bfs_depth:
                    r = [top_pt[0][0] + self.resolution, top_pt[0][1]]
                    if exploredSet.get(tuple(r)) is None:
                        bq.put((r, top_pt[1] + 1))
                        exploredSet[tuple(r)] = True
                    l = [top_pt[0][0] - self.resolution, top_pt[0][1]]
                    if exploredSet.get(tuple(l)) is None:
                        bq.put((l, top_pt[1] + 1))
                        exploredSet[tuple(l)] = True
                    t = [top_pt[0][0], top_pt[0][1] + self.resolution]
                    if exploredSet.get(tuple(t)) is None:
                        bq.put((t, top_pt[1] + 1))
                        exploredSet[tuple(t)] = True
                    b = [top_pt[0][0], top_pt[0][1] - self.resolution]
                    if exploredSet.get(tuple(b)) is None:
                        bq.put((b, top_pt[1] + 1))
                        exploredSet[tuple(b)] = True
                    tr = [top_pt[0][0] + self.resolution, top_pt[0][1] + self.resolution]
                    if exploredSet.get(tuple(tr)) is None:
                        bq.put((tr, top_pt[1] + 1))
                        exploredSet[tuple(tr)] = True
                    bl = [top_pt[0][0] - self.resolution, top_pt[0][1] - self.resolution]
                    if exploredSet.get(tuple(bl)) is None:
                        bq.put((bl, top_pt[1] + 1))
                        exploredSet[tuple(bl)] = True
                    br = [top_pt[0][0] + self.resolution, top_pt[0][1] - self.resolution]
                    if exploredSet.get(tuple(br)) is None:
                        bq.put((br, top_pt[1] + 1))
                        exploredSet[tuple(br)] = True
                    tl = [top_pt[0][0] - self.resolution, top_pt[0][1] + self.resolution]
                    if exploredSet.get(tuple(tl)) is None:
                        bq.put((tl, top_pt[1] + 1))
                        exploredSet[tuple(tl)] = True

                # check if current point is an obstacle
                if self.obstacles.get(tuple(top_pt[0])) is not None:
                    return True

            return False
        else:
            return True

    # Updates global map and gives new obstacles that were added
    def updateMap(self, obstacle_points):
        # Clear new obstacles list from last update
        self.newObstacles.clear()
        # Iterate through each point seen by lidar
        for point in obstacle_points:
            #Convert the continuous point to a discretized cell in map
            new_point = self.roundPointToCell(point)
            # Hash the point
            #hash = self.hash_coord(new_point)

            if self.obstacles.get((new_point[0], new_point[1])) is None:
                self.newObstacles.append(new_point)
            self.obstacles[(new_point[0], new_point[1])] = True

    # Takes state as [x, y, theta] coordinate in continuous space and discretizes based on map resolution
    # Returns a discretized point
    def roundPointToCell(self, point):
        nearest_x = round(float(point[0])/self.resolution) * self.resolution
        nearest_y = round(float(point[1])/self.resolution) * self.resolution

        if len(point) == 3:
            return [nearest_x, nearest_y, point[2]]

        new_point = [nearest_x, nearest_y]
        return new_point


    def getNewPoints(self):
        return self.newObstacles

    # Resets map
    def clear_map(self):
        self.obstacles.clear()

    def set_resolution(self, resolution):
        self.resolution = resolution

    ############ Private member functions ###############
    # Use cantor pairing to create hash for coordinate
    def hash_coord(self, point):
        # convert to integer
        # We know that the point will be maximum in the thousandths place
        x_coord = point[0] * 1000
        y_coord = point[1] * 1000

        # convert to positive integer
        x_coord += np.ceil(self.max_range / self.resolution)
        y_coord += np.ceil(self.max_range / self.resolution)

        cantor_pair = 0.5 * (x_coord + y_coord) * (x_coord + y_coord + 1) + y_coord
        return cantor_pair

