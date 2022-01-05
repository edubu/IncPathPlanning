import numpy as np
from map import Map
from lidar import Lidar
from pybullet_tools.utils import get_joint_positions, joint_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from utils import load_env, execute_trajectory, draw_sphere_marker, dist
from plot_utils import plot_points
import time

# Importing planners
from a_star import Astar
from dstarlite import Dstarlite
from lpastar import Lpastar

class Robot:
    def __init__(self, planner_type='D*'):
        # Set all movement params
        self.step_size = 0.2
        self.rot_step = np.pi/2

        # Create map and set map resolution
        self.resolution = round(np.sqrt((self.step_size**2)/2), 3)
        self.global_map = Map(self.resolution, self)

        # Create lidar and configure
        self.lidar = Lidar()

        # State variables
        self.curr_state = None
        self.goal_state = None
        self.start_config = None
        self.start_state = None
        self.goal_config = None
        self.seen_obstacles = []

        # Obstacle and Robot resources
        self.obstacles = None
        self.robots = None
        self.base_joints = None

        ## Planner variables
        self.planner_type = planner_type
        self.planner = None
        self.determinePlanner()

        # Shared Algorithm Variables
        self.update_path_threshold = self.step_size * 2
        self.update_ignore_threshold = self.lidar.getRange() * 1.3
        self.goal_threshold = self.step_size/2

        # Timing variables
        self.start_time = None
        self.total_time = None

        # Debug tracker
        self.debug = False


    ########### Public Member Functions ###########

    # Loads world and sets parameters
    def load_world(self, world_name):
        # load robot and obstacle resources
        robots, obstacles = load_env(world_name)
        self.robots = robots
        self.obstacles = obstacles

        # define active DoFs
        self.base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

        # set start config
        self.start_config = tuple(get_joint_positions(robots['pr2'], self.base_joints))
        self.start_state = list(self.start_config)

        # change parameters based on world and planner
        if world_name == 'pr2doorway.json' and self.planner_type == 'A*':
            self.step_size = 0.2
            self.global_map.bfs_depth = 3
            self.resolution = round(np.sqrt((self.step_size ** 2) / 2), 3)
            self.global_map.set_resolution(self.resolution)
        elif world_name == 'pr2bigmap.json' and self.planner_type == 'A*':
            self.step_size = 0.3
            self.global_map.bfs_depth = 1
            self.resolution = round(np.sqrt((self.step_size ** 2) / 2), 3)
            self.global_map.set_resolution(self.resolution)
        elif world_name == 'pr2doorway.json' and self.planner_type == 'D*':
            self.step_size = 0.3
            self.global_map.bfs_depth = 2
            self.resolution = round(np.sqrt((self.step_size ** 2) / 2), 3)
            self.global_map.set_resolution(self.resolution)
            self.goal_threshold = self.step_size / 4
            self.planner.edge_check_depth = 1
        elif world_name == 'pr2bigmap.json' and self.planner_type == 'D*':
            self.step_size = 0.45
            self.global_map.bfs_depth = 1
            self.resolution = round(np.sqrt((self.step_size ** 2) / 2), 3)#- self.step_size/16
            self.global_map.set_resolution(self.resolution)
            self.goal_threshold = self.step_size / 4
            self.planner.edge_check_depth = 0

        if self.planner_type == 'D*':
            # need to discretize start_state and move robot to start there
            new_start_state = self.global_map.roundPointToCell(self.start_state)
            self.move_in_path([self.start_state, new_start_state], 0)
            self.start_state = new_start_state



        self.planner.set_start_state(self.start_state)



    # Returns 1 if end of path is reached or path is empty
    # Returns 0 for successfully incrementing along path
    def move_in_path(self, path, path_idx):
        if len(path) == 0:
            print("Path is Empty")
            return 1
        elif path_idx == len(path):
            print("End of path reached")
            return 1
        else:
            sub_path = [tuple(path[path_idx]), tuple(path[path_idx + 1])]
            execute_trajectory(self.robots['pr2'], self.base_joints, sub_path, sleep=0.2)
            self.curr_state = path[path_idx]
            return 0

    def lidar_test(self):
        # Straight line trajectory to test Lidar
        start = self.start_config
        trans_step = 0.1
        movements = 500

        # Create path
        path = []
        path.append(list(start))
        state = list(start)
        for i in range(0, movements):
            new_state = np.copy(state)
            new_state[0] += trans_step
            path.append(new_state)
            state = new_state
        self.curr_path = path
        self.curr_path_idx = 0

        # keep moving along path
        while not self.move_in_path():
            # Get points from map
            new_points = self.lidar.getLidarScan(self.curr_state, self.obstacles)
            plot_points(new_points)

    def moveToGoalInst(self):
        start = self.start_config
        path = []
        path.append(list(start))
        path.append(self.goal_state)
        self.curr_path = path
        self.curr_path_idx = 0

        self.move_in_path()
        return 0

    #### For path searching ####

    def determinePlanner(self):
        if self.planner_type == 'A*':
            self.planner = Astar(self.global_map, self)
        elif self.planner_type == 'LPA*':
            self.planner = Lpastar(self.global_map, self)
        elif self.planner_type == 'D*':
            self.planner = Dstarlite(self.global_map, self)

    # checks whether state is the goal
    def is_goal_state(self, state):
        if dist(state, self.goal_state) <= self.goal_threshold:
            return True
        return False

    # Will use whatever planner is currently set while moving towards goal
    def run(self, debug=False):
        if debug:
            self.debug = True
            self.planner.set_debug(self.debug)
        self.start_time = time.time()
        self.planner.run()
        self.total_time = time.time() - self.start_time
        #print("Total time taken: " + str(self.total_time))

    ##### For Getting parameters #########
    def get_current_path(self):
        return self.curr_path

    def get_current_path_idx(self):
        return self.curr_path_idx

    def get_current_state(self):
        return self.curr_state

    def get_goal_state(self):
        return self.goal_state

    def get_trans_step_size(self):
        return self.step_size

    def get_rot_step_size(self):
        return self.rot_step

    def get_start_state(self):
        return self.start_state

    def get_trans_step_size(self):
        return self.step_size

    def get_rot_step_size(self):
        return self.rot_step

    def get_runtime(self):
        return self.total_time

    def get_total_distance(self):
        return self.planner.get_total_distance()

    ##### For Setting parameters
    # Sets goal configuration from tuple
    def set_goal_config(self, goal_config):
        self.goal_config = goal_config
        self.goal_state = self.global_map.roundPointToCell(list(goal_config))
        self.planner.updateGoal()



    ############ Private Member Functions #############

    """
    CALL THIS TO SCAN AREA AND UPDATE MAP WITH NEW OBSTACLE VALUES
    Updates Map through Process:
    1.) Gets Lidar scan from position
    2.) Puts obstacles into map representation
    RETURNS list of new obstacle points  
    """
    def update_map(self, curr_state):
        # Get lidar Scan
        obstacle_points = self.lidar.getLidarScan(curr_state, self.obstacles)

        # Update global map
        self.global_map.updateMap(obstacle_points)

        # Get novel obstacle points
        new_obs_points = self.global_map.getNewPoints()

        return new_obs_points

    # Given a list of obstacle points,
    def does_conflict_with_path(self, obs_points, path, path_idx=0):
        # Check if any of the new obstacle points come within a tolerance of the rest of the path
        # Replan if necessary
        for i in range(path_idx, len(path)):
            for obs_point in obs_points:

                # Check whether a new obstacle point will interfere with the current path
                # Check whether path is far away from the obstacle point to avoid redundant checks
                if dist(obs_point, [path[i][0], path[i][1]]) < self.update_path_threshold:
                    print("Obstacle in way of path")
                    return True
                elif dist(obs_point, [path[i][0], path[i][1]]) >= self.update_ignore_threshold:
                    print("Updated map successfully")
                    return False
