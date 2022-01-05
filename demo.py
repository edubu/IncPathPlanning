import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name, get_aabb, get_aabbs
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from robot import Robot

# starts demo
def main():
    # Intialize PyBullet
    connect(use_gui=True)

    # Intialize the Robot
    planner_type = 'A*'
    robot = Robot(planner_type)
    robot.load_world('pr2bigmap.json')

   # Set goal configuration for bigmap
    bigmap_goal_config = (7, 7, np.pi / 2)
    doorway_goal_config = (1.8329999999999997, -1.3, -np.pi)
    debug_goal_config = (4, 4, np.pi/2)
    robot.set_goal_config(bigmap_goal_config)

    # Go to goal state
    robot.run(debug=False)
    time_taken = robot.get_runtime()
    distance_travelled = robot.get_total_distance()
    print(planner_type + " distance travelled " + str(distance_travelled))
    print(planner_type + " time taken: " + str(time_taken))

    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == "__main__":
    main()