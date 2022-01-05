import numpy as np
from map import Map
from queue import PriorityQueue
from utils import dist
from plot_utils import plot_points, destroy_points

class Node:
    def __init__(self, state, id_in, parent_id_in, h_in, g_in):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.id = id_in
        self.parent_id = parent_id_in
        self.h = round(h_in, 2)
        self.g = round(g_in, 2)
        self.f = round(self.h + self.g, 2)

    def getState(self):
        return [self.x, self.y, self.theta]

    def getTheta(self):
        return self.theta

    def getY(self):
        return self.y

    def getX(self):
        return self.x

    def getId(self):
        return self.id

    def getF(self):
        return self.f

    def getParentId(self):
        return self.parent_id

    def getG(self):
        return self.g

class Astar:
    def __init__(self, global_map, robot):

        # External Parameters
        self.global_map = global_map
        self.myrobot = robot
        self.goal_state = None
        self.start_state = None
        self.curr_state = None

        #Global Astar parameters
        self.trans_step_size = self.myrobot.get_trans_step_size()
        self.rot_step_size = self.myrobot.get_rot_step_size()

        # Global metrics
        self.total_distance = 0

        # Debug tracker
        self.debug = False


    # starts repeated astar
    def run(self):
        # Holding visuals for the demos and debugging
        obstacle_marker_ids = []
        path_marker_ids = []


        # Start by getting first view of map
        new_obs_points = self.myrobot.update_map(self.curr_state) # We dont care about new obstacle points here
        if self.debug:
            obstacle_marker_ids.extend(plot_points(new_obs_points))

        # Run astar to get first initial path on the map
        curr_path = self.astar(self.curr_state)
        curr_path_idx = 0
        if self.debug:
            path_marker_ids = plot_points(curr_path, color=(0, 1, 0, 1))
        # move along path until either:
        # 1.) The path is obstructed by an obstacle -> then replan
        # 2.) The goal is reached -> end of journey
        while 1:
            # update the map
            new_obs_points = self.myrobot.update_map(self.curr_state)
            if self.debug:
                obstacle_marker_ids.extend(plot_points(new_obs_points))

            # check for obstacles obstructing path
            if self.myrobot.does_conflict_with_path(new_obs_points, curr_path, curr_path_idx):
                # if obstacle is in path then replan from current state
                if self.debug:
                    destroy_points(path_marker_ids)

                curr_path = self.astar(curr_path[curr_path_idx])
                curr_path_idx = 0
                if self.debug:
                    path_marker_ids = plot_points(curr_path, color=(0, 1, 0, 1))

            # move in path
            self.myrobot.move_in_path(curr_path, curr_path_idx)
            self.total_distance += dist(curr_path[curr_path_idx], curr_path[curr_path_idx + 1])
            curr_path_idx += 1
            self.curr_state = curr_path[curr_path_idx]

            # check if goal state is reached
            if self.myrobot.is_goal_state(self.curr_state):
                break
        #print("Total distance travelled: " + str(self.total_distance))
        return 0

    def astar(self, start_state):
        # local Astar parameters
        frontier = PriorityQueue()
        openSet = {}
        exploredSet = {}
        nodeSet = {}
        isActive = True
        unique_id = 0

        # Create starting node
        start_node = Node(start_state, unique_id, -1, dist(start_state, self.goal_state), 0)
        unique_id += 1

        # Insert starting node to frontier, openSet, and nodeSet
        frontier.put((start_node.getF(), start_node.getId(), start_node))
        openSet[tuple(start_node.getState())] = start_node.getF()
        nodeSet[start_node.getId()] = start_node

        curr_state = []
        while not frontier.empty():
            # grab node from top of pq
            curr_tup = frontier.get()
            curr_node = curr_tup[2]
            curr_state = curr_node.getState()

            ######### Generate Successors #############
            neighbors = []

            # 1
            new_state = [x for x in curr_state]
            new_state[0] += self.trans_step_size
            new_state = self.global_map.roundPointToCell(new_state)
            new_node = Node(new_state, unique_id, curr_node.getId(), dist(new_state, self.goal_state), curr_node.getG() + dist(curr_state, new_state))
            nodeSet[unique_id] = new_node
            neighbors.append(new_node)
            unique_id += 1

            # 2
            new_state2 = [x for x in curr_state]
            new_state2[0] -= self.trans_step_size
            new_state2 = self.global_map.roundPointToCell(new_state2)
            new_node2 = Node(new_state2, unique_id, curr_node.getId(), dist(new_state2, self.goal_state), curr_node.getG() + dist(curr_state, new_state2))
            nodeSet[unique_id] = new_node2
            neighbors.append(new_node2)
            unique_id += 1

            # 3
            new_state3 = [x for x in curr_state]
            new_state3[1] += self.trans_step_size
            new_state3 = self.global_map.roundPointToCell(new_state3)
            new_node3 = Node(new_state3, unique_id, curr_node.getId(), dist(new_state3, self.goal_state), curr_node.getG() + dist(curr_state, new_state3))
            nodeSet[unique_id] = new_node3
            neighbors.append(new_node3)
            unique_id += 1

            # 4
            new_state4 = [x for x in curr_state]
            new_state4[1] -= self.trans_step_size
            new_state4 = self.global_map.roundPointToCell(new_state4)
            new_node4 = Node(new_state4, unique_id, curr_node.getId(), dist(new_state4, self.goal_state), curr_node.getG() + dist(curr_state, new_state4))
            nodeSet[unique_id] = new_node4
            neighbors.append(new_node4)
            unique_id += 1

            # 5
            new_state5 = [x for x in curr_state]
            new_state5[2] += self.rot_step_size
            if new_state5[2] > np.pi:
                new_state5[2] = (-np.pi + (new_state5[2] - np.pi))
            new_node5 = Node(new_state5, unique_id, curr_node.getId(), dist(new_state5, self.goal_state), curr_node.getG() + dist(curr_state, new_state5))
            nodeSet[unique_id] = new_node5
            neighbors.append(new_node5)
            unique_id += 1

            # 6
            new_state6 = [x for x in curr_state]
            new_state6[2] -= self.rot_step_size
            if new_state6[2] < -np.pi:
                new_state6[2] = (np.pi - (np.abs(new_state6[2] - np.pi)))
            new_node6 = Node(new_state6, unique_id, curr_node.getId(), dist(new_state6, self.goal_state), curr_node.getG() + dist(curr_state, new_state6))
            nodeSet[unique_id] = new_node6
            neighbors.append(new_node6)
            unique_id += 1

            ############ ANALYZE SUCCESSORS ##############
            # check for neighbors being within tolerance of the goal state
            for neighbor in neighbors:
                if self.myrobot.is_goal_state(neighbor.getState()):
                    path = self.backtrack(neighbor, nodeSet)
                    path.reverse()
                    return path

            # Check neighbors for being obstacles
            for neighbor in neighbors:
                pass
                if self.global_map.isObstacle(neighbor.getState()):
                    neighbors.remove(neighbor)

            # check all neighbors to see if they have already been explored before
            for neighbor in neighbors:
                if openSet.get(tuple(neighbor.getState())) is not None:
                    if openSet.get(tuple(neighbor.getState())) <= neighbor.getF():
                        continue
                if exploredSet.get(tuple(neighbor.getState())) is not None:
                    if exploredSet.get(tuple(neighbor.getState())) <= neighbor.getF():
                        continue

                # We want to add the node to the pq and open set now
                frontier.put((neighbor.getF(), neighbor.getId(), neighbor))
                openSet[tuple(neighbor.getState())] = neighbor.getF()

            # Add current node to the exploredSet
            openSet[tuple(curr_state)] = None
            exploredSet[tuple(curr_state)] = curr_node.getF()

            #### End of while in frontier

        # if this point is reached, then no solution was found
        print("No solution found")
        return []


    def backtrack(self, final_node, nodeSet):
        path = []
        while final_node.getId() != 0:
            path.append(final_node.getState())
            final_node = nodeSet[final_node.getParentId()]
        path.append(final_node.getState())
        return path

    def updateGoal(self):
        self.goal_state = self.myrobot.get_goal_state()

    def set_start_state(self, state):
        self.start_state = state
        self.curr_state = state

    def set_debug(self, debug):
        self.debug = debug

    def get_total_distance(self):
        return self.total_distance

