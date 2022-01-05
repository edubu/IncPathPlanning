import math

import numpy as np
from map import Map
from queue import PriorityQueue
from utils import dist
from plot_utils import plot_points, destroy_points

class Node:
    def __init__(self, state, id_in, pred_id, h_in, g_in, rhs_in=0):
        self.state = state
        self.id = id_in
        self.predecessors = [pred_id]
        self.h = round(h_in, 2)
        self.g = round(g_in, 2)
        self.f = round(self.h + self.g, 2)
        self.rhs = round(rhs_in,2)
        self.k1 = min(self.rhs, self.g) + self.h
        self.k2 = min(self.h, self.rhs)

    def getState(self):
        return self.state

    def __eq__(self, other):
        return (self.state == other.state)

    def __gt__(self, other):
        if self.k1 == other.k1:
            return self.k2 < other.k2
        return self.k1 < other.k1

    def add_to_pred(self, unique_id):
        self.predecessors.append(unique_id)

    def calculateKey(self):
        self.k1 = min(self.rhs, self.g) + self.h
        self.k2 = min(self.h, self.rhs)
        return [self.k1, self.k2]

    def getId(self):
        return self.id

    def updateRHS(self, nodeSet):
        min_rhs = np.inf
        for i in self.predecessors:
            curr_rhs = nodeSet[i].g + dist(self.state, nodeSet[i].getState())
            if (min_rhs < curr_rhs):
                min_rhs = curr_rhs
        self.rhs = min_rhs

class Lpastar:
    def __init__(self, global_map, robot):
        # External Parameters
        self.global_map = global_map
        self.myrobot = robot
        self.start_state = None

        # Global lpastar parameters
        self.trans_step_size = self.myrobot.get_trans_step_size()
        self.rot_step_size = self.myrobot.get_rot_step_size()

        # Global metrics
        self.total_distance = 0

        # Debug tracker
        self.debug = False

        # Priority Queue
        self.pq = PriorityQueue()
        self.openSet = {} #Key: state as a tuple, value is the id
        self.nodeSet = {} #key: is the id, value is the Node
        self.unique_id = 0
        self.goal_state = None
        self.goal_node = None


    def set_start_state(self, state):
        self.start_state = state

    def updateNode(self, node):
        if (node != self.start):
            node.rhs = float('inf')
            pred = self.nodeSet[node.parent_id]
            node.rhs = min(node.rhs, pred.g + pred.getCostTo(node))

        if self.openSet.get(node.getState()) is not None:
            self.pq.remove(node)
        if (node.g != node.rhs):
            self.pq.insert(node, self.calculateKey(node))

    # returns true if key1 is less than key2
    def compareKeys(self, key1, key2):
        if key1[0] == key2[0]:
            return key1[1] < key2[1]
        return key1[0] < key2[0]


    def computeShortestPath(self):
        while self.compareKeys(self.pq.get().calculateKey(), self.goal_node.calculateKey()) or (self.goal_node.rhs != self.goal_node.g):
            curr_node = self.pq.pop()
            curr_state = curr_node.getState()

            ######### Generate Successors #############
            def genSuccessors():
                neighbors = []

                # 1
                new_state = [x for x in curr_state]
                new_state[0] += self.trans_step_size
                new_state = self.global_map.roundPointToCell(new_state)
                if self.openSet.get(tuple(new_state)) is None: #ie Node doesn't exist:
                    new_node = Node(new_state, self.unique_id, curr_node.getId(), dist(new_state, self.goal_state), curr_node.getG() + dist(curr_state, new_state))
                    self.nodeSet[self.unique_id] = new_node
                    neighbors.append(new_node)
                    self.unique_id += 1
                    new_node.updateRHS()
                else:
                    id = self.openSet[tuple(new_state)]
                    self.nodeSet[id].add_to_pred(curr_node.getId())
                    self.nodeSet[id].updateRHS()

                # 2
                new_state2 = [x for x in curr_state]
                new_state2[0] -= self.trans_step_size
                new_state2 = self.global_map.roundPointToCell(new_state2)
                if self.openSet.get(tuple(new_state)) is None:  # ie Node doesn't exist:
                    new_node2 = Node(new_state2, self.unique_id, curr_node.getId(), dist(new_state2, self.goal_state), curr_node.getG() + dist(curr_state, new_state2))
                    self.nodeSet[self.unique_id] = new_node2
                    neighbors.append(new_node2)
                    self.unique_id += 1
                    new_node2.updateRHS()
                else:
                    id = self.openSet[tuple(new_state2)]
                    self.nodeSet[id].add_to_pred(curr_node.getId())
                    self.nodeSet[id].updateRHS()

                # 3
                new_state3 = [x for x in curr_state]
                new_state3[1] += self.trans_step_size
                new_state3 = self.global_map.roundPointToCell(new_state3)
                if self.openSet.get(tuple(new_state)) is None:  # ie Node doesn't exist:
                    new_node3 = Node(new_state3, self.unique_id, curr_node.getId(), dist(new_state3, self.goal_state), curr_node.getG() + dist(curr_state, new_state3))
                    self.nodeSet[self.unique_id] = new_node3
                    neighbors.append(new_node3)
                    self.unique_id += 1
                    new_node3.updateRHS()
                else:
                    id = self.openSet[tuple(new_state3)]
                    self.nodeSet[id].add_to_pred(curr_node.getId())
                    self.nodeSet[id].updateRHS()

                # 4
                new_state4 = [x for x in curr_state]
                new_state4[1] -= self.trans_step_size
                new_state4 = self.global_map.roundPointToCell(new_state4)
                if self.openSet.get(tuple(new_state)) is None:  # ie Node doesn't exist:
                    new_node4 = Node(new_state4, self.unique_id, curr_node.getId(), curr_node.getId(), dist(new_state4, self.goal_state), curr_node.getG() + dist(curr_state, new_state4))
                    self.nodeSet[self.unique_id] = new_node4
                    neighbors.append(new_node4)
                    self.unique_id += 1
                    new_node4.updateRHS()
                else:
                    id = self.openSet[tuple(new_state4)]
                    self.nodeSet[id].add_to_pred(curr_node.getId())
                    self.nodeSet[id].updateRHS()

                # 5
                new_state5 = [x for x in curr_state]
                new_state5[2] += self.rot_step_size
                if new_state5[2] > np.pi:
                    new_state5[2] = (-np.pi + (new_state5[2] - np.pi))
                if self.openSet.get(tuple(new_state)) is None:  # ie Node doesn't exist:
                    new_node5 = Node(new_state5, self.unique_id, curr_node.getId(), dist(new_state5, self.goal_state), curr_node.getG() + dist(curr_state, new_state5))
                    self.nodeSet[self.unique_id] = new_node5
                    neighbors.append(new_node5)
                    self.unique_id += 1
                    new_node5.updateRHS()
                else:
                    id = self.openSet[tuple(new_state5)]
                    self.nodeSet[id].add_to_pred(curr_node.getId())
                    self.nodeSet[id].updateRHS()

                # 6
                new_state6 = [x for x in curr_state]
                new_state6[2] -= self.rot_step_size
                if new_state6[2] < -np.pi:
                    new_state6[2] = (np.pi - (np.abs(new_state6[2] - np.pi)))
                if self.openSet.get(tuple(new_state)) is None:  # ie Node doesn't exist:
                    new_node6 = Node(new_state6, self.unique_id, curr_node.getId(), dist(new_state6, self.goal_state), curr_node.getG() + dist(curr_state, new_state6))
                    self.nodeSet[self.unique_id] = new_node6
                    neighbors.append(new_node6)
                    self.unique_id += 1
                    new_node6.updateRHS()
                else:
                    id = self.openSet[tuple(new_state6)]
                    self.nodeSet[id].add_to_pred(curr_node.getId())
                    self.nodeSet[id].updateRHS()

                return neighbors

            successors = genSuccessors()
            # check successors for being obstacles
            for successor in successors:
                if self.global_map.isObstacle(successor.getState()):
                    successors.remove(successor)
                else:
                    self.openSet[successor.getState()] = successor.getId()

            if (curr_node.g > curr_node.rhs):
                curr_node.g = curr_node.rhs
                for s in successors:
                    self.updateNode(s)
            else:
                curr_node.g = float('inf')
                self.updateNode(curr_node)
                for s in successors:
                    self.updateNode(s)


        ### END OF WHILE ####
        # means that we should backtrack to find path






    def run(self, debug=False):
        self.debug = debug

        # Setup
        self.start = Node(self.start_state, self.unique_id, dist(self.start_state, self.goal_state), 0)
        self.unique_id += 1
        self.goal_node = Node(self.goal_state, -1, 0, float('inf'), 0)
        self.start.add_to_predecessors(-1)
        self.pq.put(self.start)
        self.openSet[self.start_state] = self.start.getId()


        # Holding visuals for the demos and debugging
        obstacle_marker_ids = []
        path_marker_ids = []

        # Start by getting first view of map
        # We dont care about new obstacle points here
        new_obs_points = self.myrobot.update_map(self.curr_state)
        if self.debug:
            obstacle_marker_ids.extend(plot_points(new_obs_points))

        while 1:
            curr_path = self.computeShortestPath()
            # TODO: This function should return the "current path"

            # Bring the new obstacle points
            new_obs_points = self.myrobot.update_map(self.curr_state)
            if self.debug:
                obstacle_marker_ids.extend(plot_points(new_obs_points))

            for obs in new_obs_points:
                #TODO: Add more neighbors to the obs points
                xy_obs_0 = (obs[1] + self.global_map.resolution,obs[2] + self.global_map.resolution, 0)
                xy_obs_pih = (obs[1] + self.global_map.resolution, obs[2] + self.global_map.resolution, math.pi/2.0)
                xy_obs_pi = (obs[1] + self.global_map.resolution, obs[2] + self.global_map.resolution, math.pi)
                xy_obs_twopi = (obs[1] + self.global_map.resolution, obs[2] + self.global_map.resolution, 2*math.pi)
                xy_obs = [xy_obs_0, xy_obs_pih, xy_obs_pi, xy_obs_twopi]

                for orient in xy_obs:
                    if self.openSet.get(orient) is not None: #i.e there is a node on an obstacle:
                        id = self.openSet.get(orient)
                        nodeChange = self.nodeSet[id]
                        nodeChange.g = np.inf
                        self.updateNode(nodeChange)


            # TODO: Need a curr-path for this
            # Move the robot and calculate the distance..
            self.myrobot.move_in_path(curr_path, curr_path_idx)
            self.total_distance += dist(curr_path[curr_path_idx],
                                        curr_path[curr_path_idx + 1])

        return

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
