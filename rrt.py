import sim
import pybullet as p
import numpy as np
import random
import math
from collections import deque
from queue import PriorityQueue

MAX_ITERS = 10000
delta_q = 0.3

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)

def SemiRandomSample(steer_goal_p, q_goal): 
    """  returns q_goal with probability steer_goal_p, and a random sample with probability 1-steer_goal_p """
    prob=np.random.choice([0,1],1,p=[steer_goal_p,1-steer_goal_p])

    if prob == 0:
        q_rand= q_goal

    elif prob ==1:
         q_rand=np.random.random_sample((6))*2*np.pi-np.pi

    return q_rand

def Nearest(vertices, edges, q_rand):
    #visited = [] # List for visited nodes.
    #queue = PriorityQueue()     #Initialize a queue

    q_nearest = None
    minDist = float("inf")

    for v in vertices:
        dist = Distance(v, q_rand)
        if dist < minDist:
            minDist = dist
            q_nearest = v

    return q_nearest

    """queue.put(q_rand)
    while queue.empty()==False:          # Creating loop to visit each node
        m = queue.get() 

        for neighbour in edges[m]:
          if neighbour not in visited:
            visited.append(neighbour)
            queue.append(neighbour)"""

def Steer(q_nearest, q_rand, delta_q):
    distance= Distance(q_rand, q_nearest)
    if distance <= delta_q:
        q_new=q_rand
    else:
        dir_v= q_rand-q_nearest
        #dir_v= dir_v/Distance(q_nearest, q_rand)
        q_new = q_nearest+ (dir_v)*delta_q
    return q_new

def Distance(a, b): #L1 Distance
    L1=sum(abs(val1-val2) for val1, val2 in zip(a,b))
    result=[]

    for i in range(len(a)):
        L1=abs(a[i]-b[i])
        result.append(min(2*np.pi-L1, L1))
    
    return sum(result)
    

def BFS(vertices, edges, q_init, q_goal):
    path_list = [[q_init]]
    path_index = 0
    # To keep track of previously visited nodes
    previous_nodes = [q_init]
    
    
    while path_index < len(path_list):
        current_path = path_list[path_index]
        last_node = current_path[-1]


        next_nodes_indices = [i for i, v in enumerate(edges) if np.all(v[0] == last_node) or np.all(v[1] == last_node)]
        next_nodes=[]
        for index in next_nodes_indices:
          if np.all(edges[index][0]==last_node):
            next_nodes.append(edges[index][1])
          else: 
            next_nodes.append(edges[index][0])

        for next_node in next_nodes:
            if equal(next_node, q_goal):
              current_path.append(q_goal)
              return current_path
            if np.any(next_node==previous_nodes) == False:

                new_path = current_path[:]
                new_path.append(next_node)
                path_list.append(new_path)
                # To avoid backtracking
                previous_nodes.append(next_node)
        # Continue to next path in list
        path_index += 1
    # No path is found
    return []

def equal(node, goal):
    flag=True
    for i in range(len(node)):
        if abs(node[i]-goal[i])>1e-5:
            flag=False
    return flag




def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= TODO: Problem 3 ========
    # Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)
    E= []
    V= np.empty(shape=[0, 6])
    V= np.append(V, [q_init], axis=0)

    for i in range(MAX_ITERS):
        q_rand=SemiRandomSample(steer_goal_p, q_goal)
        q_nearest=Nearest(V, E, q_rand)
        q_new=Steer(q_nearest, q_rand, delta_q)

        if env.check_collision(q_new)==False:

            if np.any(E==q_new)==False:
                V= np.append(V, [q_new], axis=0)
            if len([i for i, v in enumerate(E) if np.all(v[0] == q_nearest) and np.all(v[1] == q_new)])==0:
                E.append((q_nearest, q_new)) 
            visualize_path(q_nearest, q_new, env)
            if Distance(q_new, q_goal) < delta_q:
                if np.any(E==q_goal)==False:
                    V= np.append(V, [q_goal], axis=0)
                if len([i for i, v in enumerate(E) if np.all(v[0] == q_new) and np.all(v[1] == q_goal)])==0:
                    E.append((q_new, q_goal))
                path=BFS(V, E, q_init, q_goal)
                return path
    # ==================================
    return None

def execute_path(path_conf, env):
    # ========= TODO: Problem 3 ========
    # 1. Execute the path while visualizing the location of joint 5 
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, step the simulation, close gripper)
    # 3. Return the robot to original location by retracing the path 
    if len(path_conf) > 0 :

        sphere=[]
        for configuration in path_conf:
            sphere.append(sim.SphereMarker(p.getLinkState(env.robot_body_id, 9)[0]))
            env.move_joints(configuration)

        env.open_gripper()
        env.step_simulation(1)
        env.close_gripper()

        for configuration in reversed(path_conf):
            sphere.append(sim.SphereMarker(p.getLinkState(env.robot_body_id, 9)[0]))
            env.move_joints(configuration)
    else:
        print("No path found")

    # ==================================
    return None