import numpy as np
from ipdb import set_trace
import math
import copy

PI = math.pi

zone_min = np.array([1.15, 0.55])
zone_max = np.array([1.55, 0.95])

# zone_min = np.array([1.2, 0.6])
# zone_max = np.array([1.5, 0.9])
original_outside_count = 0

def get_boundary(episode):

    global original_outside_count
    max_ag_x = episode['ag'][0][:,0].max() # get the ag x boundary in the sequence
    min_ag_x = episode['ag'][0][:,0].min()

    max_ag_y = episode['ag'][0][:,1].max() # get the ag y boundary in the sequence
    min_ag_y = episode['ag'][0][:,1].min()

    g_x, g_y = episode['g'][0][0,:2]   # get the goal pos

    x_range_min = min(max_ag_x, min_ag_x, g_x)
    x_range_max = max(max_ag_x, min_ag_x, g_x)
    y_range_min = min(max_ag_y, min_ag_y, g_y)
    y_range_max = max(max_ag_y, min_ag_y, g_y)

    min_xy = zone_min - (x_range_min, y_range_min)
    max_xy = zone_max - (x_range_max, y_range_max)

    original_traj_outside = False
    if (max_xy-min_xy<0).any():       # this is because the object is moved out of the zone.
        original_outside_count+=1
        print('original_traj_outside:', original_outside_count)
        original_traj_outside = True

    return min_xy, max_xy, original_traj_outside

def translation(episode, n_translation):
    tran_episodes = []
    tran_episodes.append(episode)
    min_delta_xy, max_delta_xy, original_traj_outside = get_boundary(episode)
    new_episode = copy.deepcopy(episode)

    if original_traj_outside == True:   # we don't translate the outside trajectories.
        return tran_episodes

    for i in range(n_translation):

        # set_trace()

        outside = True
        count = 0
        while outside:
            delta_x = np.random.uniform(min_delta_xy[0],max_delta_xy[0], size=1)
            delta_y = np.random.uniform(min_delta_xy[1],max_delta_xy[1], size=1)
            delta_xy = np.concatenate((delta_x,delta_y))

            # only need to change g,ag and o in ['o', 'u', 'g', 'ag', 'info_is_success']
            # change g and ag
            new_episode['g'][0][:,:2] = episode['g'][0][:,:2] + delta_xy
            new_episode['ag'][0][:,:2] = episode['ag'][0][:,:2] + delta_xy

            # in case the goal is outside
            outside = False
            outside = (new_episode['g'][0][:, :2] < zone_min).any() or outside
            outside = (new_episode['g'][0][:, :2] > zone_max).any() or outside
            outside = (new_episode['ag'][0][:, :2] < zone_min).any() or outside
            outside = (new_episode['ag'][0][:, :2] > zone_max).any() or outside
            # assert outside == False
            if outside: print('translation_outside:', count)
            count += 1
            # if count>100: set_trace()

        # change o
        o_len = len(new_episode['o'][0])
        o = new_episode['o'][0]
        if o_len == 10:     # observation without object
            # grip pos o[0:3]
            o[:,:2] = o[:,:2] + delta_xy
        elif o_len >= 25:     # observation with object

            # grip_pos           0-3
            # object_pos         3-6
            # object_rel_pos     6-9
            # gripper_state      9-11
            # object_rot         11-14
            # object_velp        14-17
            # object_velr        17-20
            # grip_velp          20-23
            # gripper_vel        23-25

            o[:,:2] = o[:,:2] + delta_xy  # grip pos o[0:3]
            o[:,3:5] = o[:,3:5] + delta_xy  # obj pos o[3:6]

            if  o_len == 31:  # obstacle state
                o[:,25:27] = o[:,25:27] + delta_xy  # 1. pos transform
        # new_episode['o'][0] = o
        tran_episodes.append(new_episode)
    return  tran_episodes

def translation_learning(episodes, n_translation, n_KER):
    # set_trace()
    results = []
    if n_KER:
        for episode in episodes:
            tran_episodes = translation(episode, n_translation)
            for tran_episode in tran_episodes:
                results.append(tran_episode)
    else:
        tran_episodes = translation(episodes, n_translation)
        for tran_episode in tran_episodes:
            results.append(tran_episode)

    return results

