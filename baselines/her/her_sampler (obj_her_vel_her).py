import numpy as np
from baselines.her.ger_learning_method import ger_learning
from ipdb import set_trace

xxtest = 0.8
reduce_HER = 0.8/(100*100*50)

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions,env_name=None, n_GER=0,err_distance=0.05):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        # batch_size = 256

        # if rollout_batch_size == 256:
        # set_trace()

        batch_size = batch_size_in_transitions

        # if batch_size == 256: set_trace()

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        # ---------------- Added by Ray decrease her ---------------------------
        # global xxtest
        # if xxtest >= 0.3:
        #     xxtest -= reduce_HER
        # if (xxtest*100)//5==0:
        #     print('future_p:',xxtest)
        # her_indexes = np.where(np.random.uniform(size=batch_size) < xxtest)
        # ---------------- Added by Ray decrease her ---------------------------

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        ########################################################################################
        # # Replace goal with achieved goal but only for the previously-selected
        # # HER transitions (as defined by her_indexes). For the other transitions,
        # # keep the original goal.
        # future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        # transitions['g'][her_indexes] = future_ag.copy()

        ########################################################################################
        # ---------------- Added by Ray velocity-HER ---------------------------
        ######### v4 after v origin ##########
        # if batch_size == 256: set_trace()
        obs = episode_batch['o'][episode_idxs[her_indexes], future_t]
        obj_vel_abs_vecs = obs[:, 14:17]
        obj_v_vecs_norm = np.linalg.norm(obj_vel_abs_vecs, axis=1)
        obj_v_vecs_norm_reshape = obj_v_vecs_norm.reshape(-1,1)
        obj_v_vecs_norm_reshape = np.tile(obj_v_vecs_norm_reshape, (1, 3))
        e_vecs = obj_vel_abs_vecs/obj_v_vecs_norm_reshape
        alpha = np.random.uniform(size=len(her_indexes[0]))*0.2
        future_ag = [e_vecs[i]*alpha[i] for i in range(len(alpha))]
        transitions['g'][her_indexes] = transitions['ag'][her_indexes] + future_ag.copy()

        ######### v3 , if v > v_threshold ##########    untested!!!
        # set_trace()
        # obs = episode_batch['o'][episode_idxs[her_indexes], future_t]
        # # obs = transitions['o'][her_indexes]
        # obj_vel_abs_vecs = obs[:, 14:17] + obs[:, 20:23]
        #
        # obj_v_vecs_norm = np.linalg.norm(obj_vel_abs_vecs, axis=1)
        # obj_v_vecs_norm_reshape = obj_v_vecs_norm.reshape(-1, 1)
        # obj_v_vecs_norm_reshape = np.tile(obj_v_vecs_norm_reshape, (1, 3))
        #
        # e_vecs = obj_vel_abs_vecs / obj_v_vecs_norm_reshape
        #
        # alpha = np.random.uniform(size=len(her_indexes[0])) * 0.2
        # future_ag = [e_vecs[i] * alpha[i] for i in range(len(alpha))]
        #
        # index = np.where(obj_v_vecs_norm > 0.05)
        # transitions['g'][her_indexes][index] = transitions['ag'][her_indexes][index] + future_ag[index].copy()


        # ######### v2 after v ##########
        # # if batch_size == 256: set_trace()
        # obs = episode_batch['o'][episode_idxs[her_indexes], future_t]
        # # obs = transitions['o'][her_indexes]
        # # obj_vel_abs_vecs = obs[:, 14:17] + obs[:, 20:23]
        # obj_vel_abs_vecs = obs[:, 14:17] - obs[:, 20:23]
        #
        # obj_v_vecs_norm = np.linalg.norm(obj_vel_abs_vecs, axis=1)
        # obj_v_vecs_norm_reshape = obj_v_vecs_norm.reshape(-1,1)
        # obj_v_vecs_norm_reshape = np.tile(obj_v_vecs_norm_reshape, (1, 3))
        #
        # e_vecs = obj_vel_abs_vecs/obj_v_vecs_norm_reshape
        #
        # alpha = np.random.uniform(size=len(her_indexes[0]))*0.2
        # future_ag = [e_vecs[i]*alpha[i] for i in range(len(alpha))]
        #
        # transitions['g'][her_indexes] = transitions['ag'][her_indexes] + future_ag.copy()

        ######### v1 current v ##########
        # set_trace()
        # # future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        ## vel_her = her_indexes[0] + 2
        ## vel_her[np.where(vel_her>T)] = T
        # obs = transitions['o'][her_indexes]
        # obj_vel_abs_vecs = obs[:, 14:17] + obs[:, 20:23]
        # # obj_vel_abs_vecs = obs[:, 14:17] - obs[:, 20:23]
        #
        # obj_v_vecs_norm = np.linalg.norm(obj_vel_abs_vecs, axis=1)
        # obj_v_vecs_norm_reshape = obj_v_vecs_norm.reshape(-1,1)
        # obj_v_vecs_norm_reshape = np.tile(obj_v_vecs_norm_reshape, (1, 3))
        #
        # e_vecs = obj_vel_abs_vecs/obj_v_vecs_norm_reshape
        #
        # alpha = np.random.uniform(size=len(her_indexes[0]))*0.2
        # future_ag = [e_vecs[i]*alpha[i] for i in range(len(alpha))]
        #
        # transitions['g'][her_indexes] = transitions['ag'][her_indexes] + future_ag.copy()
        # ---------------- Added by Ray velocity-HER ---------------------------
        ########################################################################################


        ########################################################################################
        # ---------------- Added by Ray PBJ-HER ---------------------------
        ### if batch_size == 256: set_trace()
        # starting_ags = episode_batch['ag'][episode_idxs[her_indexes]][:, 0, :]
        # starting_ags = np.tile(starting_ags, (1, T)).reshape(-1, T, 3)
        # remaining_ags = episode_batch['ag'][episode_idxs[her_indexes]][:, 1:, :]
        # delta_ags = remaining_ags - starting_ags
        # delta_ags_norm = np.linalg.norm(delta_ags, axis=2)
        # move_or_not = [any(traj > 0.05) for traj in delta_ags_norm]
        # meaningless_traj_index = np.where(np.array(move_or_not)==False)
        #
        # # starting_ags = episode_batch['ag'][episode_idxs][:, 0, :]
        # # starting_ags = np.tile(starting_ags, (1, T)).reshape(-1, T, 3)
        # # remaining_ags = episode_batch['ag'][episode_idxs][:, 1:, :]
        # # delta_ags = remaining_ags - starting_ags
        # # delta_ags_norm = np.linalg.norm(delta_ags, axis=2)
        # # move_or_not = [any(traj > 0.05) for traj in delta_ags_norm]
        # # meaningless_traj_index = np.where(np.array(move_or_not)==False)
        #
        #
        # future_offset_ag = np.random.uniform(size=batch_size) * future_offset # the ag should be between in t_samples and future_t
        # future_offset_ag = future_offset_ag.astype(int)
        # future_t_ag = (t_samples + 1 + future_offset_ag)[meaningless_traj_index]
        #
        # future_grip = episode_batch['o'][episode_idxs[meaningless_traj_index], future_t_ag][:,:3]
        # transitions['g'][meaningless_traj_index] = future_grip.copy()
        # transitions['ag'][meaningless_traj_index] = future_grip.copy()
        # transitions['ag_2'][meaningless_traj_index] = future_grip.copy()
        # transitions['o'][meaningless_traj_index][:, 3:6] = future_grip.copy()
        # transitions['o_2'][meaningless_traj_index][:, 3:6] = future_grip.copy()


        # future_offset_ag = np.random.uniform(size=batch_size) * future_offset # the ag should be between in t_samples and future_t
        # future_offset_ag = future_offset_ag.astype(int)
        # future_t_ag = (t_samples + 1 + future_offset_ag)[her_indexes]
        #
        # future_grip = episode_batch['o'][episode_idxs[her_indexes], future_t_ag][:,:3]
        # transitions['ag'][her_indexes] = future_grip.copy()
        # transitions['ag_2'][her_indexes] = future_grip.copy()
        # transitions['o'][her_indexes][:, 3:6] = future_grip.copy()
        # transitions['o_2'][her_indexes][:, 3:6] = future_grip.copy()
        # ---------------- Added by Ray PBJ-HER ---------------------------
        ########################################################################################

        # create a new dict to store all the original, KER, HER, AGER data.
        all_transitions = {key: transitions[key].copy()
                           for key in episode_batch.keys()}

        # ----------------Goal-augmented ER--------------------------- 
        if n_GER:
            # when n_GER != 0
            for _ in range (n_GER):
                PER_transitions = {key: transitions[key].copy()
                                   for key in episode_batch.keys()}
                ger_machine = ger_learning(env_name = env_name,err_distance=err_distance)
                PER_indexes= np.array((range(0,batch_size)))
                HER_KER_future_ag = PER_transitions['g'][PER_indexes].copy()
                PER_future_g = ger_machine.process_goals(HER_KER_future_ag.copy())
                PER_transitions['g'][PER_indexes] = PER_future_g.copy()
                for key in episode_batch.keys():
                    all_transitions[key] = np.vstack([all_transitions[key], PER_transitions[key].copy()])
        # -----------------------End--------------------------- 

        # After GER, the minibatch size enlarged
        batch_size = batch_size * (1+n_GER)
        batch_size_in_transitions =batch_size

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in all_transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: all_transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        all_transitions['r'] = reward_fun(**reward_params)

        all_transitions = {k: all_transitions[k].reshape(batch_size, *all_transitions[k].shape[1:])
                           for k in all_transitions.keys()}

        assert(all_transitions['u'].shape[0] == batch_size_in_transitions)
        return all_transitions

    return _sample_her_transitions
