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


    # ### V2  obstacle her
    # def _sample_her_transitions(episode_batch, batch_size_in_transitions,env_name=None, n_GER=0, grade_GER=0, err_distance=0.05):
    #     """episode_batch is {key: array(buffer_size x T x dim_key)}
    #     """
    #     T = episode_batch['u'].shape[1]
    #     rollout_batch_size = episode_batch['u'].shape[0]
    #     # batch_size = 256
    #
    #     # if rollout_batch_size >= 60:
    #     #     set_trace()
    #
    #     batch_size = batch_size_in_transitions
    #
    #     # Select which episodes and time steps to use.
    #     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #
    #
    #     # set_trace()
    #     ########### meanningful traj? ##########
    #     # selected_episodes = {}
    #     selected_episodes = {key: episode_batch[key][episode_idxs].copy() for key in episode_batch.keys()}
    #     ag = selected_episodes['ag']
    #     delta_movement = np.linalg.norm(ag[:,1:] - ag[:,0:1], axis=2)  # compare with the object starting pos
    #     m_less_idx = []
    #     m_ful_idx = []
    #     for i,eps_delt in enumerate(delta_movement):
    #         if any(eps_delt < 0.05):  # if the object is not moved
    #             m_less_idx.append(i)
    #             # print('meaningless')
    #         else:
    #             m_ful_idx.append(i)
    #
    #     if m_ful_idx != []:
    #         set_trace()
    #
    #     # set_trace()
    #     if m_less_idx != []:
    #         # Select which episodes and time steps to use.
    #         episode_idxs_ = episode_idxs[m_less_idx]
    #         t_samples = np.random.randint(T-2, size=len(m_less_idx))
    #         transitions = {key: episode_batch[key][episode_idxs_, t_samples].copy()
    #                        for key in episode_batch.keys()}
    #
    #         her_indexes = np.where(np.random.uniform(size=len(m_less_idx)) < future_p)
    #         future_offset = np.random.uniform(size=len(m_less_idx)) * (T - t_samples)
    #         future_offset = future_offset.astype(int)
    #         future_t_ag = (t_samples + 1 + future_offset)
    #
    #         future_offset = np.random.uniform(size=len(m_less_idx)) * (T - future_t_ag)
    #         future_offset = future_offset.astype(int)
    #         future_t_g = (future_t_ag + future_offset)
    #
    #         # obstacle
    #
    #         # grip_curr = episode_batch['o'][episode_idxs_[her_indexes], t_samples[her_indexes]][:,3]
    #         # grip_curr_v = episode_batch['o'][episode_idxs_[her_indexes], t_samples[her_indexes]][:,3]
    #         # grip_next = episode_batch['o_2'][episode_idxs_[her_indexes], t_samples[her_indexes]][:,3]
    #
    #         ### fake ag and g in both curr and next step
    #         future_ag = episode_batch['o'][episode_idxs_[her_indexes], future_t_ag[her_indexes]][:,:3]
    #         future_g = episode_batch['o'][episode_idxs_[her_indexes], future_t_g[her_indexes]][:,:3]
    #
    #         z_ground = episode_batch['ag'][0,0,-1]
    #         dis_z = future_ag[:,-1] - z_ground
    #
    #         ### translate to the ground
    #         future_ag[:, -1] -= dis_z
    #         future_g[:,-1]   -= dis_z
    #
    #
    #         # grip_pos = transitions['o'][her_indexes][:, :3]
    #         # object_pos = transitions['o'][her_indexes][:, 3:6]
    #         # object_rel_pos = transitions['o'][her_indexes][:, 6:9]
    #         # # object_velp = transitions['o'][her_indexes][:, 14:17]   ### no need, cause we assume that obj_v = 0
    #         # # grip_velp = transitions['o'][her_indexes][:, 20:23]
    #         #
    #         # grip_pos[:, -1] -= dis_z
    #         # object_pos = future_ag
    #         # object_rel_pos = object_pos - grip_pos
    #
    #         # set_trace()
    #
    #         ## ['o']
    #         o = transitions['o'][her_indexes]
    #
    #         grip_pos = o[:, :3]
    #         grip_pos[:, -1] -= dis_z
    #         object_pos = future_ag
    #         object_rel_pos = object_pos - grip_pos
    #
    #         # transitions['o'][her_indexes][:, :3] = grip_pos
    #         # transitions['o'][her_indexes][:, 3:6] = object_pos
    #         # transitions['o'][her_indexes][:, 6:9] = object_rel_pos
    #
    #         o[:, :3] = grip_pos
    #         o[:, 3:6] = object_pos
    #         o[:, 6:9] = object_rel_pos
    #
    #         transitions['o'][her_indexes] = o
    #
    #
    #         ## ['o', 'u', 'g', 'ag', 'info_is_success', 'o_2', 'ag_2']
    #         transitions['ag'][her_indexes] = future_ag.copy()
    #         transitions['g'][her_indexes] = future_g.copy()
    #
    #         ## ['o_2']
    #         o_ = transitions['o_2'][her_indexes]
    #
    #         grip_pos_ = transitions['o_2'][her_indexes][:, :3]
    #         grip_pos_[:, -1] -= dis_z
    #         object_pos_ = future_ag
    #         object_rel_pos_ = object_pos_ - grip_pos_
    #
    #         o_[:, :3] = grip_pos_
    #         o_[:, 3:6] = object_pos_
    #         o_[:, 6:9] = object_rel_pos_
    #
    #         transitions['o_2'][her_indexes] = o_
    #
    #         ## ['ag_2']
    #         transitions['ag_2'][her_indexes] = future_ag.copy()             ## change next ag
    #
    #     if m_ful_idx != []:   #### vanilia HER
    #         set_trace()
    #         # Select which episodes and time steps to use.
    #         episode_idxs_ = episode_idxs[m_ful_idx]
    #         t_samples = np.random.randint(T, size=batch_size)
    #         transitions = {key: episode_batch[key][episode_idxs_, t_samples].copy()
    #                        for key in episode_batch.keys()}
    #
    #         her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    #         future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #         future_offset = future_offset.astype(int)
    #         future_t = (t_samples + 1 + future_offset)[her_indexes]
    #
    #         #######################################################################################
    #         # Replace goal with achieved goal but only for the previously-selected
    #         # HER transitions (as defined by her_indexes). For the other transitions,
    #         # keep the original goal.
    #         future_ag = episode_batch['ag'][episode_idxs_[her_indexes], future_t]
    #         transitions['g'][her_indexes] = future_ag.copy()
    #
    #     # set_trace()
    #
    #
    #
    #     # # Select which episodes and time steps to use.
    #     # t_samples = np.random.randint(T, size=batch_size)
    #     # transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
    #     #                for key in episode_batch.keys()}
    #     # her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    #     # future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #     # future_offset = future_offset.astype(int)
    #     # future_t = (t_samples + 1 + future_offset)[her_indexes]
    #     # future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
    #     # transitions['g'][her_indexes] = future_ag.copy()
    #
    #
    #
    #
    #     all_transitions = {key: transitions[key].copy()
    #                        for key in episode_batch.keys()}
    #
    #     ########################################################################################
    #     # ----------------Goal-augmented ER---------------------------
    #     # if n_GER:
    #     if n_GER and grade_GER==0:
    #         # when n_GER != 0
    #         for _ in range (n_GER):
    #             PER_transitions = {key: transitions[key].copy()
    #                                for key in episode_batch.keys()}
    #             ger_machine = ger_learning(env_name = env_name,err_distance=err_distance)
    #             PER_indexes= np.array((range(0,batch_size)))
    #             HER_KER_future_ag = PER_transitions['g'][PER_indexes].copy()
    #             PER_future_g = ger_machine.process_goals(HER_KER_future_ag.copy())
    #             PER_transitions['g'][PER_indexes] = PER_future_g.copy()
    #             for key in episode_batch.keys():
    #                 all_transitions[key] = np.vstack([all_transitions[key], PER_transitions[key].copy()])
    #         # -----------------------End---------------------------
    #         # After GER, the minibatch size enlarged
    #         batch_size = batch_size * (1+n_GER)
    #         batch_size_in_transitions =batch_size
    #     ########################################################################################
    #
    #     ########################################################################################
    #     # ---------------- Added by Ray distance-GER ---------------------------
    #     # ----------------Goal-augmented ER---------------------------
    #     # if batch_size == 256:
    #     #     set_trace()
    #     # set_trace()
    #     # n_GER = 6
    #     # GER_para_dict = {}
    #     # GER_para_dict = {'GER_index': index, 'radius': radius, 'n_GER_grade': n_GER_grade,}
    #
    #     if grade_GER:
    #         GER_index_list=[]
    #         radius_list = []
    #         distance = err_distance
    #         n_Grade = 5
    #
    #         # set_trace()
    #         n_GER_grade = [int(i) for i in str(grade_GER)]
    #
    #         # n_GER_grade = [2,2,2,2,2] #G1
    #         #n_GER_grade = [2,1,1,1,1]   #G2
    #         # n_GER_grade = [4,1,1,1,1]   #G3
    #         assert n_Grade == len(n_GER_grade)
    #         for i in range(n_Grade):
    #             index = np.where(future_offset >= 10*i)
    #             GER_index_list.append(index)
    #             radius_list.append(distance)
    #             distance += 0.01
    #         for n_GER, GER_index, radius in zip(n_GER_grade, GER_index_list, radius_list):
    #             for i in range (n_GER):
    #                 PER_transitions = {key: transitions[key].copy()
    #                                    for key in episode_batch.keys()}
    #                 ger_machine = ger_learning(env_name = env_name,err_distance=radius)
    #                 # PER_indexes= np.array((range(0,batch_size)))
    #                 HER_KER_future_ag = PER_transitions['g'][GER_index].copy()
    #                 PER_future_g = ger_machine.process_goals(HER_KER_future_ag.copy())
    #                 PER_transitions['g'][GER_index] = PER_future_g.copy()
    #                 for key in episode_batch.keys():
    #                     all_transitions[key] = np.vstack([all_transitions[key], PER_transitions[key][GER_index].copy()])
    #         # -----------------------End----------------------------
    #         # After GER, the minibatch size enlarged
    #         batch_size = all_transitions['g'].shape[0]
    #         batch_size_in_transitions =batch_size
    #         # set_trace()
    #     # ---------------- Added by Ray distance-GER ---------------------------
    #     ########################################################################################
    #
    #
    #
    #
    #     # Reconstruct info dictionary for reward  computation.
    #     info = {}
    #     for key, value in all_transitions.items():
    #         if key.startswith('info_'):
    #             info[key.replace('info_', '')] = value
    #
    #     # Re-compute reward since we may have substituted the goal.
    #     reward_params = {k: all_transitions[k] for k in ['ag_2', 'g']}
    #     reward_params['info'] = info
    #     all_transitions['r'] = reward_fun(**reward_params)
    #
    #     all_transitions = {k: all_transitions[k].reshape(batch_size, *all_transitions[k].shape[1:])
    #                        for k in all_transitions.keys()}
    #
    #     assert(all_transitions['u'].shape[0] == batch_size_in_transitions)
    #     return all_transitions

    ### V1
    def _sample_her_transitions(episode_batch, batch_size_in_transitions,env_name=None, n_GER=0, grade_GER=0, err_distance=0.05):
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

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        #######################################################################################
        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag.copy()

        # create a new dict to store all the original, KER, HER, AGER data.

        all_transitions = {key: transitions[key].copy()
                           for key in episode_batch.keys()}

        ########################################################################################
        # ----------------Goal-augmented ER---------------------------
        # if n_GER:
        if n_GER and grade_GER==0:
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
        ########################################################################################

        ########################################################################################
        # ---------------- Added by Ray distance-GER ---------------------------
        # ----------------Goal-augmented ER---------------------------
        # if batch_size == 256:
        #     set_trace()
        # set_trace()
        # n_GER = 6
        # GER_para_dict = {}
        # GER_para_dict = {'GER_index': index, 'radius': radius, 'n_GER_grade': n_GER_grade,}

        if grade_GER:
            GER_index_list=[]
            radius_list = []
            distance = err_distance
            n_Grade = 5

            # set_trace()
            n_GER_grade = [int(i) for i in str(grade_GER)]

            # n_GER_grade = [2,2,2,2,2] #G1
            #n_GER_grade = [2,1,1,1,1]   #G2
            # n_GER_grade = [4,1,1,1,1]   #G3
            assert n_Grade == len(n_GER_grade)
            for i in range(n_Grade):
                index = np.where(future_offset >= 10*i)
                GER_index_list.append(index)
                radius_list.append(distance)
                distance += 0.01
            for n_GER, GER_index, radius in zip(n_GER_grade, GER_index_list, radius_list):
                for i in range (n_GER):
                    PER_transitions = {key: transitions[key].copy()
                                       for key in episode_batch.keys()}
                    ger_machine = ger_learning(env_name = env_name,err_distance=radius)
                    # PER_indexes= np.array((range(0,batch_size)))
                    HER_KER_future_ag = PER_transitions['g'][GER_index].copy()
                    PER_future_g = ger_machine.process_goals(HER_KER_future_ag.copy())
                    PER_transitions['g'][GER_index] = PER_future_g.copy()
                    for key in episode_batch.keys():
                        all_transitions[key] = np.vstack([all_transitions[key], PER_transitions[key][GER_index].copy()])
            # -----------------------End----------------------------
            # After GER, the minibatch size enlarged
            batch_size = all_transitions['g'].shape[0]
            batch_size_in_transitions =batch_size
            # set_trace()
        # ---------------- Added by Ray distance-GER ---------------------------
        ########################################################################################

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
