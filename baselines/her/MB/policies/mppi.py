# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import copy
import matplotlib.pyplot as plt
from ipdb import set_trace
# my imports
from baselines.her.MB.samplers import trajectory_sampler
from baselines.her.MB.utils.helper_funcs import do_groundtruth_rollout
from baselines.her.MB.utils.helper_funcs import turn_acs_into_acsK
from baselines.her.MB.utils.calculate_costs import calculate_costs


class MPPI(object):

    def __init__(self, dyn_models, ac_dim, params):

        ###########
        ## params
        ###########
        self.K = params.K
        self.horizon = params.horizon
        self.N = params.num_control_samples
        self.dyn_models = dyn_models
        self.reward_func = None

        #############
        ## init mppi vars
        #############
        self.ac_dim = ac_dim
        self.mppi_kappa = params.mppi_kappa
        self.sigma = params.mppi_mag_noise * np.ones(self.ac_dim)
        self.beta = params.mppi_beta
        self.mppi_mean = np.zeros((self.horizon, self.ac_dim))  #start mean at 0

    ###################################################################
    ###################################################################
    #### update action mean using weighted average of the actions (by their resulting scores)
    ###################################################################
    ###################################################################

    def mppi_update(self, scores, mean_scores, std_scores, all_samples):

        #########################
        ## how each sim's score compares to the best score
        ##########################
        S = np.exp(self.mppi_kappa * (scores - np.max(scores)))  # [N,]
        denom = np.sum(S) + 1e-10

        ##########################
        ## weight all actions of the sequence by that sequence's resulting reward
        ##########################
        S_shaped = np.expand_dims(np.expand_dims(S, 1), 2)  #[N,1,1]
        weighted_actions = (all_samples * S_shaped)  #[N x H x acDim]
        self.mppi_mean = np.sum(weighted_actions, 0) / denom

        ##########################
        ## return 1st element of the mean, which corresps to curr timestep
        ##########################
        return self.mppi_mean[0]

    def get_action(self, curr_state, goal, act_ddpg):

        # # remove the 1st entry of mean (mean from past timestep, that was just executed)
        # # and copy last entry (starting point, for the next timestep)
        # past_action = self.mppi_mean[0].copy()
        # self.mppi_mean[:-1] = self.mppi_mean[1:]

        ##############################################
        ## sample candidate action sequences
        ## by creating smooth filtered trajecs (noised around a mean)
        ##############################################
        np.random.seed()  # to get different action samples for each rollout

        eps = np.random.normal(loc=0, scale=1.0, size=(self.N, self.ac_dim)) * 0.2
        act_ddpg_tile = np.tile(act_ddpg, (self.N, 1))
        first_acts = act_ddpg_tile + eps

        #################################################
        ### Get result of executing those candidate action sequences
        #################################################
        resulting_states_list = self.dyn_models.do_forward_sim(curr_state, goal, first_acts)
        resulting_states_list = np.swapaxes(resulting_states_list, 0,1)  #[ensSize, horizon+1, N, statesize]

        ############################
        ### evaluate the predicted trajectories
        ############################

        # calculate costs [N,]
        costs, std_costs = calculate_costs(resulting_states_list, goal)

        if (costs == costs.min()).all():
            selected_action = act_ddpg
            # self.act_NN+=1
        else:
            # self.act_plan+=1
            # set_trace()
            idx = np.where(costs==costs.min())[0]
            # selected_action = first_acts[idx][0]
            selected_action = first_acts[idx].mean(axis=0)
            
            selected_action = np.tile(selected_action,(1,1))

        # uses all paths to update action mean (for horizon steps)
        # Note: mppi_update needs rewards, so pass in -costs
        # selected_action = self.mppi_update(-costs, -mean_costs, std_costs, first_acts)


        return selected_action
