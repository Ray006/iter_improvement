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

# ############  to search module in current path ###########################
import os
import sys
addr_ = os.getcwd()
sys.path.append(addr_)
# ############  to search module in current path ###########################
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import numpy.random as npr
from random import shuffle
import tensorflow as tf
import pickle
import argparse
import traceback
from ipdb import set_trace
# set_trace()


#my imports
from baselines.her.MB.policies.policy_random import Policy_Random
from baselines.her.MB.utils.helper_funcs import *
from baselines.her.MB.regressors.dynamics_model import Dyn_Model
from baselines.her.MB.policies.mpc_rollout import MPCRollout
from baselines.her.MB.utils.loader import Loader
from baselines.her.MB.utils.saver import Saver
from baselines.her.MB.utils.data_processor import DataProcessor
from baselines.her.MB.utils.data_structures import *
from baselines.her.MB.utils.convert_to_parser_args import convert_to_parser_args
from baselines.her.MB.utils import config_reader

from baselines.her.MB.policies.mppi import MPPI

SCRIPT_DIR = os.path.dirname(__file__)


class MB_class:
    def __init__(self, buffer_size, dims, policy):
        self.buffer_size = buffer_size
        self.rollouts = []
        # self.num_data = 0

        para_dict={
                    'use_gpu': [1],
                    # 'use_gpu': [0],
                    'gpu_frac': [0.5],
                    #########################
                    ##### run options
                    #########################
                    'job_name': ['ant'],
                    'seed': [0],
                    #########################
                    ##### experiment options
                    #########################
                    'print_minimal': [False],

                    ## noise
                    'make_aggregated_dataset_noisy': [True],
                    'make_training_dataset_noisy': [True],
                    'rollouts_noise_actions': [False],
                    'rollouts_document_noised_actions': [False],

                    ##########################
                    ##### dynamics model
                    ##########################
                    ## arch
                    'num_fc_layers': [2],
                    'depth_fc_layers': [400],
                    'ensemble_size': [3],
                    # 'ensemble_size': [5],
                    'K': [1],
                    ## model training
                    'warmstart_training': [False],
                    'always_use_savedModel': [False],
                    'batchsize': [512],
                    'lr': [0.001],
                    'nEpoch': [30],
                    'nEpoch_init': [30],
                    ##########################
                    ##### controller
                    ##########################
                    ## MPC
                    'horizon': [5],
                    'num_control_samples': [500],
                    'controller_type': ['mppi'],
                    ## mppi
                    'mppi_kappa': [10],
                    'mppi_mag_noise': [0.8],
                    'mppi_beta': [0.7],
                  }
        #convert job dictionary to different format
        args_list = config_dict_to_flags(para_dict)
        self.args = convert_to_parser_args(args_list)



        ### set seeds
        npr.seed(self.args.seed)
        tf.set_random_seed(self.args.seed)
        ### data types
        self.args.tf_datatype = tf.float32
        self.args.np_datatype = np.float32
        ### supervised learning noise, added to the training dataset
        self.args.noiseToSignal = 0.01
        #initialize data processor
        self.data_processor = DataProcessor(self.args)
        # #initialize saver
        # saver = Saver(save_dir, sess)
        # saver_data = DataPerIter()

        # self.sess = tf.Session(config=get_gpu_config(self.args.use_gpu, self.args.gpu_frac))
        
        ### init model
        s_dim, a_dim = dims['o'], dims['u']
        inputSize = s_dim + a_dim
        outputSize = s_dim
        acSize = a_dim

        self.dyn_models = Dyn_Model(inputSize, outputSize, acSize, policy, params=self.args)
        self.planner = MPPI(self.dyn_models, a_dim, params=self.args)
        self.model_was_learned = False


    def store_rollout(self, episode):

        rollout = Rollout(episode['o'][0], episode['u'][0])
        self.rollouts.append(rollout)
        
        num_rollouts = len(self.rollouts)
        lenth_each_rollout = self.rollouts[0].actions.shape[0]
        num_data = num_rollouts * lenth_each_rollout
        
        if num_data > self.buffer_size: 
            # set_trace()
            self.rollouts.pop(0)
            # print('d')


    def get_data_dim(self):
        assert len(self.rollouts)>0
        state_dim = self.rollouts[0].states.shape[-1]
        action_dim = self.rollouts[0].actions.shape[-1]
        return state_dim, action_dim

    def get_rollout(self):
               
        rollouts_train = []
        rollouts_val = []

        num_mpc_rollouts = len(self.rollouts)
        shuffle(self.rollouts)

        for i,rollout in enumerate(self.rollouts):
            if i<int(num_mpc_rollouts * 0.9):
                rollouts_train.append(rollout)
            else:
                rollouts_val.append(rollout)

        return rollouts_train, rollouts_val

    def run_job(self):  ## v3, outside session 
        self.model_was_learned = True
        ### get data from the buffer
        rollouts_trainOnPol, rollouts_valOnPol = self.get_rollout()
        #convert (rollouts --> dataset)
        dataset_trainOnPol = self.data_processor.convertRolloutsToDatasets(rollouts_trainOnPol)
        dataset_valOnPol = self.data_processor.convertRolloutsToDatasets(rollouts_valOnPol)
        ### update model mean/std
        inputSize, outputSize, acSize = check_dims(dataset_trainOnPol) # just for printing
        self.data_processor.update_stats(self.dyn_models, dataset_trainOnPol) # mean/std of all data
        #preprocess datasets to mean0/std1 + clip actions
        preprocessed_data_trainOnPol = self.data_processor.preprocess_data(dataset_trainOnPol)
        preprocessed_data_valOnPol = self.data_processor.preprocess_data(dataset_valOnPol)
        #convert datasets (x,y,z) --> training sets (inp, outp)
        inputs_onPol, outputs_onPol = self.data_processor.xyz_to_inpOutp(preprocessed_data_trainOnPol)
        inputs_val_onPol, outputs_val_onPol = self.data_processor.xyz_to_inpOutp(preprocessed_data_valOnPol)

        # set_trace()
        # tf.reset_default_graph()  #### ????   
        # with tf.Session(config=get_gpu_config(self.args.use_gpu, self.args.gpu_frac)) as sess:

        # #re-initialize all vars (randomly) if training from scratch
        # self.sess.run(tf.global_variables_initializer())

        ## train model
        training_loss, training_lists_to_save = self.dyn_models.train(
            inputs_onPol,
            outputs_onPol,
            self.args.nEpoch,
            inputs_val_onPol=inputs_val_onPol,
            outputs_val_onPol=outputs_val_onPol)

            # #########################################################
            # ### save everything about this iter of model training
            # #########################################################
            # trainingLoss_perIter.append(training_loss)

            # saver_data.training_losses = trainingLoss_perIter
            # saver_data.training_lists_to_save = training_lists_to_save

            # saver_data.train_rollouts_onPol = rollouts_trainOnPol
            # saver_data.val_rollouts_onPol = rollouts_valOnPol

            # ### save all info from this training iteration
            # saver.save_model()
            # saver.save_training_info(saver_data)

        return 0


    # # def run_job(self, save_dir=None):  ## work, v2 

    #     # set_trace()
    #     tf.reset_default_graph()  #### ????

    #     # sess_MB = tf.Session(config=get_gpu_config(self.args.use_gpu, self.args.gpu_frac)) 
    #     with tf.Session(config=get_gpu_config(self.args.use_gpu, self.args.gpu_frac)) as sess:

    #         ### set seeds
    #         npr.seed(self.args.seed)
    #         tf.set_random_seed(self.args.seed)
    #         ### data types
    #         self.args.tf_datatype = tf.float32
    #         self.args.np_datatype = np.float32
    #         ### supervised learning noise, added to the training dataset
    #         self.args.noiseToSignal = 0.01
    #         #initialize data processor
    #         data_processor = DataProcessor(self.args)

    #         # #initialize saver
    #         # saver = Saver(save_dir, sess)
    #         # saver_data = DataPerIter()

    #         ### init model
    #         s_dim, a_dim = self.get_data_dim()
    #         inputSize = s_dim + a_dim
    #         outputSize = s_dim
    #         acSize = a_dim
    #         dyn_models = Dyn_Model(inputSize, outputSize, acSize, sess, params=self.args)

    #         ### get data from the buffer
    #         rollouts_trainOnPol, rollouts_valOnPol = self.get_rollout()
    #         #convert (rollouts --> dataset)
    #         dataset_trainOnPol = data_processor.convertRolloutsToDatasets(rollouts_trainOnPol)
    #         dataset_valOnPol = data_processor.convertRolloutsToDatasets(rollouts_valOnPol)
    #         ### update model mean/std
    #         inputSize, outputSize, acSize = check_dims(dataset_trainOnPol) # just for printing
    #         data_processor.update_stats(dyn_models, dataset_trainOnPol) # mean/std of all data
    #         #preprocess datasets to mean0/std1 + clip actions
    #         preprocessed_data_trainOnPol = data_processor.preprocess_data(dataset_trainOnPol)
    #         preprocessed_data_valOnPol = data_processor.preprocess_data(dataset_valOnPol)
    #         #convert datasets (x,y,z) --> training sets (inp, outp)
    #         inputs_onPol, outputs_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_trainOnPol)
    #         inputs_val_onPol, outputs_val_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_valOnPol)


    #         #re-initialize all vars (randomly) if training from scratch
    #         sess.run(tf.global_variables_initializer())
    #         ## train model
    #         training_loss, training_lists_to_save = dyn_models.train(
    #             inputs_onPol,
    #             outputs_onPol,
    #             self.args.nEpoch,
    #             inputs_val_onPol=inputs_val_onPol,
    #             outputs_val_onPol=outputs_val_onPol)

    #         # #########################################################
    #         # ### save everything about this iter of model training
    #         # #########################################################
    #         # trainingLoss_perIter.append(training_loss)

    #         # saver_data.training_losses = trainingLoss_perIter
    #         # saver_data.training_lists_to_save = training_lists_to_save

    #         # saver_data.train_rollouts_onPol = rollouts_trainOnPol
    #         # saver_data.val_rollouts_onPol = rollouts_valOnPol

    #         # ### save all info from this training iteration
    #         # saver.save_model()
    #         # saver.save_training_info(saver_data)

    #     return 0


    # def run_job(self, save_dir=None):  ### work, v1

    #     # set_trace()

    #     # tf.reset_default_graph()
    #     with tf.Session(config=get_gpu_config(self.args.use_gpu, self.args.gpu_frac)) as sess:

    #         ### set seeds
    #         npr.seed(self.args.seed)
    #         tf.set_random_seed(self.args.seed)
    #         ### data types
    #         self.args.tf_datatype = tf.float32
    #         self.args.np_datatype = np.float32
    #         ### supervised learning noise, added to the training dataset
    #         self.args.noiseToSignal = 0.01
    #         #initialize data processor
    #         data_processor = DataProcessor(self.args)

    #         # #initialize saver
    #         # saver = Saver(save_dir, sess)
    #         # saver_data = DataPerIter()

    #         ### init model
    #         s_dim, a_dim = self.get_data_dim()
    #         inputSize = s_dim + a_dim
    #         outputSize = s_dim
    #         acSize = a_dim
    #         dyn_models = Dyn_Model(inputSize, outputSize, acSize, sess, params=self.args)

    #         ### get data from the buffer
    #         rollouts_trainOnPol, rollouts_valOnPol = self.get_rollout()
    #         #convert (rollouts --> dataset)
    #         dataset_trainOnPol = data_processor.convertRolloutsToDatasets(rollouts_trainOnPol)
    #         dataset_valOnPol = data_processor.convertRolloutsToDatasets(rollouts_valOnPol)
    #         ### update model mean/std
    #         inputSize, outputSize, acSize = check_dims(dataset_trainOnPol) # just for printing
    #         data_processor.update_stats(dyn_models, dataset_trainOnPol) # mean/std of all data
    #         #preprocess datasets to mean0/std1 + clip actions
    #         preprocessed_data_trainOnPol = data_processor.preprocess_data(dataset_trainOnPol)
    #         preprocessed_data_valOnPol = data_processor.preprocess_data(dataset_valOnPol)
    #         #convert datasets (x,y,z) --> training sets (inp, outp)
    #         inputs_onPol, outputs_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_trainOnPol)
    #         inputs_val_onPol, outputs_val_onPol = data_processor.xyz_to_inpOutp(preprocessed_data_valOnPol)


    #         #re-initialize all vars (randomly) if training from scratch
    #         sess.run(tf.global_variables_initializer())
    #         ## train model
    #         training_loss, training_lists_to_save = dyn_models.train(
    #             inputs_onPol,
    #             outputs_onPol,
    #             self.args.nEpoch,
    #             inputs_val_onPol=inputs_val_onPol,
    #             outputs_val_onPol=outputs_val_onPol)

    #         # #########################################################
    #         # ### save everything about this iter of model training
    #         # #########################################################
    #         # trainingLoss_perIter.append(training_loss)

    #         # saver_data.training_losses = trainingLoss_perIter
    #         # saver_data.training_lists_to_save = training_lists_to_save

    #         # saver_data.train_rollouts_onPol = rollouts_trainOnPol
    #         # saver_data.val_rollouts_onPol = rollouts_valOnPol

    #         # ### save all info from this training iteration
    #         # saver.save_model()
    #         # saver.save_training_info(saver_data)

    #     return 0




# def run_job(args, save_dir=None):   #### not work

#     set_trace()

#     tf.reset_default_graph()
#     with tf.Session(config=get_gpu_config(args.use_gpu, args.gpu_frac)) as sess:

#         ##############################################
#         ### initialize some commonly used parameters (from args)
#         ##############################################

#         env_name = args.env_name
#         continue_run = args.continue_run
#         K = args.K
#         num_iters = args.num_iters
#         num_trajectories_per_iter = args.num_trajectories_per_iter
#         horizon = args.horizon

#         ### set seeds
#         npr.seed(args.seed)
#         tf.set_random_seed(args.seed)

#         #######################
#         ### hardcoded args
#         #######################

#         ### data types
#         args.tf_datatype = tf.float32
#         args.np_datatype = np.float32

#         ### supervised learning noise, added to the training dataset
#         args.noiseToSignal = 0.01


#         ########################################
#         ### create loader, env, rand policy
#         ########################################
#         # loader = Loader(save_dir)
#         # env, dt_from_xml = create_env(env_name)
#         # args.dt_from_xml = dt_from_xml


#         #################################################
#         ### initialize or load in info
#         #################################################

#         duplicateData_switchObjs = False
#         indices_for_switching=[]

#         #initialize data processor
#         data_processor = DataProcessor(args, duplicateData_switchObjs, indices_for_switching)


#         get_data_from_HER_buffer!!!!!

#         #convert (rollouts --> dataset)
#         dataset_trainRand = data_processor.convertRolloutsToDatasets(rollouts_trainRand)
#         dataset_valRand = data_processor.convertRolloutsToDatasets(rollouts_valRand)
#         #onPol train/val data
#         dataset_trainOnPol = Dataset()
#         rollouts_trainOnPol = []
#         rollouts_valOnPol = []
#         #lists for saving
#         trainingLoss_perIter = []
#         rew_perIter = []
#         scores_perIter = []
#         trainingData_perIter = []
#         #initialize counter
#         counter = 0


#         ### check data dims
#         inputSize, outputSize, acSize = check_dims(dataset_trainRand, env)

#         ### amount of data
#         numData_train_rand = get_num_data(rollouts_trainRand)

#         ##############################################
#         ### dynamics model + controller
#         ##############################################

#         dyn_models = Dyn_Model(inputSize, outputSize, acSize, sess, params=args)

#         ### these are for *during* MPC rollouts,
#         # they allow you to run the H-step candidate actions on the real dynamics
#         # and compare the model's predicted outcomes vs. the true outcomes
#         execute_sideRollouts = False
#         plot_sideRollouts = True
#         random_policy = Policy_Random(env.env)
#         mpc_rollout = MPCRollout(env, dyn_models, random_policy, execute_sideRollouts, plot_sideRollouts, args)


#         # from ipdb import set_trace
#         # set_trace()
#         # print(tf.global_variables())
        

#         ### init TF variables
#         sess.run(tf.global_variables_initializer())

#         ##############################################
#         ###  saver
#         ##############################################

#         saver = Saver(save_dir, sess)
#         saver.save_initialData(args, rollouts_trainRand, rollouts_valRand)

#         ##############################################
#         ### THE MAIN LOOP
#         ##############################################

#         firstTime = True

#         rollouts_info_prevIter, list_mpes, list_scores, list_rewards = None, None, None, None
#         while counter < num_iters:

#             #init vars for this iteration
#             saver_data = DataPerIter()
#             saver.iter_num = counter

#             #convert (rollouts --> dataset)
#             dataset_trainOnPol = data_processor.convertRolloutsToDatasets(
#                 rollouts_trainOnPol)
#             dataset_valOnPol = data_processor.convertRolloutsToDatasets(
#                 rollouts_valOnPol)

#             # amount of data
#             numData_train_onPol = get_num_data(rollouts_trainOnPol)

#             # mean/std of all data
#             data_processor.update_stats(dyn_models, dataset_trainRand, dataset_trainOnPol)

#             #preprocess datasets to mean0/std1 + clip actions
#             preprocessed_data_trainOnPol = data_processor.preprocess_data(
#                 dataset_trainOnPol)
#             preprocessed_data_valOnPol = data_processor.preprocess_data(
#                 dataset_valOnPol)

#             #convert datasets (x,y,z) --> training sets (inp, outp)
#             inputs_onPol, outputs_onPol = data_processor.xyz_to_inpOutp(
#                 preprocessed_data_trainOnPol)
#             inputs_val_onPol, outputs_val_onPol = data_processor.xyz_to_inpOutp(
#                 preprocessed_data_valOnPol)

#             #####################################
#             ## Training the model
#             #####################################

#             if (not (args.print_minimal)):
#                 print("\n#####################################")
#                 print("Training the dynamics model..... iteration ", counter)
#                 print("#####################################\n")
#                 print("    amount of random data: ", numData_train_rand)
#                 print("    amount of onPol data: ", numData_train_onPol)

#             #re-initialize all vars (randomly) if training from scratch
#             sess.run(tf.global_variables_initializer())

#             #number of training epochs
#             if counter==0: nEpoch_use = args.nEpoch_init
#             else: nEpoch_use = args.nEpoch

#             ## train model
#             training_loss, training_lists_to_save = dyn_models.train(
#                 inputs_onPol,
#                 outputs_onPol,
#                 nEpoch_use,
#                 inputs_val_onPol=inputs_val_onPol,
#                 outputs_val_onPol=outputs_val_onPol)

#             #saving rollout info
#             rollouts_info = []
#             list_rewards = []
#             list_scores = []
#             list_mpes = []

#             if not args.print_minimal:
#                 print("\n#####################################")
#                 print("performing on-policy MPC rollouts... iter ", counter)
#                 print("#####################################\n")

#             for rollout_num in range(num_trajectories_per_iter):

#                 ###########################################
#                 ########## perform 1 MPC rollout
#                 ###########################################

#                 if not args.print_minimal:
#                     print("\n####################### Performing MPC rollout #",
#                           rollout_num)

#                 #reset env randomly
#                 starting_observation, starting_state = env.reset(return_start_state=True)

#                 rollout_info = mpc_rollout.perform_rollout(
#                     starting_state,
#                     starting_observation,
#                     controller_type=args.controller_type,
#                     take_exploratory_actions=False)

#                 # Note: can sometimes set take_exploratory_actions=True
#                 # in order to use ensemble disagreement for exploration

#                 ###########################################
#                 ####### save rollout info (if long enough)
#                 ###########################################

#                 if len(rollout_info['observations']) > K:
#                     list_rewards.append(rollout_info['rollout_rewardTotal'])
#                     list_scores.append(rollout_info['rollout_meanFinalScore'])
#                     list_mpes.append(np.mean(rollout_info['mpe_1step']))
#                     rollouts_info.append(rollout_info)

#             rollouts_info_prevIter = rollouts_info.copy()

#             #########################################################
#             ### aggregate MPC rollouts into train/val
#             #########################################################

#             num_mpc_rollouts = len(rollouts_info)
#             rollouts_train = []
#             rollouts_val = []

#             for i in range(num_mpc_rollouts):
#                 rollout = Rollout(rollouts_info[i]['observations'],
#                                   rollouts_info[i]['actions'],
#                                   rollouts_info[i]['rollout_rewardTotal'],
#                                   rollouts_info[i]['starting_state'])

#                 if i<int(num_mpc_rollouts * 0.9):
#                     rollouts_train.append(rollout)
#                 else:
#                     rollouts_val.append(rollout)

#             #aggregate into training data
#             if counter==0: rollouts_valOnPol = []
#             rollouts_trainOnPol = rollouts_trainOnPol + rollouts_train
#             rollouts_valOnPol = rollouts_valOnPol + rollouts_val

#             #########################################################
#             ### save everything about this iter of model training
#             #########################################################

#             trainingData_perIter.append(numData_train_rand +
#                                         numData_train_onPol)
#             trainingLoss_perIter.append(training_loss)

#             ### stage relevant info for saving
#             saver_data.training_numData = trainingData_perIter
#             saver_data.training_losses = trainingLoss_perIter
#             saver_data.training_lists_to_save = training_lists_to_save
#             # Note: the on-policy rollouts include curr iter's rollouts
#             # (so next iter can be directly trained on these)
#             saver_data.train_rollouts_onPol = rollouts_trainOnPol
#             saver_data.val_rollouts_onPol = rollouts_valOnPol
#             saver_data.normalization_data = data_processor.get_normalization_data()
#             saver_data.counter = counter

#             ### save all info from this training iteration
#             saver.save_model()
#             saver.save_training_info(saver_data)

#             #########################################################
#             ### save everything about this iter of MPC rollouts
#             #########################################################

#             # append onto rewards/scores
#             rew_perIter.append([np.mean(list_rewards), np.std(list_rewards)])
#             scores_perIter.append([np.mean(list_scores), np.std(list_scores)])

#             # save
#             saver_data.rollouts_rewardsPerIter = rew_perIter
#             saver_data.rollouts_scoresPerIter = scores_perIter
#             saver_data.rollouts_info = rollouts_info
#             saver.save_rollout_info(saver_data)
#             counter = counter + 1

#             firstTime = False
#         return


def main():

    #####################
    # training args
    #####################

    parser = argparse.ArgumentParser(
        # Show default value in the help doc.
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        '-c',
        '--config',
        nargs='*',
        help=('Path to the job data config file. This is specified relative '
            'to working directory'))

    parser.add_argument(
        '-o',
        '--output_dir',
        default='output',
        help=
        ('Directory to output trained policies, logs, and plots. A subdirectory '
         'is created for each job. This is speficified relative to  '
         'working directory'))

    parser.add_argument('--use_gpu', action="store_true")
    parser.add_argument('-frac', '--gpu_frac', type=float, default=0.9)
    general_args = parser.parse_args()

    #####################
    # job configs
    #####################

    # Get the job config files
    jobs = config_reader.process_config_files(general_args.config)
    assert jobs, 'No jobs found from config.'

    # Create the output directory if not present.
    output_dir = general_args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    # Run separate experiment for each variant in the config
    for index, job in enumerate(jobs):

        #add an index to jobname, if there is more than 1 job
        if len(jobs)>1:
            job['job_name'] = '{}_{}'.format(job['job_name'], index)

        #convert job dictionary to different format
        args_list = config_dict_to_flags(job)
        args = convert_to_parser_args(args_list)

        #copy some general_args into args
        args.use_gpu = general_args.use_gpu
        args.gpu_frac = general_args.gpu_frac

        #directory name for this experiment
        job['output_dir'] = os.path.join(output_dir, job['job_name'])

        ################
        ### run job
        ################

        try:
            run_job(args, job['output_dir'])
        except (KeyboardInterrupt, SystemExit):
            print('Terminating...')
            sys.exit(0)
        except Exception as e:
            print('ERROR: Exception occured while running a job....')
            traceback.print_exc()


if __name__ == '__main__':
    main()
