import numpy as np
import argparse
import gym
import json
## Visualize a rollout, given actions

def visualize_rendering(rollout_info, env):

    ### reset env to the starting state
    curr_state = env.reset()

    from ipdb import set_trace;
    set_trace()

    reset_state = self.unwrapped_env.sim.get_state()
    initial_qpos = reset_state[1][:3]

    initial_qpos = {'robot0:slide0': 0.4049, 'robot0:slide1': 0.55, 'robot0:slide2': 0.0}
    env.unwrapped._env_setup(initial_qpos)

    env.unwrapped.goal = goal

    # object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    # assert object_qpos.shape == (7,)
    # object_qpos[:2] = object_xpos
    # self.sim.data.set_joint_qpos('object0:joint', object_qpos)

    def do_reset(self, initial_state):   #used for using true dynamic in pddm

        self.sim.set_state(initial_state[0])
        self.goal = initial_state[1]
        self.sim.forward()
        return self._get_obs()

    scores, rewards = [], []
    for action in rollout_info['actions']:

        # from ipdb import set_trace;
        # set_trace()

        if action.shape[0] == 1:
            next_state, rew, done, env_info = env.step(action[0])
        else:
            next_state, rew, done, env_info = env.step(action)

        render_env(env)
        rewards.append(rew)

    print("Done taking ", count, " steps.")
    print("FINAL REW: ", np.sum(rewards))
    print("    TIME TAKEN : {:0.4f} s".format(time.time() - starttime))
    return rewards, scores

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--job_path', type=str)  #address this path WRT your working directory
    # parser.add_argument('--iter_num', type=int, default=1)
    # args = parser.parse_args()

    # rollouts_info = np.load('xxx.npy')
    rollouts_info = ['sfw','fg']

    Json = json.load(open('params.json'))
    env_name = Json.get('env_name')
    env = gym.make(env_name)

    for rollout in rollouts_info:
        visualize_rendering(rollout, env)

if __name__ == '__main__':
    main()