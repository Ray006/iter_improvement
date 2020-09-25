from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
from ipdb import set_trace

def main():

    # results = pu.load_results('data_the_best')
    results = pu.load_results('data_Test_obstacle_origin/log_data')
    r = results[0]
    # plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    # plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
    #### plt.plot(r.progress.total_timesteps, r.progress.eprewmean)

    # print('keys:', r.progress.keys())
    # plt.plot(r.progress['epoch'], r.progress['test/success_rate'])
    # plt.plot(r.progress['epoch'], pu.smooth(r.progress['test/success_rate'], radius=5))

    # pu.plot_results(results)
    pu.plot_results(results, average_group=True, split_fn=lambda _: '')
    set_trace()

if __name__ == '__main__':
    main()
