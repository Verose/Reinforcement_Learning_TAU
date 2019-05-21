import pickle
import platform
if not platform.system() == 'Windows':
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt


with open('statistics.pkl', 'rb') as f:
    saved_state = pickle.load(f)


plt.clf()
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward (past 100 episodes)')
num_items = len(saved_state.stats["mean_episode_rewards"])
plt.plot(range(num_items), saved_state.stats["mean_episode_rewards"], label='mean reward')
plt.plot(range(num_items), saved_state.stats["best_mean_episode_rewards"], label='best mean rewards')
plt.legend()
plt.savefig('../statistics.png')
