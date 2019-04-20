import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = [] TODO: From Eytan - Don't know what this if for
rList = []
np.random.seed(123) # TODO: We can comment this out in the submission
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0  # Total reward during current episode
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # TODO: Implement Q-Learning
        # 1. Choose an action by greedily (with noise) picking from Q table
        e_t = 1 / (j ** 0.9)
        if np.random.uniform(0,1) <= 1-e_t and Q[s].max() != 0:
            a = Q[s].argmax()
        else:
            a = env.action_space.sample()
          
        # 2. Get new state and reward from environment
        s_t1, r, d, info = env.step(a)
        # 3. Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr * (r + y * np.max(Q[s_t1]) - Q[s,a])
        s = s_t1
        # 4. Update total reward
        rAll += r
        # 5. Update episode if we reached the Goal State
        if d:
            # jList.append(j)
            j=100
    
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
env.close()
