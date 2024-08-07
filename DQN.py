import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle

class DQN(nn.Module):
    def __init__(self, action_shape, state_shape):
        super().__init__()
        self.fc1 = nn.Linear(in_features=state_shape,out_features=512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,256)
        self.out = nn.Linear(in_features = 256, out_features=action_shape)
        pass

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class LunarLanderDQL():
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.95
    SYNC_TIME = 5

    REPLAY_MEMORY_SIZE = 5000
    MIN_REPLAY_MEMORY_SIZE = 500
    REPLAY_MEMORY_BATCH = 32


    optimizer = None
    replay_mem = None

    loss_function = nn.MSELoss()

    def train(self, episodes, render=False):
        env = gym.make("LunarLander-v2", render_mode='human' if render else None)
        n_states = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.replay_mem = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        policy_dqn = DQN(n_action,n_states)
        target_dqn = DQN(n_action,n_states)

        epsilon = 1
        epsilon_stats = []
        rewards_stats = np.zeros(episodes)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.LEARNING_RATE)

        step = 0


        curr_reward = 0
        for i in range(episodes):
            print(f'episodes: {i} / {episodes} rewards:')
            current_state = env.reset()[0]
            terminated = False
            truncated = False

            while(not terminated and not truncated):

                if episodes % 10 == 0:
                    env.render()

                if random.random() < epsilon: #Epsilon greedy policy
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(torch.from_numpy(current_state)).argmax().item()

                next_state, reward, terminated, truncated, info = env.step(action)


                self.replay_mem.append([current_state, next_state, action, reward, terminated, truncated, info])

                curr_reward = reward
                current_state = next_state

                step += 1

                if reward == 100:
                    rewards_stats[i] = 100
    
                if len(self.replay_mem) > self.MIN_REPLAY_MEMORY_SIZE:
                    mini_batch = random.sample(self.replay_mem, self.REPLAY_MEMORY_BATCH)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon - 1/episodes, 0)

                    if step % self.SYNC_TIME == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
            print(curr_reward) 

        env.close()
        
        torch.save(policy_dqn.state_dict(), "LunarLander_DQL.pt")

        plt.figure(1)

        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_stats[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_stats)
        
        # Save plots
        plt.savefig('lunar_lander.png')



    def optimize(self,mini_batch, policy_dqn, target_dqn):
        q_value_policy_array = []
        q_value_target_array = []
        for current_state, next_state, action, reward, terminated, truncated, info in mini_batch:
            if not terminated and not truncated:
                with torch.no_grad():
                    max_value = torch.max(target_dqn(torch.from_numpy(next_state)))
                    target = torch.FloatTensor(reward + self.DISCOUNT_FACTOR * max_value)
            else:
                target = torch.FloatTensor([reward])


            q_value_policy = policy_dqn(torch.from_numpy(current_state))
            q_value_policy_array.append(q_value_policy)

            q_value_target = target_dqn(torch.from_numpy(current_state))
            q_value_target[action] = target
            q_value_target_array.append(q_value_target)


        q_value_policy_array = torch.stack(q_value_policy_array)
        q_value_target_array = torch.stack(q_value_target_array)
        loss = self.loss_function(q_value_policy_array, q_value_target_array)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class GYMEnv():
    def start(self):
        env = gym.make("LunarLander-v2", render_mode="human")
        observation, info = env.reset(seed=42)
        for _ in range(1):
            action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)
            # observation, reward, terminated, truncated, info = env.step(3)
            print(env.observation_space.shape[0])
            print(env.observation_space)
            print(env.action_space.n)
            print(env.action_space)
            # print(env.step(action))
            if terminated or truncated:
               observation, info = env.reset()
        
        env.close()

def main():
    # LunarLander = GYMEnv();
    # LunarLander.start();

    lunarLander = LunarLanderDQL()
    lunarLander.train(1000)
    


if __name__ == "__main__":
    main()
