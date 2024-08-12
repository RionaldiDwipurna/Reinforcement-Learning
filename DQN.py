#reference code:
#https://github.com/johnnycode8/gym_solutions/


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
        self.fc1 = nn.Linear(state_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_shape)
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)  
        return x


class LunarLanderDQL():
    LEARNING_RATE = 0.0005
    DISCOUNT_FACTOR = 0.95
    SYNC_TIME = 10000

    REPLAY_MEMORY_SIZE = 100000
    MIN_REPLAY_MEMORY_SIZE = 500
    REPLAY_MEMORY_BATCH = 64
    NUM_DIVISION = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = None
    replay_mem = None

    loss_function = nn.MSELoss()

    def input_discrete(self, state)->torch.tensor:
        state_x_pos = np.digitize(state[0], self.x_pos)
        state_y_pos = np.digitize(state[1], self.y_pos)
        state_x_vel = np.digitize(state[2], self.x_vel)
        state_y_vel = np.digitize(state[3], self.y_vel)
        state_angle = np.digitize(state[4], self.angle)
        state_ang_v = np.digitize(state[5], self.ang_v)
        return torch.tensor([state_x_pos, state_y_pos, state_x_vel, state_y_vel, state_angle,state_ang_v, state[6], state[7]]).to(self.device)

    def train(self, episodes, render=False):
        env = gym.make("LunarLander-v2", render_mode='human' if render else None)

        n_states = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.replay_mem = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        policy_dqn = DQN(n_action,n_states).to(self.device)
        target_dqn = DQN(n_action,n_states).to(self.device)

        epsilon = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995

        epsilon_stats = []
        rewards_stats = np.zeros(episodes)


        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.LEARNING_RATE)

        self.x_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.NUM_DIVISION) 
        self.y_pos = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.NUM_DIVISION) 
        self.x_vel = np.linspace(env.observation_space.low[2], env.observation_space.high[2], self.NUM_DIVISION) 
        self.y_vel = np.linspace(env.observation_space.low[3], env.observation_space.high[3], self.NUM_DIVISION) 
        self.angle = np.linspace(env.observation_space.low[4], env.observation_space.high[4], self.NUM_DIVISION) 
        self.ang_v = np.linspace(env.observation_space.low[5], env.observation_space.high[5], self.NUM_DIVISION) 

        step = 0


        for i in range(episodes):
            current_state = env.reset()[0]
            terminated = False
            truncated = False

            curr_reward = []
            while(not terminated and not truncated):

                if random.random() < epsilon: #Epsilon greedy policy
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # action = policy_dqn(torch.from_numpy(current_state).float().to(self.device)).argmax().item()
                        action = policy_dqn(self.input_discrete(current_state)).argmax().item()

                next_state, reward, terminated, truncated, info = env.step(action)


                self.replay_mem.append([current_state, next_state, action, reward, terminated, truncated, info])

                curr_reward.append(reward)

                current_state = next_state

                step += 1

                if reward == 100:
                    rewards_stats[i] = 100

                if len(self.replay_mem) > self.MIN_REPLAY_MEMORY_SIZE:
                    mini_batch = random.sample(self.replay_mem, self.REPLAY_MEMORY_BATCH)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # epsilon = max(epsilon - 1/episodes, 0)
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)

                    if step % self.SYNC_TIME == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                if (i+1) % 500 == 0:
                    torch.save(policy_dqn.state_dict(), f"LunarLander_DQL_{i}.pt")


            print(f'episodes: {i} / {episodes} rewards: {np.mean(curr_reward)}')

        env.close()
        
        torch.save(policy_dqn.cpu().state_dict(), "LunarLander_DQL.pt")

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
            # current_state = torch.from_numpy(current_state).float().to(self.device)
            # next_state =    torch.from_numpy(next_state).float().to(self.device)
            current_state = self.input_discrete(current_state) 
            current_state = self.input_discrete(nex_state) 

            if not terminated and not truncated:
                with torch.no_grad():
                    max_value = torch.max(target_dqn(next_state))
                    target = reward + self.DISCOUNT_FACTOR * max_value
            else:
                target = torch.tensor([reward]).to(self.device)


            q_value_policy = policy_dqn(current_state)
            q_value_policy_array.append(q_value_policy)

            q_value_target = target_dqn(current_state)
            q_value_target[action] = target
            q_value_target_array.append(q_value_target)


        q_value_policy_array = torch.stack(q_value_policy_array)
        q_value_target_array = torch.stack(q_value_target_array)
        loss = self.loss_function(q_value_policy_array, q_value_target_array)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def testing(self, episodes):
        env = gym.make("LunarLander-v2", render_mode="human")
        observation, info = env.reset(seed=42)

        n_states = env.observation_space.shape[0]
        n_action = env.action_space.n

        policy_dqn = DQN(n_action,n_states).to(self.device)
        policy_dqn.load_state_dict(torch.load("LunarLander_DQL.pt"))
        policy_dqn.eval()    

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            while(not terminated and not truncated):  
                with torch.no_grad():
                    action = policy_dqn(torch.from_numpy(state).float().to(self.device)).argmax().item()

                observation, reward, terminated, truncated, info = env.step(action)

        env.close()



def main():
    lunarLander = LunarLanderDQL()
    lunarLander.train(2000)
    


if __name__ == "__main__":
    main()
