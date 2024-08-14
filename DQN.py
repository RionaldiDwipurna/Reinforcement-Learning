#reference code
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
        self.fc1 = nn.Linear(state_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_shape)
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)  
        return x


class LunarLanderDQL():
    LEARNING_RATE = 0.0005
    DISCOUNT_FACTOR = 0.95
    SYNC_TIME = 10000

    REPLAY_MEMORY_SIZE = 100000
    MIN_REPLAY_MEMORY_SIZE = 500
    REPLAY_MEMORY_BATCH = 128
    NUM_DIVISION = 40
    OPTIMIZE_EVERY = 20
    # OPTIMIZE_EVERY = 'episode'

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

    def initialize_discrete_param(self, env):
        self.x_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.NUM_DIVISION) 
        self.y_pos = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.NUM_DIVISION) 
        self.x_vel = np.linspace(env.observation_space.low[2], env.observation_space.high[2], self.NUM_DIVISION) 
        self.y_vel = np.linspace(env.observation_space.low[3], env.observation_space.high[3], self.NUM_DIVISION) 
        self.angle = np.linspace(env.observation_space.low[4], env.observation_space.high[4], self.NUM_DIVISION) 
        self.ang_v = np.linspace(env.observation_space.low[5], env.observation_space.high[5], self.NUM_DIVISION) 


    def train(self, episodes, render=False, model_path=None):
        env = gym.make("LunarLander-v2", render_mode='human' if render else None)

        n_states = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.replay_mem = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        policy_dqn = DQN(n_action,n_states).to(self.device)
        target_dqn = DQN(n_action,n_states).to(self.device)

        if model_path != None:
            policy_dqn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            policy_dqn.eval()    

        epsilon = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995

        epsilon_stats = []
        rewards_stats = np.zeros(episodes)


        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.LEARNING_RATE)

        self.initialize_discrete_param(env)


        step = 0

        best_rewards = -1000

        for i in range(episodes):
            current_state = env.reset()[0]
            terminated = False
            truncated = False

            curr_reward = []
            rewards = 0
            count_optimize = 0
            
            while(not terminated and not truncated):

                if random.random() < epsilon: #Epsilon greedy policy
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # action = policy_dqn(torch.from_numpy(current_state).float().to(self.device)).argmax().item()
                        action = policy_dqn(self.input_discrete(current_state)).argmax().item()

                next_state, reward, terminated, truncated, info = env.step(action)


                self.replay_mem.append([current_state, next_state, action, reward, terminated, truncated, info])
                rewards += reward

                current_state = next_state

                step += 1

                if reward == 100:
                    print("reward 100 get")
                    rewards_stats[i] = 100

                # if (i+1) % 500 == 0:
                #     torch.save(policy_dqn.state_dict(), f"LunarLander_DQL_{i}.pt")
                if rewards > best_rewards:
                    best_rewards = rewards
                    print(f'best_rewards: {best_rewards}')
                    torch.save(policy_dqn.cpu().state_dict(),f"LunarLander_DQL_{i}.pt")
                    # torch.save(policy_dqn.state_dict(), f"LunarLander_DQL_{i}.pt")


                if len(self.replay_mem) > self.MIN_REPLAY_MEMORY_SIZE:
                    if isinstance(self.OPTIMIZE_EVERY, int) and count_optimize % self.OPTIMIZE_EVERY == 0:
                        mini_batch = random.sample(self.replay_mem, self.REPLAY_MEMORY_BATCH)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # epsilon = max(epsilon - 1/episodes, 0)
                        epsilon = max(epsilon_end, epsilon * epsilon_decay)

                    if step % self.SYNC_TIME == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

                count_optimize += 1


            curr_reward.append(rewards)
            print(f'episodes: {i} / {episodes} rewards avg: {np.mean(curr_reward)} reward sum: {rewards}')


            if len(self.replay_mem) > self.MIN_REPLAY_MEMORY_SIZE:
                if self.OPTIMIZE_EVERY == 'episode':
                    mini_batch = random.sample(self.replay_mem, self.REPLAY_MEMORY_BATCH)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # epsilon = max(epsilon - 1/episodes, 0)
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)

                if step % self.SYNC_TIME == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())


        env.close()
        
        torch.save(policy_dqn.cpu().state_dict(), "LunarLander_DQL.pt")

        plt.figure(1)

        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_stats[max(0, x-100):(x+1)])
        plt.subplot(121)
        plt.plot(sum_rewards)
        
        plt.subplot(122)
        plt.plot(epsilon_stats)
        
        plt.savefig('lunar_lander.png')



    def optimize(self,mini_batch, policy_dqn, target_dqn):
        q_value_policy_array = []
        q_value_target_array = []
        for current_state, next_state, action, reward, terminated, truncated, info in mini_batch:
            # current_state = torch.from_numpy(current_state).float().to(self.device)
            # next_state =    torch.from_numpy(next_state).float().to(self.device)
            current_state = self.input_discrete(current_state) 
            next_state = self.input_discrete(next_state) 

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

    def testing(self, episodes,model_path):
        env = gym.make("LunarLander-v2", render_mode="human")
        self.initialize_discrete_param(env)

        n_states = env.observation_space.shape[0]
        n_action = env.action_space.n


        policy_dqn = DQN(n_action,n_states).to(self.device)
        policy_dqn.load_state_dict(torch.load(model_path, map_location=self.device))
        policy_dqn.eval()    

        for i in range(episodes):
            current_state = env.reset()[0]  
            terminated = False      
            truncated = False                

            while(not terminated and not truncated):  
                with torch.no_grad():
                    discretized_state = self.input_discrete(current_state)
                    q_values = policy_dqn(discretized_state)
                    action = q_values.argmax().item()

                next_state, reward, terminated, truncated, info = env.step(action)
                current_state = next_state
                print(reward)

        env.close()

def main():
    lunarLander = LunarLanderDQL()
    # lunarLander.train(episodes=500, model_path="LunarLander_DQL_76.pt", render=False)
    lunarLander.testing(episodes=5, model_path="LunarLander_DQL_2387.pt")
    


if __name__ == "__main__":
    main()
