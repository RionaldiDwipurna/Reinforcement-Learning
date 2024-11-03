import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import multiprocessing as mp
import functools
from warnings import filterwarnings

# WORKS DONT TOUCH (pretty good)
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_action):
        super().__init__()
        self.shared_1 = nn.Linear(input_dim, hidden_dim)
        self.shared_2 = nn.Linear(hidden_dim, hidden_dim)

        # policy head
        self.out_policy = nn.Linear(hidden_dim, num_action)

    def forward(self, x):
        x = self.shared_1(x)
        x = F.gelu(x)
        x = self.shared_2(x)
        x = F.gelu(x)

        policy_network = self.out_policy(x)

        return policy_network

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.shared_1 = nn.Linear(input_dim, hidden_dim)
        self.shared_2 = nn.Linear(hidden_dim, hidden_dim)

        # value head
        self.out_value = nn.Linear(hidden_dim,1)

    def forward(self, x):
        x = self.shared_1(x)
        x = F.gelu(x)
        x = self.shared_2(x)
        x = F.gelu(x)

        value_network = self.out_value(x)

        return value_network

class TrainingPPO:
    def __init__(self):
        self.cum_reward = []
        self.At = []
        self.lambda_GAE = 0.95
        self.adam_step = 3e-4
        self.epsilon = 0.2
        self.discount = 0.99
        self.epoch = 10
        self.beta = 3
        self.old_policy = None
        self.current_policy = None
        self.c1 = 1
        self.c2 = 0.01
        self.max_step_per_eps = 1000
        self.num_eps = 1000
        self.n_action = None
        self.n_states = None
        self.logger = {
            'eps': 0,
            'iter': 0,
            'reward': [],
            'loss': [],
        }

        self.render = False
        self.env = gym.make("LunarLander-v2", render_mode='human' if self.render else None)
        self.n_states = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n

        self.actor_model = Actor(self.n_states , 128, self.n_action)
        self.critic_model = Critic(self.n_states , 128)
        # self.actor = 1

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
    
    def calculate_RTG(self, reward_batched,dones):
        lst_total_r = []
        total_r = 0  # Initialize total return
    
        # Iterate backwards to accumulate returns
        for t in reversed(range(len(reward_batched))):
            total_r = reward_batched[t] + self.discount * total_r  # Use the immediate reward and the discounted total return
            lst_total_r.append(total_r)

        # Reverse the list to maintain the correct order
        lst_total_r.reverse()
    
        return torch.tensor(lst_total_r)  # Convert to tensor before returning

        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(reward_batched), reversed(dones)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.discount * discounted_reward)
        #     rewards.insert(0, discounted_reward)
        
        # # Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # return rewards

        # batch_rtgs = []
        # for ep_rews in reversed(reward_batched):
        #     discounted_reward = 0
        #     for rew in reversed(ep_rews):
        #         discounted_reward = rew + discounted_reward + self.discount
        #         batch_rtgs.insert(0, discounted_reward)
        # batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        # return batch_rtgs

    def compute_gae(self, rewards, values, dones):
        lambda_discount = self.lambda_GAE * self.discount
        returns = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.discount * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + lambda_discount * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
        # Normalize advantages
        returns = torch.stack(returns)
        advantages = returns - values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def calculate_At(self, lst_delta_t,lst_dones, V, normalize=True):
        lambda_discount = self.lambda_GAE * self.discount
        lst_at_val = []
        returns = []
    
        advantage = 0
    
        for t in reversed(range(len(lst_delta_t))):
            advantage = lst_delta_t[t] + lambda_discount * advantage * (1 - lst_dones[t])
            lst_at_val.append(advantage)
            # Calculate return (Vt target: GAE + V[t])
            returns.append(advantage + V[t].detach())

        lst_at_val.reverse()
        returns.reverse()

        lst_at_val = torch.stack(lst_at_val)
        returns = torch.stack(returns)

        if normalize:
            lst_at_val = (lst_at_val - torch.mean(lst_at_val)) / (torch.std(lst_at_val) + 1e-8) #prevent zero
    
        return lst_at_val, returns

    def calculate_delta_t(self, reward_lst, value_lst, dones_lst):
        # lst_delta_t = [] 
        # state = curr_lst # Get current state
        # state_next = next_lst # Get next state

        # _, value_next = model_current(state_next)
        # _, value_current = model_current(state)
        # value_next = value_next.view(-1)
        # value_current = value_current.view(-1)

        value_current = value_lst[:-1]
        value_next = value_lst[1:]
        value_current = torch.cat((value_current, torch.tensor([0.0])))
        value_next = torch.cat((value_next, torch.tensor([0.0])))
        lst_delta_t = reward_lst + self.discount * value_next * (1 - dones_lst) - value_current


        return lst_delta_t



    def clip_loss(self, At, log_prob_current, log_prob_old):

        rt = torch.exp(log_prob_current - log_prob_old)
        clipped_val = torch.clamp(rt, 1 - self.epsilon, 1 + self.epsilon)
        min_val = torch.min(rt * At, clipped_val * At).mean()
        return -min_val

    def value_loss(self, value_current, returns):
        return F.mse_loss(value_current, returns)

    def entropy_bonus(self, logits):
        prob = F.softmax(logits, dim=-1) # pi(a | t)

        return -torch.sum(prob * torch.log(prob + 1e-10), dim=-1).mean()
    
    def total_loss(self, clip_l, val_l, entropy_l):
        return clip_l + self.c1 * val_l - self.c2 * entropy_l
    
    def get_action(self, current_state, actor_model, critic_model):
        policy = actor_model(current_state)
        value = critic_model(current_state)
        probs = torch.softmax(policy, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample() # sample an action based on the distribution of actions, ex: prob = [0.1, 0.5, 0.2, 0.3], action selected = 1 (most likely)
        log_prob = dist.log_prob(action)
        # log(PI(at | st)): log prob of selected action given state

        return action, log_prob, value.squeeze(-1)

    def learn(self, curr_lst_batched, action_lst_batched, actor_model, critic_model):

        policy = actor_model(curr_lst_batched)
        V = critic_model(curr_lst_batched)


        probs = torch.softmax(policy, dim=-1)

        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(action_lst_batched)
        # log(PI(at | st)): log prob of selected action given state, ex: action = 2 , prob = [0.1, 0.5, 0.2, 0.3], pi(at |st) = 0.2

        return policy, log_prob, V.squeeze(-1)

    def run_policy(self, actor, critic, env, value_lst_batched, probs_lst_batched, curr_lst_batched, action_lst_batched, reward_lst_batched, dones_lst_batched, pid):

        curr_lst = []
        # next_lst = []
        action_lst = []
        reward_lst = []
        probs_lst = []
        value_lst = []
        dones_lst = []

        terminated = False
        truncated = False
        # render=False
        current_state = torch.Tensor(env.reset()[0])

        # while (not terminated and not truncated):
        for eps in range(self.max_step_per_eps):
            with torch.no_grad():
                action, probs, value = self.get_action(current_state,actor_model=actor, critic_model=critic)

            next_state, reward, terminated, truncated, _ = env.step(action.item())


            # current_state = torch.tensor(current_state, dtype=torch.float32)
            # probs = torch.tensor(probs, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            # action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)

            value_lst.append(value)
            probs_lst.append(probs)
            curr_lst.append(current_state)
            # next_lst.append(next_state)
            action_lst.append(action)
            reward_lst.append(reward)
            dones_lst.append(torch.tensor(terminated or truncated, dtype=torch.int32))

            if reward == 100:
                print("reward 100 get")


            current_state = next_state

            if terminated or truncated:
                break

        

        value_lst = torch.stack(value_lst)
        probs_lst = torch.stack(probs_lst)
        curr_lst = torch.stack(curr_lst)
        # next_lst = torch.stack(next_lst)
        action_lst = torch.stack(action_lst)
        reward_lst = torch.stack(reward_lst)
        dones_lst =  torch.stack(dones_lst)

        value_lst_batched[pid] = value_lst
        probs_lst_batched[pid] = probs_lst
        curr_lst_batched[pid] = curr_lst
        # next_lst_batched[pid] = next_lst
        action_lst_batched[pid] = action_lst
        reward_lst_batched[pid] = reward_lst
        dones_lst_batched[pid] = dones_lst


    def multiple_policy(self, actor, critic, env):
        n_cpu = mp.cpu_count()
        manager = mp.Manager()
        probs_lst_batched = manager.dict()
        curr_lst_batched = manager.dict()
        # next_lst_batched = manager.dict()
        action_lst_batched = manager.dict()
        reward_lst_batched = manager.dict()
        dones_lst_batched = manager.dict()
        value_lst_batched = manager.dict()

        processes = []
        pool = mp.Pool(processes=n_cpu)

        # Debug
        # self.run_policy(policy,env, probs_lst_batched, curr_lst_batched, next_lst_batched, action_lst_batched, reward_lst_batched, 1)

        pool.map(functools.partial(self.run_policy, actor, critic,env, value_lst_batched, probs_lst_batched, curr_lst_batched, action_lst_batched, reward_lst_batched, dones_lst_batched), range(n_cpu))

        value_lst_batched   = [value_lst_batched[i] for i in range(n_cpu)]
        probs_lst_batched   = [probs_lst_batched[i] for i in range(n_cpu)]
        curr_lst_batched   = [curr_lst_batched[i] for i in range(n_cpu)]
        # next_lst_batched   = [next_lst_batched[i] for i in range(n_cpu)]
        action_lst_batched = [action_lst_batched[i] for i in range(n_cpu)]
        reward_lst_batched = [reward_lst_batched[i] for i in range(n_cpu)]
        dones_lst_batched = [dones_lst_batched[i] for i in range(n_cpu)]

        value_lst_batched  = torch.cat(list(value_lst_batched))
        probs_lst_batched  = torch.cat(list(probs_lst_batched))
        curr_lst_batched   = torch.cat(list(curr_lst_batched))
        # next_lst_batched   = torch.cat(list(next_lst_batched))
        action_lst_batched = torch.cat(list(action_lst_batched))
        reward_lst_batched = torch.cat(list(reward_lst_batched))
        dones_lst_batched = torch.cat(list(dones_lst_batched))


        return value_lst_batched, probs_lst_batched, curr_lst_batched, action_lst_batched, reward_lst_batched, dones_lst_batched
    
    def run_non_paralel(self, actor, critic, env):
        probs_lst_batched   = []
        curr_lst_batched    = []
        # next_lst_batched    = []
        action_lst_batched  = []
        reward_lst_batched  = []
        dones_lst_batched   = []
        value_lst_batched   = []

        render=False

        # while (not terminated and not truncated):
        current_state = torch.Tensor(env.reset()[0])
        terminated = False
        truncated = False
        for steps in range(self.max_step_per_eps):
            with torch.no_grad():
                action, probs, value = self.get_action(current_state,actor, critic)

            next_state, reward, terminated, truncated, _ = env.step(action.item())


            # current_state = torch.tensor(current_state, dtype=torch.float32)
            # probs = torch.tensor(probs, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            # action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)

            value_lst_batched.append(value)
            probs_lst_batched.append(probs)
            curr_lst_batched.append(current_state)
            # next_lst_batched.append(next_state)
            action_lst_batched.append(action)
            reward_lst_batched.append(reward)
            dones_lst_batched.append(torch.tensor(terminated or truncated, dtype=torch.int32))

            if reward == 100:
                print("reward 100 get")

            current_state = next_state

            if terminated or truncated:
                break

        # GAE
        with torch.no_grad():
            _,_,pred_value = self.get_action(current_state,actor,critic)
        value_lst_batched.append(pred_value)

        value_lst_batched = torch.stack(value_lst_batched)
        probs_lst_batched = torch.stack(probs_lst_batched)
        curr_lst_batched  = torch.stack(curr_lst_batched)
        # next_lst_batched = torch.stack(next_lst_batched)
        action_lst_batched = torch.stack(action_lst_batched)
        reward_lst_batched = torch.stack(reward_lst_batched)
        dones_lst_batched =  torch.stack(dones_lst_batched)

        return value_lst_batched, probs_lst_batched, curr_lst_batched, action_lst_batched, reward_lst_batched, dones_lst_batched


    def train(self, render=False, paralel=False, with_gae=True):

        env = self.env
        actor_model = self.actor_model
        critic_model = self.critic_model
        optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=self.adam_step)
        optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=self.adam_step)

        # if old_model != None:
        #     old_model.load_state_dict(self.old_policy.state_dict())

        for i in range(self.num_eps):

            # with torch.autograd.set_detect_anomaly(True):
            if paralel:
                value_lst_batched, probs_lst_batched, curr_lst_batched, action_lst_batched, reward_lst_batched, dones_lst_batched = self.multiple_policy(actor_model, critic_model, env)

                # Calculate the temporal difference
                lst_delta_t = self.calculate_delta_t(reward_lst=reward_lst_batched, value_lst=value_lst_batched, dones_lst=dones_lst_batched)

                # Calculate the Generalized Advantage Estimation (GAE)
                lst_at_val, returns = self.calculate_At(lst_delta_t=lst_delta_t, lst_dones=dones_lst_batched, V=value_lst_batched)
            else:
                if with_gae:
                    value_lst_batched, probs_lst_batched, curr_lst_batched, action_lst_batched, reward_lst_batched, dones_lst_batched = self.run_non_paralel(actor_model,critic_model, env)
                    lst_at_val, returns = self.compute_gae(rewards=reward_lst_batched, values=value_lst_batched ,dones=dones_lst_batched)
                else:
                    value_lst_batched, probs_lst_batched, curr_lst_batched, action_lst_batched, reward_lst_batched, dones_lst_batched = self.run_non_paralel(actor_model,critic_model, env)
                    returns = self.calculate_RTG(reward_lst_batched, dones_lst_batched)
                    lst_at_val = returns - value_lst_batched
                    lst_at_val = (lst_at_val - lst_at_val.mean())/ (lst_at_val.std() + 1e-10)
                    

            if i % 20 == 0:
                print(f"Reward episode {i}: {torch.sum(reward_lst_batched).item()} max reward: {torch.max(reward_lst_batched).item()}")
                # print(f"Average Reward for {i} episode: {torch.mean(reward_lst_batched).item()} | Max Reward: {torch.max(reward_lst_batched).item()}")


            value_loss_avg = 0
            ppo_loss_avg = 0
            total_loss_avg = 0


            for epoch in range(self.epoch):

                # Calculate the new Value and log probability
                logits, new_log_prob, V = self.learn(curr_lst_batched=curr_lst_batched, action_lst_batched=action_lst_batched,actor_model=actor_model,critic_model=critic_model)


                rt = torch.exp(new_log_prob - probs_lst_batched)
                clipped_val = torch.clamp(rt, 1 - self.epsilon, 1 + self.epsilon)
                L_CLIP = (- torch.min(rt * lst_at_val, clipped_val * lst_at_val)).mean()
                L_value = nn.MSELoss()(V,returns)

                # Calculate the ppo loss
                # L_CLIP = self.clip_loss(At=lst_at_val, log_prob_current=new_log_prob, log_prob_old=probs_lst_batched)

                # Calculate the value loss
                # L_value = self.value_loss(V, returns)

                # Calculate the entropy bonus
                # entropy_bonus = self.entropy_bonus(logits)

                # Calculate the total loss
                # total_loss = self.total_loss(clip_l=L_CLIP, val_l = L_value, entropy_l= entropy_bonus)

                value_loss_avg += L_value
                ppo_loss_avg += L_CLIP
                # total_loss_avg += total_loss

                optimizer_actor.zero_grad() 
                L_CLIP.backward()
                optimizer_actor.step()


                optimizer_critic.zero_grad() 
                L_value.backward()
                optimizer_critic.step()
            
            # print(f"PPO loss for episode: {i}: {L_CLIP/self.epoch}")
            # print(f"Value loss for episode: {i}: {L_value/self.epoch}")
            # print(f"Total loss for episode: {i}: {total_loss/self.epoch}")
# 
        torch.save(actor_model.state_dict(), './ppo_act_model.pth')
        torch.save(critic_model.state_dict(), './ppo_crit_model.pth')


            

def main():
    filterwarnings(action='ignore', category=DeprecationWarning)
    torch.manual_seed(1337)
    PPOTraining = TrainingPPO()
    PPOTraining.train()

if __name__ == "__main__":
    main()