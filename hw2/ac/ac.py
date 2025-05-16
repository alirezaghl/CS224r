import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(nn.Linear(obs_shape[0], hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, num_critics,
                 hidden_dim):
        super().__init__()

        self.critics = nn.ModuleList([nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
            for _ in range(num_critics)])

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


class ACAgent:
    def __init__(self, obs_shape, action_shape, device, lr,
                 hidden_dim, num_critics, critic_target_tau, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(obs_shape, action_shape,
                           hidden_dim).to(device)

        self.critic = Critic(obs_shape, action_shape,
                             num_critics, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape, action_shape,
                                    num_critics, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update_critic(self, replay_iter):
        '''
        This function updates the critic and target critic parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the critic
                 loss, or the mean Bellman targets.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        ### YOUR CODE HERE ###
        random_indices = random.sample(range(len(self.critic_target.critics)), 2)
        selected_target_critics = [self.critic_target.critics[i] for i in random_indices]
        critic1, critic2 = selected_target_critics
        next_dist = self.actor(next_obs)
        next_action = next_dist.sample()
        estimated_q_1 = critic1(next_obs, next_action)
        estimated_q_2 = critic2(next_obs, next_action)
        minimum_q = torch.min(estimated_q_1, estimated_q_2)
        y = reward.unsqueeze(-1) + discount.unsqueeze(-1) * minimum_q.detach()

        total_loss = 0
        for i in range(len(self.critic.critics)):
            critic = self.critic.critics[i]
            q = critic(obs, action)
            critic_loss = F.mse_loss(q, y)
            total_loss += critic_loss

        self.critic_opt.zero_grad()
        total_loss.backward()
        self.critic_opt.step()
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        metrics['critic_loss'] = total_loss.item()        
        return metrics

        



        #####################
        return metrics

    def update_actor(self, replay_iter):
        '''
        This function updates the policy parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the actor
                 loss.
        '''
        metrics = dict()

        batch = next(replay_iter)
        obs, _, _, _, _ = utils.to_torch(
            batch, self.device)

        ### YOUR CODE HERE ###
        total_q = 0
        dist = self.actor(obs)
        action = dist.sample()
        for i in range(len(self.critic.critics)):
             critic = self.critic.critics[i]
             q = critic(obs, action)
             total_q += q
        actor_loss = -(total_q / len(self.critic.critics)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        metrics['actor_loss'] = actor_loss.item()
        return metrics

    def bc(self, replay_iter):
        '''
        This function updates the policy with end-to-end
        behavior cloning

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the loss.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _ = utils.to_torch(batch, self.device)

        ### YOUR CODE HERE ###
        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        bc_loss = -log_prob.mean()
    
    # Optimize the actor
        self.actor_opt.zero_grad()
        bc_loss.backward()
        self.actor_opt.step()
        metrics['actor_loss'] = bc_loss.item()
        return metrics