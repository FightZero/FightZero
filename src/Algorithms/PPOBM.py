# an implementation of PPO algorithm based on black mamba weights
import os
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import BatchSampler, RandomSampler
from typing import Tuple
import numpy as np

# this class implements an actor critic model with linear networks
class ActorCritic(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        # save dimensions
        self.d_state = state_dimension
        self.d_action = action_dimension
        # create actor network
        self.actor = nn.Sequential(
            nn.Linear(self.d_state, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 600),
            nn.LeakyReLU(),
            nn.Linear(600, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, self.d_action),
            nn.Softmax(dim=1)
        )

        if not os.path.exists(os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'is_loaded_bm.txt')):
            self.actor.load_state_dict(torch.load(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "blackmamba.pt"
                )
            ))
            print("BlackMamba checkpoint loaded")
            with open(os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'is_loaded_bm.txt'), 'w') as f:
                f.write('')
            f.close()

        # create critic network
        self.critic = nn.Sequential(
            nn.Linear(self.d_state, 800),
            nn.LeakyReLU(),
            nn.Linear(800, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Empty forward function
        """
        return x

    def action(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and log probs
        """
        # get probabilities of actions
        probs = self.actor(state)
        dist = Categorical(probs=probs)
        # sample an action
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.detach(), logprob.detach()

    def evaluate(self, state, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates an action
        """
        # get probabilities of actions
        probs = self.actor(state)
        dist = Categorical(probs=probs)
        # get distribution entropy and log probs of chosen action
        entropy = dist.entropy()
        logprob = dist.log_prob(action)
        # logprob = dist.log_prob(action).diagonal().view(action.shape)
        # get critic value
        critics = self.critic(state)
        return entropy, logprob, critics


# this structure stores buffer info for PPO
class PPOBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def reset(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def isEmpty(self):
        return len(self.actions) <= 0


# this class implements PPO model
# with actor and critic updated together
# class PPO(object):
#     def __init__(self, 
#         state_dimension, action_dimension,
#         lr_actor, lr_critic,
#         num_epochs, discount,
#         eps_clip, batch_size,
#         max_grad_norm, train
#     ):
#         self.discount = discount
#         self.num_epochs = num_epochs
#         self.eps_clip = eps_clip
#         self.lr_actor = lr_actor
#         self.lr_critic = lr_critic
#         self.batch_size = batch_size
#         self.max_grad_norm = max_grad_norm
#         self.training = train
#         self.iter_count = 0

#         # create buffer
#         self.buffer = PPOBuffer()
#         # select running environment for train
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         # create actor critic model
#         self.AC = ActorCritic(state_dimension, action_dimension).to(self.device)
#         # set optimizer
#         self.optim = Adam([
#             {"params": self.AC.actor.parameters(), "lr": lr_actor},
#             {"params": self.AC.critic.parameters(), "lr": lr_critic}
#         ])
#         # self.optim = RMSprop(self.AC.parameters(), lr)
#         # set saved model
#         self.AC_saved = ActorCritic(state_dimension, action_dimension).to(self.device)
#         self.AC_saved.load_state_dict(self.AC.state_dict())
#         self.AC_saved.eval()
#         # set loss function
#         self.loss = nn.MSELoss()

#     def action(self, state):
#         """
#         Choose next action
#         """
#         with torch.no_grad():
#             # get new action from actor
#             state = torch.FloatTensor(state).to(self.device)
#             action, logprob = self.AC_saved.action(state)
#         # store into buffer
#         if self.training:
#             self.buffer.states.append(state)
#             self.buffer.actions.append(action)
#             self.buffer.logprobs.append(logprob)
#         return action.cpu().item()

#     def save(self, filename):
#         """
#         Save current network to file path
#         """
#         torch.save(self.AC_saved.state_dict(), filename)

#     def load(self, filename):
#         """
#         Load network from file path
#         """
#         self.AC.load_state_dict(torch.load(filename, map_location=lambda storage, _: storage))
#         self.AC_saved.load_state_dict(torch.load(filename, map_location=lambda storage, _: storage))

#     def train(self, writer : SummaryWriter):
#         """
#         Update policy
#         """
#         if not self.training: return
#         if self.buffer.isEmpty(): return
#         rewards = []
#         reward_disc = 0.0
#         for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#             # if is terminal state, set reward to 0
#             if is_terminal:
#                 reward_disc = 0.0
#             reward_disc = reward + (self.discount * reward_disc)
#             rewards.insert(0, reward_disc)
#         rewards = rewards[::-1][1:]
#         # normalize the rewards
#         target_values = torch.FloatTensor(rewards).to(self.device)
#         target_values = (target_values - target_values.mean()) / (target_values.std() + 1e-8)
#         # target_values = target_values.view(-1, 1)
#         # convert list to tensor
#         old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
#         old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
#         old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
#         # start training
#         self.AC.train()
#         for _ in range(self.num_epochs):
#             # Evaluating old actions and values
#             entropy, logprob, critics = self.AC.evaluate(old_states, old_actions)
#             # match state_values tensor dimensions with rewards tensor
#             state_values = torch.squeeze(critics)
            
#             # Finding the ratio (pi_theta / pi_theta__old)
#             ratios = torch.exp(logprob - old_logprobs.detach())

#             # Finding Surrogate Loss
#             advantages = target_values - state_values.detach()   
#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

#             # final loss of clipped objective PPO
#             loss = -torch.min(surr1, surr2) + 0.5*self.loss(state_values, target_values) - 0.01*entropy
            
#             # take gradient step
#             self.optim.zero_grad()
#             loss.mean().backward()
#             self.optim.step()
#             writer.add_scalar("PPO/Loss", loss.cpu().detach().mean().item(), self.iter_count)
#             writer.add_scalar("PPO/Advantage", advantages.cpu().detach().mean().item(), self.iter_count)
#             self.iter_count += 1
#         self.AC.eval()
#         # save weights after training
#         self.AC_saved.load_state_dict(self.AC.state_dict())
#         self.AC_saved.eval()
#         # clear buffer
#         self.buffer.reset()

#     def update(self, reward, is_terminal):
#         """
#         Update buffer
#         """
#         if not self.training: return
#         self.buffer.rewards.append(reward)
#         self.buffer.is_terminals.append(is_terminal)



# this class implements PPO model
# with actor and critic updated individually
class PPO(object):
    def __init__(self, 
        state_dimension, action_dimension,
        lr_actor, lr_critic,
        num_epochs, discount,
        eps_clip, batch_size,
        max_grad_norm, train
    ):
        self.discount = discount
        self.num_epochs = num_epochs
        self.eps_clip = eps_clip
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.training = train
        if os.path.exists('iter_count.txt'):
            with open('iter_count.txt','r') as f:
                self.iter_count=int(f.read())
        else:
            self.iter_count = 0

        # create buffer
        self.buffer = PPOBuffer()
        # select running environment for train
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # create actor critic model
        self.AC = ActorCritic(state_dimension, action_dimension).to(self.device)
        # set optimizer
        self.optim_actor = Adam(self.AC.actor.parameters(), lr_actor)
        self.optim_critic = Adam(self.AC.critic.parameters(), lr_critic)
        # set saved model
        self.AC_saved = ActorCritic(state_dimension, action_dimension).to(self.device)
        self.AC_saved.load_state_dict(self.AC.state_dict())
        self.AC_saved.eval()
        # set loss function
        self.loss = nn.MSELoss()

    def action(self, state):
        """
        Choose next action
        """
        with torch.no_grad():
            # get new action from actor
            state = torch.FloatTensor(state).to(self.device)
            action, logprob = self.AC_saved.action(state)
        # store into buffer
        if self.training:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(logprob)
        return action.cpu().item()

    def save(self, filename):
        """
        Save current network to file path
        """
        torch.save((
            self.AC_saved.state_dict(),
            self.optim_actor.state_dict(),
            self.optim_critic.state_dict(),
        ), filename)

    def load(self, filename):
        """
        Load network from file path
        """
        states, opt1, opt2 = torch.load(filename, map_location=lambda storage, _: storage)
        self.AC.load_state_dict(states)
        self.AC_saved.load_state_dict(states)
        self.optim_actor.load_state_dict(opt1)
        self.optim_critic.load_state_dict(opt2)

    def train(self, writer : SummaryWriter):
        """
        Update policy
        """
        if not self.training: return
        if self.buffer.isEmpty(): return
        rewards = []
        reward_disc = 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            # if is terminal state, set reward to 0
            if is_terminal:
                reward_disc = 0.0
            reward_disc = reward + (self.discount * reward_disc)
            rewards.insert(0, reward_disc)
        length = len(rewards)
        # normalize the rewards
        target_values = torch.FloatTensor(rewards)
        target_values = (target_values - target_values.mean()) / (target_values.std() + 1e-8)
        target_values = target_values.view(-1, 1)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states[:length], dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:length], dim=0)).view(-1, 1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:length], dim=0)).view(-1, 1).detach()
        # start training
        self.AC.train()
        for _ in range(self.num_epochs):
            for indices in BatchSampler(RandomSampler(range(length)), batch_size=self.batch_size, drop_last=True):
                target_values_gpu = target_values[indices].to(self.device)
                old_states_gpu = old_states[indices].to(self.device)
                old_actions_gpu = old_actions[indices].to(self.device)
                old_logprobs_gpu = old_logprobs[indices].to(self.device)
                # get critics
                _, logprob, state_values = self.AC.evaluate(old_states_gpu, old_actions_gpu)
                # compute advantages
                advantages = (target_values_gpu - state_values).detach()
                # find the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprob - old_logprobs_gpu)
                # find Surrogate Loss (Clipped Surrogate Objective)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # compute actor loss
                loss_actor = -torch.min(surr1, surr2).mean()
                # optimize actor
                self.optim_actor.zero_grad()
                loss_actor.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(self.AC.actor.parameters(), max_norm=self.max_grad_norm)
                self.optim_actor.step()
                # compute critic loss
                loss_critic = self.loss(state_values, target_values_gpu)
                self.optim_critic.zero_grad()
                loss_critic.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(self.AC.critic.parameters(), max_norm=self.max_grad_norm)
                self.optim_critic.step()
                # log in tensorboard
                writer.add_scalar("PPO/Loss Actor", loss_actor.cpu().detach().item(), self.iter_count)
                writer.add_scalar("PPO/Loss Critic", loss_critic.cpu().detach().mean().item(), self.iter_count)
                writer.add_scalar("PPO/Advantage", advantages.cpu().detach().mean().item(), self.iter_count)
                self.iter_count += 1
                with open('iter_count.txt','w') as f:
                    f.write(str(self.iter_count))
        self.AC.eval()
        # save weights after training
        self.AC_saved.load_state_dict(self.AC.state_dict())
        self.AC_saved.eval()
        # clear buffer
        self.buffer.reset()

    def update(self, reward, is_terminal):
        """
        Update buffer
        """
        if not self.training: return
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)