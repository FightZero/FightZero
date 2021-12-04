# an implementation of PPO algorithm
# reference to: https://github.com/nikhilbarhate99/PPO-PyTorch
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from typing import Tuple

# this class implements an actor critic model with linear networks
class ActorCritic(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        # save dimensions
        self.d_state = state_dimension
        self.d_action = action_dimension
        # create actor network
        self.actor = nn.Sequential(
            nn.Linear(self.d_state, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.d_action),
            nn.Softmax(dim=1)
        )
        # create critic network
        self.critic = nn.Sequential(
            nn.Linear(self.d_state, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
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


# this class implements PPO model
class PPO:
    def __init__(self, 
        state_dimension, action_dimension,
        lr_actor, lr_critic,
        num_epochs, discount,
        eps_clip
    ):
        self.discount = discount
        self.num_epochs = num_epochs
        self.eps_clip = eps_clip
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # create buffer
        self.buffer = PPOBuffer()
        # select running environment for train
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # create actor critic model
        self.AC = ActorCritic(state_dimension, action_dimension).to(self.device)
        # set optimizer
        self.optim = Adam([
            {"params": self.AC.actor.parameters(), "lr": lr_actor},
            {"params": self.AC.critic.parameters(), "lr": lr_critic},
        ])
        # set saved model
        self.AC_saved = ActorCritic(state_dimension, action_dimension).to(self.device)
        self.AC_saved.load_state_dict(self.AC.state_dict())
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
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        return action.cpu().item()

    def save(self, filename):
        """
        Save current network to file path
        """
        torch.save(self.AC_saved.state_dict(), filename)

    def load(self, filename):
        """
        Load network from file path
        """
        self.AC.load_state_dict(torch.load(filename, map_location=lambda storage, _: storage))
        self.AC_saved.load_state_dict(torch.load(filename, map_location=lambda storage, _: storage))

    def update(self):
        """
        Update policy
        """
        rewards = []
        reward_disc = 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards, self.buffer.is_terminals)):
            # if is terminal state, set reward to 0
            if is_terminal:
                reward_disc = 0.0
            reward_disc = reward + (self.gamma * reward_disc)
            rewards.append(reward_disc)
        # normalize the rewards
        rewards = torch.FloatTensor(reversed(rewards)).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        # start training
        for _ in range(self.num_epochs):
            # get critics
            entropy, logprob, critics = self.AC.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(critics)
            # find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprob - old_logprobs.detach())
            # find Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # compute loss
            loss = -torch.min(surr1, surr2) + \
                0.5*self.loss(state_values, rewards) -\
                0.01*entropy
            # optimize
            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()
        # save weights after training
        self.AC_saved.load_state_dict(self.AC.state_dict())
        # clear buffer
        self.buffer.reset()