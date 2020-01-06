import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.6
BETA = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        self.is_ER_prioritized = False
        print("loaded")
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        ## convert to tensors
        t_state = torch.from_numpy(np.vstack([state])).float().to(device)
        t_action = torch.from_numpy(np.vstack([action])).long().to(device)
        t_reward = torch.from_numpy(np.vstack([reward])).float().to(device)
        t_next_state = torch.from_numpy(np.vstack([next_state])).float().to(device)
        t_done = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)
        ##
        Q_target, Q_expected = self.get_target_local_Qs( 
            (t_state, t_action, t_reward, t_next_state, t_done), GAMMA)
        self.memory.add_TD_error((Q_target-Q_expected).cpu().detach().numpy()[0][0])
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        self.qnetwork_local.train() #sets training mode, that is all - opposite to eval
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def get_target_local_Qs(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        return (Q_targets, Q_expected)
        
    def learn(self, experiences, gamma):
        
        Q_targets, Q_expected = self.get_target_local_Qs(experiences, gamma)
        
        if self.is_ER_prioritized:
            self.memory.update_TD_errors([q[0] for q in (Q_targets-Q_expected).cpu().detach().numpy()])
        
        #choose between prioritized experienced replay and unifrom one
        #modify update rule
        if self.is_ER_prioritized:
            P = torch.from_numpy(np.vstack(self.memory.chosen_probs)).float().to(device)
            loss = torch.mean( ALPHA*(torch.pow(Q_targets-Q_expected, 2)/torch.pow(P, BETA)) * torch.pow(torch.max(P), BETA) )
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
        
        #print("loss", loss)
        #print("loss_uni", F.mse_loss(Q_expected, Q_targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() #|dive into pytorch documentation| it updates params
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   
            
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def use_prioritized_ER(self, flag=True):
        #when flag==True we use prioritized experience replay instead of uniform one
        self.is_ER_prioritized = flag
        self.memory.is_ER_prioritized = flag
        
    
        
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.TD_errors = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.is_ER_prioritized = False
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def add_TD_error(self, TD_error):
        self.TD_errors.append(TD_error)
        assert len(self.TD_errors) == len(self.memory)

    def update_TD_errors(self, vals):
        if len(vals) == len(self.mem_ind_to_recalc_TDE):
            for i, ind in enumerate(self.mem_ind_to_recalc_TDE):
                self.TD_errors[ind] = vals[i]
        else:
            print("memory.update_TD_error: len(vals) not equal to len(self.mem_ind_to_recalc_TDE)")
            
    def sample(self):
        if self.is_ER_prioritized:
            abs_TD_errors = list(map(abs, self.TD_errors))
            eps_prob = 0.1*max(abs_TD_errors)
            
            Ps = np.power((abs_TD_errors + eps_prob), ALPHA)
            self.prob = Ps/sum(Ps)
            indices = np.arange(len(self.memory))

            chosen_indices = np.random.choice(indices, self.batch_size, replace=False, p=self.prob)
            self.mem_ind_to_recalc_TDE = chosen_indices.copy()
            #chosen_indices = list(map(int, chosen_indices))
            experiences = []
            [experiences.append(self.memory[ind]) for ind in chosen_indices]
            self.chosen_probs = []
            [self.chosen_probs.append(self.prob[ind]) for ind in chosen_indices]

            #experiences = np.asarray(self.memory)[chosen_indices]
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
        
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)