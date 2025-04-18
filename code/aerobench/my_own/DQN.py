import torch
import torch.nn.functional as F
import numpy as np


class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,hidden_dim)
        self.fc3=torch.nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=torch.tanh(self.fc1(x))#特征数值偏大，sigmoid使其处于0-1范围
        x=torch.sigmoid(self.fc2(x))
        return self.fc3(x)
    


class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,
                 epsilon):
        self.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.action_dim=action_dim
        self.q_net=Qnet(state_dim,hidden_dim,self.action_dim).to(self.device)
        self.target_q_net=Qnet(state_dim,hidden_dim,self.action_dim).to(self.device)
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma=gamma
        self.epsilon=epsilon
        self.count=0
        self.target_update=10

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state=torch.tensor([state],dtype=torch.float).to(self.device)
            action=self.q_net(state).argmax().item()
        return action
    
    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(
            self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        q_values=self.q_net(states).gather(1,actions)#Q(s,a)
        max_next_q_values=self.target_q_net(next_states).max(1)[0].view(-1,1)#max(1)按行找最大，返回(值，索引)，故取第一个元素
        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)
        dqn_loss=torch.mean(F.mse_loss(q_values,q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count%self.target_update==0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count+=1