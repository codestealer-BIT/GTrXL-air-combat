import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 设置超参数
state_dim = 4       # 状态维度（根据实际问题调整）
action_dim = 2      # 行动数目
hidden_dim = 64
gamma = 0.99
lr_q = 1e-3
lr_pred = 1e-3
lr_r = 1e-3         # RewardCombiner 的学习率

# Q网络：用于估计每个状态-动作对的价值
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        return self.fc2(x)

# 状态预测网络：输入当前状态和动作（动作以 one-hot 编码），输出对下一状态的特征预测
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, hidden_dim):
        super(StatePredictor, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, state, action):
        x = torch.cat([phi(state), action], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# RewardCombiner：利用 MLP 对外在奖励和内在奖励进行非线性融合
class RewardCombiner(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super(RewardCombiner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, rewards):
        x = self.fc1(rewards)
        x = self.relu(x)
        out = self.fc2(x)
        return out

# 特征提取函数 phi：此处简单地取状态本身，也可以设计为更复杂的网络
def phi(state):
    return state  # 直接用状态

# Epsilon-贪婪策略选择动作
def select_action(q_net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_tensor)
        return q_values.argmax().item()

# 将动作转换为 one-hot 编码
def one_hot(action, action_dim):
    vec = np.zeros(action_dim, dtype=np.float32)
    vec[action] = 1.0
    return vec

# 构造一个简单的环境（示例），实际使用时替换为具体任务环境
class DummyEnv:
    def __init__(self):
        self.state_dim = state_dim
    
    def reset(self):
        return np.random.rand(state_dim)
    
    def step(self, action):
        next_state = np.random.rand(state_dim)
        extrinsic_reward = np.random.choice([0, 1])
        done = False
        return next_state, extrinsic_reward, done, {}

# 初始化网络和优化器
q_net = QNetwork(state_dim, action_dim, hidden_dim)
target_q_net = QNetwork(state_dim, action_dim, hidden_dim)
target_q_net.load_state_dict(q_net.state_dict())
state_predictor = StatePredictor(state_dim, action_dim, state_dim, hidden_dim)
reward_combiner = RewardCombiner(input_dim=2, hidden_dim=32)

optimizer_q = optim.Adam(q_net.parameters(), lr=lr_q)
optimizer_pred = optim.Adam(state_predictor.parameters(), lr=lr_pred)
optimizer_combiner = optim.Adam(reward_combiner.parameters(), lr=lr_r)

env = DummyEnv()
num_episodes = 100
epsilon = 0.1

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        # 选择动作（epsilon-greedy 策略）
        action = select_action(q_net, state, epsilon)
        action_oh = one_hot(action, action_dim)
        # 与环境交互：获得下一个状态与外在奖励
        next_state, r_ex, done, _ = env.step(action)
        
        # 利用状态预测网络计算内在奖励，即预测误差
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action_oh).unsqueeze(0)
        predicted_feature = state_predictor(state_tensor, action_tensor)
        true_feature = phi(torch.FloatTensor(next_state)).unsqueeze(0)
        prediction_error = torch.norm(true_feature - predicted_feature, p=2)
        r_int = prediction_error.item()
        
        # 使用 RewardCombiner 将 r_ex 和 r_int 结合起来得到总奖励
        # 构造输入向量
        reward_input = torch.FloatTensor([r_ex, r_int]).unsqueeze(0)  # shape: (1, 2)
        fusion_reward = reward_combiner(reward_input)
        total_r = fusion_reward.item()
        
        # Q-learning 更新
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_value = q_net(state_tensor)[0, action]
        with torch.no_grad():
            max_next_q = target_q_net(next_state_tensor).max(1)[0]
            target = total_r + gamma * max_next_q.item()
        loss_q = (q_value - target) ** 2
        
        optimizer_q.zero_grad()
        loss_q.backward()
        optimizer_q.step()
        
        # 更新状态预测网络（最小化预测误差）
        loss_pred = 0.5 * torch.norm(true_feature - predicted_feature, p=2) ** 2
        optimizer_pred.zero_grad()
        loss_pred.backward()
        optimizer_pred.step()
        
        # 同时更新 RewardCombiner 的参数，目标可以设计为使最终奖励拟合某个目标
        # 这里仅作为示例，假设目标奖励与 r_ex 接近
        loss_comb = (fusion_reward - r_ex) ** 2
        optimizer_combiner.zero_grad()
        loss_comb.backward()
        optimizer_combiner.step()
        
        state = next_state
        total_reward += total_r
        
    if episode % 10 == 0:
        target_q_net.load_state_dict(q_net.state_dict())
    print("Episode:", episode, "Total Reward:", total_reward)
