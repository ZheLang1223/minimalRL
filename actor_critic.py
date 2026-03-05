# Actor-Critic 演员-评论家算法
# Actor（策略网络）：负责根据当前环境决定动作，使用策略梯度算法
# Critic（价值网络）：负责打分，只要Actor有动作就给出一个状态价值V(s)
# 相比于REINFORCE算法只能等回合结束才拿到最终得分，如果中间一步好棋，可能会被最终结果全盘否定
# Critic可以即时算出动作的价值，让模型学得又快又稳
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10  # 回合内次数

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        # 4个状态，256个神经元，2个输出（动作），1个Critic的价值评价分数
        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        # 现实情况：即时奖励 + 对未来的预测
        td_target = r + gamma * self.v(s_prime) * done
        # 现实与最初预期的差值
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        # 左：Actor的损失；右：Critic的损失
        # detach()：只要数值，不要背后的梯度信息。即不需要做梯度计算
        # F.smooth_l1_loss()：均方误差。误差大时，可以给出一个平稳、线性的惩罚；误差小时，表现得像平方损失，能捕捉到微小的变化。
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
      
def main():  
    env = gym.make('CartPole-v1')
    model = ActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(4000):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):  # 一回合最多10次就停下来反思，这叫“在线学习”，优势是学得快
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s,a,r,s_prime,done))
                
                s = s_prime
                score += r
                
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()