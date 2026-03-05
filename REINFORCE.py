# 策略梯度算法的蒙特卡洛版本（Monte Carlo Policy Gradient）
import gymnasium as gym # 模型练习环境
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters 超参数配置
learning_rate = 0.0002  # 学习率，随机梯度下降步长的大小
gamma         = 0.98    # 折扣因子，体现算法对未来奖励的重视程度，越接近1就算法就越有远见

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        # 线性层：将4维的状态空间映射到128维的隐藏层
        # 4个状态数据，神经元128个。这4个数据传入神经网络，被发散成128种不同的理解（映射）
        self.fc1 = nn.Linear(4, 128)

        # 输出层：将隐藏层映射到2维的动作空间（左移或右移）
        # 2个输出。这128个神经元被集中到2个输出上（映射），分别用于评估向左/向右移动的价值
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    # 前向传播：这是推理阶段。它将环境观测到的状态向量输入网络
    # input: 状态向量
    # output: 动作概率分布x
    def forward(self, x):
        # 通过ReLU激活函数增加非线性表达能力
        # f(x)=max(0,x)：如果是负数，就变成0；如果是正数，就保持原样。
        x = F.relu(self.fc1(x))

        # 最后通过Softmax函数输出一个动作概率分布x
        # Softmax归一化，将神经网络输出的打分变成概率分布。比如左边20分，右边80分，那么左边的概率为0.2，右边的概率为0.8
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    # 策略更新逻辑
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            # 计算折扣累积回报：当前状态下动作的即时奖励 + 折扣因子 * 下一个状态的即时奖励
            R = r + gamma * R

            # 计算策略损失函数：-log(动作概率) * 折扣累积回报
            # 实现了策略梯度定理，R是该动作带来的最终回报，如果R很高，损失函数会引导网络在未来增加该动作出现的概率
            loss = -torch.log(prob) * R
            # 反向传播：根据奖惩调整神经元权重，找到了调整的方向
            loss.backward()
        # 利用Adam优化算法更新网络参数theta，这一步是修改神经元权重
        self.optimizer.step()
        self.data = []

# 主函数：智能体在主函数中与环境进行采样交互
def main():
    # 初始化标准强化学习环境
    env = gym.make('CartPole-v1')
    pi = Policy()   # 游戏策略网络，需要张量作为输入
    score = 0.0     # 游戏初始分数0.0
    print_interval = 20 # 每20回合显示一次分数
    
    for n_epi in range(10000):
        # 开始一个新回合
        s, _ = env.reset()  # _ 意思是返回的数据不需要用到
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            # 得到动作的概率分布prob
            prob = pi(torch.from_numpy(s).float())  # 执行了一次前向传播，不断执行直到游戏结束

            # 动作采样：根据网络输出的概率分布，选择一个动作。这体现了强化学习中的探索机制。
            m = Categorical(prob) 
            a = m.sample()

            # 执行动作，环境发生状态转移，并返回即时奖励和终止信号
            s_prime, r, terminated, truncated, info = env.step(a.item())
            done = terminated or truncated
            pi.put_data((r,prob[a]))
            s = s_prime # 更新状态
            score += r
        
        # 回合结束，再次执行梯度策略更新
        pi.train_net()
        
        # 20回合显示一次分数
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg_score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()