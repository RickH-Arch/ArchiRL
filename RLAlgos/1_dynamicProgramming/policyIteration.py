#策略迭代是策略评估和策略提升不断循环交替，直至最后得到最优策略的过程。

import copy

class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        #转移矩阵P[state][action] = [(p, next_state, reward, done)]
        self.P = self.createP()

    #动态规划算法要求环境的状态转移概率、奖励值、结束条件都已知（即马尔可夫决策过程已知），而不是通过智能体与环境交互得到
    def createP(self):
        #初始化
        P = [[[]for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。
        # 坐标系原点(0,0) 定义在左上角
        change = [[0,-1],[0,1],[-1,0],[1,0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j>0:
                        P[i* self.ncol + j][a] = [(1, i*self.ncol + j,0,True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

class PolicyIteration:
    """策略迭代算法"""
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol *self.env.nrow
        self.pi = [[0.25,0.25,0.25,0.25]
                   for i in range(self.env.ncol * self.env.nrow)] #初始化为均匀随机策略
        self.theta = theta
        self.gamma = gamma

    #根据策略计算V值（状态价值函数)
    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol *self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1-done))
                        # 为什么是动态规划思想？因为可将上一轮的V值看作已求得的子问题，这一轮的V值看作当前问题
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v 
            if max_diff < self.theta : break
            cnt += 1
        print(f"策略评估进行{cnt}轮后完成")

    #根据V值更新策略
    def policy_improvment(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1-done))
                qsa_list.append( qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list] # 直接贪心得选择动作价值最大的动作
        print("策略提升完成")
        return self.pi
    
    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvment()
            if old_pi == new_pi: break

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])