# Reinforcement Learning 学习笔记

## C1：基本概念

1. 强化学习的目标
   - 寻找最优策略（什么是最优策略？）
2. 重要概念：
   - State：智能体在环境中的状态$$s\in\mathcal{S}$$
   - Action：智能体可以采取的动作$$a\in\mathcal{A}$$（引发状态转移）
   - Policy：$$\pi(a|s)$$是一个概率分布，指示了智能体在给定状态下的行为
   - Reward：智能体执行一个动作所得到的分数$$r\in\mathcal R$$
   - trajectory：智能体与环境的互动序列：$$s_0,a_0,s_1,r_1,a_1...,s_t,r_t,a_t$$
   - return：一个trajectory得到的分数和，$$G=\sum_{i=1}^t{\gamma^{t-1}r_t}$$
3. 强化学习要素与MDP马尔可夫决策过程
   - 三个集合：$$\mathcal S, \mathcal A, \mathcal R(s,a)$$
   - 三个概率分布：
     - 状态转移概率：$$p(s'|s,a)$$
     - 奖励概率：$$p(r|s,a)$$
     - 策略：$$\pi(a|s)$$
   - 马尔可夫性质：
     - $$p(s_{t+1}|a_{t+1},s_t,...,a_1,s_0)=p(s_{t+1}|a_{t+1},s_t)$$
     - $$p(r_{t+1}|a_{t+1},s_t,...,a_1,s_0)=p(r_{t+1}|a_{t+1},s_t)$$

## C2：Bellman Equation

1.  贝尔曼方程的作用：求解state value

2.  state value：$$v_\pi(s)=\mathbb E[G_t|S_t=s]$$，表示在策略 *π* 下，智能体从状态 *s* 开始，未来所有时间步的累积回报的期望值。

3.  action value：$$q_\pi(s,a)=\mathbb E[G_t|S_t=s,A_t=a]$$，表示在策略 *π* 下，智能体从状态 *s* 开始，采取动作a，未来所有时间步的累积回报的期望值。

4.  方程导出：

$$
\begin{align}
G_t &= r_{t+1}+\gamma r_{t+2} + \gamma^2 r_{t+3} + \dots
\\ &= r_{t+1}+\gamma G_{t+1} 
\\
\\ v_\pi(s) &= \mathbb E[G_t|S_t=s] 
\\ &= \mathbb E[r_{t+1}+\gamma G_{t+1}|S_t=s] 
\\ &= \mathbb E[r_{t+1}|S_t=s] + \gamma \mathbb E[G_{t+1}|S_t=s]
\\
\\ \mathbb E[r_{t+1}|S_t=s] &= \sum_a{\pi(a|s) \mathbb E[r_{t+1}|S_t=s,A_t=a]}(期望的全概率公式)
\\ &= \sum_a{\pi(a|s)\sum_r{p(r|s,a)r}}
\\ &= r_\pi(s)
\\
\\ \mathbb E[G_{t+1}|S_t=s] &= \sum_{s'}p(s'|s)\mathbb E[G_{t+1}|S_t=s, S_{t+1}=s']
\\ & = \sum_{s'}p(s'|s)\mathbb E[G_{t+1}|S_{t+1}=s'](马尔可夫性质)
\\ & = \sum_{s'}p(s'|s)v_\pi(s')
\\ & = \sum_{s'}v_\pi(s')\sum_a{p(s'|s,a)\pi(a|s)}
\\ & = \sum_a\pi(a|s)\sum_{s'}v_\pi(s')p(s'|s,a)
\\
\\ Bellman Equation&:
\\ v_\pi(s) &= \sum_a\pi(a|s)[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')]\quad\forall s\in\mathcal S
\\ &= \sum_a\pi(a|s)\times q_\pi(s,a)
\end{align}
$$

3. 矩阵形式

   - 推导：

   $$
   v_\pi(s_i)=r_\pi(s_i)+\gamma\sum_{s_j}p_\pi(s_j|s_i)v_\pi(s_j)
   $$

   $$
   \begin{align}
   \begin{bmatrix}
   v_\pi(s_1)\\
   \vdots\\
   v_\pi(s_n)\\
   \end{bmatrix}
   &=
   \begin{bmatrix}
   r_\pi(s_1)\\
   \vdots\\
   r_\pi(s_n)\\
   \end{bmatrix}
   +
   \gamma
   \begin{bmatrix}
   \sum_{s_j}p_\pi(s_j|s_1)v_\pi(s_j)\\
   \vdots\\
   \sum_{s_j}p_\pi(s_j|s_n)v_\pi(s_j)\\
   \end{bmatrix}
   
   \\& = \begin{bmatrix}
   r_\pi(s_1)\\
   \vdots\\
   r_\pi(s_n)\\
   \end{bmatrix}
   +
   \gamma
   \begin{bmatrix}
   p_\pi(s_1|s_1)& \dots & p_\pi(s_n|s_1) \\
   \vdots & &\vdots\\
   p_\pi(s_1|s_n)& \dots & p_\pi(s_n|s_n) \\
   \end{bmatrix}
   
   \begin{bmatrix}
   v_\pi(s_1)\\
   \vdots\\
   v_\pi(s_n)\\
   \end{bmatrix}
   \\
   \\ \bold v_\pi &=\bold r_\pi+\gamma P_\pi \bold v_\pi
   
   \end{align}
   $$

4. 贝尔曼方程求解方法（需要知道系统状态）

   1. 贝尔曼方程的解析解：

   $$
   \bold v_\pi=(I-\gamma P_\pi)^{-1}\bold r_\pi
   $$
   
   2.  迭代方法：
   $$
   \bold v_{k+1}=r_\pi+\gamma P_\pi \bold v_k
   $$


## C3：Bellman Optimality Equation

- 最优策略$$\pi^*$$的定义：

    $$
    \begin{align}
    v_{\pi^*}(s) \geq v_{\pi}(s) \quad \forall s\in\mathcal S \quad \forall \pi
    \\

    \end{align}
    $$

- 贝尔曼最优公式：

    $$
    \begin{align}
    \\ v_\pi(s) &= \max_\pi(\sum_a\pi(a|s)\times q_\pi(s,a))
    \\ &= \max_\pi(\sum_a\pi(a|s) \times [\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')])
    
    \\ \bold v_\pi &= \max_\pi(\bold r_\pi+\gamma P_\pi \bold v_\pi) 
    \end{align}
    $$

- 求解贝尔曼最优公式

  1. 先求解max部分，实际上只需要确定$$\pi(a|s)$$，根据式子结构可以看出，右边取得最大值时为下式：

  $$
  \begin{align}
  \max_\pi(\sum_a\pi(a|s)\times q_\pi(s,a))&=\max_{a\in\mathcal A(s)}q_\pi(s,a)
  
  \\when \quad\pi(a|s)&= \begin{cases}
  1 & a=a^*\\
  0 & a\neq a^*
  \end{cases}
  
  \\ where \quad a^* &= argmax_{a} \, q(s,a)
  \end{align}
  $$
  
  2. 从另一个角度上看：
     $$
     \bold v_\pi =\max_\pi(\bold r_\pi+\gamma P_\pi \bold v_\pi) =f(\bold v_\pi)
     $$
  
  3. 基于Contraction Mapping Theorem：f是一个压缩映射，那么仅存在一个不动点x\*使得f(x\*)=x\*，且通过迭代方法$$x_{k+1}=f(x_k)$$可以逼近x\*。
  
  4. 迭代算法：$$\bold v_{\pi,k+1} =\max_\pi(\bold r_\pi+\gamma P_\pi \bold v_{\pi,k})$$

## C4：值迭代与策略迭代

- 值迭代：从$$v_0$$开始
  - 策略更新：Greedy Policy
  $$
  \begin{aligned}
  \pi_{k+1}&=argmax_\pi (r_\pi+\gamma P_\pi v_k)
  \\\pi_{k+1}(s)&=argmax_\pi(\sum_a \pi(a|s)q_k(s,a))
  \end{aligned}
  $$
  - 值更新： 

  $$
  v_{k+1}=\max_a q_k(s,a)
  $$

- 策略迭代：从$$\pi_0$$开始

  - 策略评估PE：求解贝尔曼公式（通过迭代算法，执行无穷多步）
    $$
    v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}
    $$
    
  - 策略改进PI：
    $$
    \pi_{k+1}=argmax_\pi (r_\pi+\gamma P_\pi v_{\pi_k})
    $$
  
- 截断策略迭代：

  - 策略评估步骤改为只计算n步

## C5：Monte Carlo方法

- model-free方法

- 基本思想：

  - 考虑进行N次实验，得到了集合数据$$\{g^{(j)}(s,a)|j=1,...,N\}$$。可以用于Action value的估计：
    $$
    q_{\pi_k}(s,a)\approx \frac{1}{N} \sum_{i=1}^N g^{(i)}(s,a)
    $$

- MC-basic方法：

  - 策略评估：通过多次实验获得所有$$q_{\pi_k}(s,a)$$的估计值。 

  - 策略更新：根据$$q_{\pi_k}(s,a)$$更新策略得到$$\pi_{k+1}(s)$$。

  - 改进策略：MC-Exploring Starts

    - 一个trajectory中的数据可以重复使用

    - first-visit与every-visit
    - Generalized Policy Iteration：搜集数据时可以即时改进策略，但要确保从每一个状态开始采取行动都被采样

- MC $$\epsilon$$-Greedy 
  $$
  \pi(a|s)=\begin{cases}
  \begin{aligned}
  & 1-\frac{\epsilon}{|\mathcal A{s}|}(|\mathcal A{s}|-1) &\text{for the greedy action}\\
  &\frac{\epsilon}{|\mathcal A{s}|}  &\text{for the other}\;|\mathcal A(s)|-1 \;\text{action}
  \end{aligned}
  \end{cases}
  $$

## C6：随机近似理论与随机梯度下降

- Robbins-Monro算法：解决$$g(w)=0$$的求根问题

  - 迭代方法：
    $$
    w_{k+1}=w_k-a_k\widetilde g(w_k,\eta_k)
    $$

  - g(w)是一个黑盒，不需要知道具体表达

  - $$\tilde g(w_k,\eta_k)=g(w_k)+\eta_k$$表示含噪声观测值

  - 算法条件：

    - $$0 < c_1\leq\nabla_wg(w) \leq c_2 \; \text{for all }w$$ 严格递增条件且增长不能过快
    - $$\sum_{k=1}^\infin a_k=\infin \;\text{and}\; \sum_{k=1}^\infin a_k^2<\infin$$ 学习率限制
    - $$\mathbb E[\eta_k|\mathcal H_k]=0 \;\text{and}\; \mathbb E[\eta_k^2|\mathcal H_k]<\infin $$ 噪声期望为0，且方差存在

  - $$\mathcal H_k=\{w_k,w_{k-1},...\}$$，那么$$w_k$$依概率收敛到$$w^*$$。

- GD算法

  - 目标：求解$$\nabla_w \mathbb E[f(w, X)]=0$$
  - 应用RM算法：

    - $$w_{k+1}=w_k-a_k \nabla_w \mathbb E[f(w_k, X)]=w_k-a_k \mathbb E[\nabla_wf(w_k, X)]$$

- BGD算法

  - SGD+Monte Carlo：$\mathbb E[\nabla_wf(w_k, X)]\approx\frac{1}{n}\sum_{i=1}^n \nabla_wf(w_k,x_i)$
  - $w_{k+1}=w_k-a_k \frac{1}{n}\sum_{i=1}^n \nabla_wf(w_k,x_i)$

- SGD算法

  - $\mathbb E[\nabla_wf(w_k, X)]\approx\nabla_wf(w_k,x_k)$
  - 相当于Batchsize=1的BGD
  - 不够精确

## C7：时序差分方法 Temporal-Difference

- 方法本质：
  - 提供了一种比MC方法更高效的只需要数据进行state/action value的估计方法

- TD-Learning of State Value：
  - 数据：$$\{(s_t,r_{t+1},s_{t+1})|t=0,1,\dots\}$$
  - $$v_{k+1}(s_t)=v_k(s_t)-\alpha_k(s_t)[v_k(s_t)-[r_{t+1}+\gamma v_k(s_{t+1})]]$$
    - TD error：$$\delta_k=[v_k(s_t)-[r_{t+1}+\gamma v_k(s_{t+1})]]$$
    - TD target：$$\overline v = [r_{t+1}+\gamma v_k(s_{t+1})]$$

  - 实质：RM算法求解Bellman Equation：
    $$
    v_\pi(s)=\mathbb E[R+\gamma v_\pi(S')|S=s] \quad s\in \mathcal S
    $$

  - Online算法+Continuing tasks算法+Bootstrapping

  - TD算法只能计算state value（策略评估），而无法用于改进策略

- TD-Learning of Action Value——SARSA算法

  - 数据：$$\{(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})|t=0,1,\dots\}$$

  - $q_{k+1}(s_t,a_t)=q_k(s_t,a_t)-\alpha_k(s_t,a_t)[q_k(s_t,a_t)-[r_{t+1}+\gamma q_k(s_{t+1},a_{t+1})]]$

  - 实质：
    $$
    q_\pi(s,a)=\mathbb E[R+\gamma q_\pi(S',A')|s,a] \quad \forall s,a
    $$

- Expected Sarsa:

  - $q_{k+1}(s_t,a_t)=q_k(s_t,a_t)-\alpha_k(s_t,a_t)[q_k(s_t,a_t)-[r_{t+1}+\gamma \mathbb E[q_k(s_{t+1},A)]]]$

  - $\mathbb E[q_k(s_{t+1,A})]=v_k(s_{t+1})$

  - 实质：
    $$
    \begin{align}
    q_\pi(s,a)&=\mathbb E[R+\gamma \mathbb E[q_k(s_{t+1},A)]|s] \quad \forall s
    \\
    &=\mathbb E[R+\gamma v_k(s_{t+1})|s] \quad \forall s
    \end{align}
    $$

- n-step Sarsa

  - 数据：$\{(s_t,a_t,r_{t+1},s_{t+1},a_{t+1},...,r_{t+n},s_{t+n},a_{t+n})|t=0,1,\dots\}$

  - 实质：

      $$
      \begin{align}
      q_\pi(s,a)&=\mathbb E[G_t^{(n)}|s,a]
      \\&=\mathbb E[R_{t+1}+\gamma R_{t+2}  + \dots+\gamma^n q_\pi(S_{t+n},A_{t+n})|s,a]
      \end{align}
      $$

- TD-Learning of optimal action values——Q-learning：

    - 数据：$$\{(s_t,a_t,r_{t+1},s_{t+1})|t=0,1,\dots\}$$

    - $q_{k+1}(s_t,a_t)=q_k(s_t,a_t)-\alpha_k(s_t,a_t)[q_k(s_t,a_t)-[r_{t+1}+\gamma \max_{a\in \mathcal A}q_k(s_{t+1},a)]]$

    - 实质：
        $$
        q(s,a)=\mathbb E[R_{t+1}+\gamma \max_aq(S_{t+1},a)|S_t=s,A_t=a]\quad \forall s,a
        $$

    - off-policy与on-policy

        - behavior policy：用于生成数据的策略
        - target policy：不断更新逼近最优的策略
        - on-policy：behavior policy = target policy。反之即为off-policy
        - Q-Learning是一个off-policy方法

- 不同TD算法的本质区别是TD-target不同

## C8：深度强化学习：值函数近似

- 问题背景：状态个数十分巨大，无法用表格形式表示。需要找到值函数直接映射一个状态的state value。$v:s \rightarrow v(s)$

- 解决方法：神经网络方法拟合$v(s;w)$

  - 目标函数：	
    $$
    J(w)=\mathbb E[(v_\pi(S)-\hat v(S,w))^2]
    $$

    - 如何计算目标函数中的期望？
      - 平均分布：$J(w)=\frac{1}{|\mathcal S|}\sum_{s\in \mathcal S}[(v_\pi(s)-\hat v(s,w))^2]$
      - stationary distribution：$J(w)=\sum_{s\in \mathcal S}[d_\pi(s)(v_\pi(s)-\hat v(s,w))^2] \\ d_\pi(s)\text{为Markov Stationary Distribution}$

  - GD：$\nabla_wJ(w)=-2\mathbb E[(v_\pi(S)-\hat v(S,w))\nabla_w \hat v_(S,w)]$

  - SGD：$\nabla_wJ(w)=-2[(v_\pi(s_t)-\hat v(s_t,w))\nabla_w \hat v_(s_t,w)]$

  - 问题：$v_\pi(s_t)$未知

- MC方法+值函数方法：

  - $v_\pi(s_t)$用采样$g_t$代替

- TD方法+值函数方法：

  - $v_\pi(s_t)$用采样$r_{t+1}+\gamma \hat v(s_{t+1},w_t)$代替

- Sarsa+值函数方法：

  - 值函数拟合$q(s_t,a_t;w_t)$
  - $q_\pi(s_t,a_t)$用$r_{t+1}+\gamma \hat q(s_{t+1},a_{t+1};w_t)$代替

- Q-Learning+值函数方法：

  - $q_\pi(s_t,a_t)$用$r_{t+1}+\gamma \max_{a\in\mathcal A(s_{t+1})}\hat q(s_{t+1},a_{t+1};w_t)$代替

- DQN：

  - 目标函数：
    $$
    J(w)=\mathbb E[(R+\gamma \max_{a\in\mathcal A(S')}\hat q(S',a,w)-\hat q(S,A,w))^2]
    $$

  - main network与target network
    $$
    J(w)=\mathbb E[(R+\gamma \max_{a\in\mathcal A(S')}\hat q(S',a,w_{target})-\hat q(S,A,w_{main}))^2]
    $$
  
    - 固定target不动，更新main网络
    - 一段时间后将target网络更新为main
  
  - Experience replay：
  
    - 构建一个样本池，从中均匀采样进行训练

## C9：深度强化学习：策略函数近似

- 神经网络直接拟合$$\pi(a|s;\theta)=softmax(f(s;\theta))$$

- 优势：可以处理连续的A空间

- Metric：

  - average state value：
    $$
    \overline v_\pi=\sum_{s\in \mathcal S} d(s)v_\pi(s)=\mathbb E[v_\pi(S)]
    \\
    \bar v_\pi=\mathbb E[\sum_{t=0}^\infin \gamma^tR_{t+1}]
    $$

  - average one-step reward
    $$
    \overline r_\pi=\sum_{s\in\mathcal S}d_\pi(s)r_\pi(s)=\mathbb E[r_\pi(S)]
    \\
    r_\pi(s)=\sum_{a\in \mathcal A}\pi(a|s)r(s,a)
    \\
    r(s,a)=\mathbb E[R|s,a]=\sum_r rp(r|s,a)
    \\
    \bar r_\pi = \lim_{n \rightarrow \infin}\frac{1}{n}\mathbb E[\sum_{k=1}^nR_{t+k}|S_t=s_0] = \lim_{n \rightarrow \infin}\frac{1}{n}\mathbb E[\sum_{k=1}^nR_{t+k}]
    $$

  - 需要区分$\gamma=1和\gamma \neq 1$

- 梯度：
  $$
  \nabla_\theta J(\theta)=\sum_{s\in \mathcal S}\eta(s)\sum_{a\in \mathcal A}\nabla_\theta \pi(a|s,\theta)q_\pi(s,a)
  \\=\mathbb E[\nabla_\theta \ln \pi(A|S,\theta)q_\pi(S,A)]
  $$
  
- REINFORCE算法：

  - 梯度更新：
    $$
    \begin{align}
    \theta_{k+1}&=\theta_k + \alpha \mathbb E[\nabla_\theta \ln \pi(A|S,\theta_k)q_\pi(S,A)]
    \\&=\theta_k + \alpha \nabla_\theta \ln \pi(a_t|s_t,\theta_k)q_\pi(s_t,a_t)
    \\
    &\text{采用其他方法拟合}q_\pi(s_t,a_t)\text{例如MC方法用}G_t
    \\&=\theta_k + \alpha  \frac{\nabla_\theta\pi(a_t|s_t,\theta_k)}{\pi(a_t|s_t,\theta_k)}q_\pi(s_t,a_t)
    \\&=\theta_k + \alpha  {\nabla_\theta\pi(a_t|s_t,\theta_k)}[\frac{q_\pi(s_t,a_t)}{\pi(a_t|s_t,\theta_k)}]
    \\&=\theta_k + \alpha  {\nabla_\theta\pi(a_t|s_t,\theta_k)}\beta_t
    \end{align}
    $$
  
  - $\beta_t$是一个平衡因子，可以平衡探索与数据利用

## C10：Actor-Critic方法

- Actor：用于执行策略的模型

- Critic：用于策略评估的模型

- 本质：采用其他方法估计REINFORCE算法中的$q_\pi(s,a)$

- QAC：

  - Value update：SARSA方法学习q
  - Policy update：PG方法学习策略

- Adavantaged actor-critic——A2C：
  $$
  \nabla_\theta J(\theta)=\mathbb E[\nabla_\theta \ln \pi(A|S,\theta)q_\pi(S,A)]\\=\mathbb E[\nabla_\theta \ln \pi(A|S,\theta)[q_\pi(S,A)-b(S)]]
  $$

  - 添加一个$b(S)$项作为baseline降低方差，期望不变。方差最小时：
    $$
    b^*(s)=\frac{\mathbb E_{A\sim \pi}[||\nabla_\theta\ln\pi(A|s,\theta_t)||^2q(s,A)]}{\mathbb E_{A\sim \pi}[||\nabla_\theta\ln\pi(A|s,\theta_t)||^2]}
    \\
    \approx\mathbb E_{A\sim\pi}[q(s,A)]=v_\pi(s)
    $$

  - 更新梯度时：
    $$
    \begin{align}
    \theta_{k+1}&=\theta_k + \alpha \mathbb E[\nabla_\theta \ln \pi(A|S,\theta_k)[q_\pi(S,A)-v_\pi(S)]]\\
    \\&\rightarrow\theta_k+\alpha \nabla_\theta \pi(a_t|s_t,\theta_k)\frac{\delta_k(s_t,a_t)}{\pi(a_t|s_t,\theta_k)}
    \\
    &q_\pi(S,A)-v_\pi(S)\text{称为优势函数}\delta_\pi(S,A)
    \end{align}
    $$

  - 如何计算？
    $$
    \delta_k \rightarrow r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)
    $$

    - 利用一个神经网络估计$v(s)$即可得出$\delta$

- Off-policy actor-critic

  - Importance Sampling：

    - 如何使用${x_i}\sim p1$估计$\mathbb E_{X\sim p_0}[X]$？
      $$
      \mathbb E_{X\sim p_0}[X]=\sum_x x p_0(x)=\sum_x x p_1(x)\frac{p_0(x)}{p_1(x)}
      \\=\mathbb E_{X\sim p_1}[X\frac{p_0(X)}{p_1(X)}]
      \\ \approx \frac{1}{n}\sum_{i=1}^{n} \frac{p_0(x_i)}{p_1(x_i)}x_i
      $$

    - 应用到AC方法中：$\beta$为behevior policy
      $$
      \begin{align}
      \nabla_\theta J(\theta)&=\mathbb E_{S\sim \rho,A\sim \beta}[\frac{\pi(A|S;\theta)}{\beta(A|S)} \nabla_\theta \ln \pi(A|S,\theta)\delta_\pi(S,A)]
      \\ &\text{此时梯度计算更改为：}
      \\
      \theta_{k+1}&=
      \theta_k+\alpha \nabla_\theta \pi(a_t|s_t,\theta_k)\frac{\delta_k(s_t,a_t)}{\pi(a_t|s_t,\theta_k)}\times \frac{\pi(a_t|s_t;\theta_k)}{\beta(a_t|s_t)}
      \\&=\theta_k+\alpha \nabla_\theta \pi(a_t|s_t,\theta_k)\frac{\delta_k(s_t,a_t)}{\beta(a_t|s_t)}
      \end{align}
      $$
      

- Deterministic actor-critic——DPG

  - 如何处理无限的行为空间$\mathcal A$？不进行行为分类，而直接输出$a=\mu(s;\theta)$

  - 本质是分类任务转为回归任务

  - 该算法是off-policy

  - 损失函数：
    $$
    J(\theta)=\mathbb E_{S\sim d_0}[v_\mu(S)]
    $$

    - 如何选择$d_0$：
      - 分布与策略无关
      - 特殊情况1：选择起始状态$d_0(s_0)=1$
      - 特殊情况2：选择行为策略$\beta$的稳态分布

  - 梯度计算：
    $$
    \nabla_\theta J(\theta)=\mathbb E_{S\sim \rho_\mu }[\nabla_\theta\mu(S)(\nabla_aq_\mu(S,a))|_{a=\mu(S)}]
    $$

## C???：PPO DPO与其他

