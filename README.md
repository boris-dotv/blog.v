# The-Cecilia

我始终以最真诚, 严肃的态度, 怀着敬畏之心把*The Cecilia*当作一件艺术品来打磨.



## Reinforcement Learning

强化学习的几个组成部分:  
* **Agent**, agents learn to take actions to maximize expected reward.
* **Action**, change the environment.
* **Environment**, the place where agents are expected to maximize expected reward.
* **Reward**, a feedback from pre-defined rule or reward model.
* **Observation**, agents 对当前环境 state 的观察, 注意 state 不总是等于 observation, 比如在象棋游戏中, 当前棋盘的 state 就是 agents 的 observation, 在扑克牌中, 当前棋盘的 state 就是打出的牌和所有 agents 手里的牌, 但 observation 是 agents 手里的牌. state 是"上帝视角", observation 是 agent 视角.

### Policy-based Approach (Learning an Actor)
Machine learning 的终极目标可以大致描述为 Looking for a function. 在 RL 中, Observation, Actor/Policy, Action 的关系可以表示为:  

$$Action = \pi(Observation)$$

我们定义 
$$
\bar{R}_{\theta}
$$
作为 
$$
R_{\theta}
$$
的期望值.
