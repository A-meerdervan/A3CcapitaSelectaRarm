In this run, when a terminal state is reached and the buffer is not full, the buffer is filled with copies of the terminal state. This is to prevent the discounted reward to be much lower when the buffer is not full. 

Problem: when the buffer is only half full, the terminal state is copied many times, vs when almost full then only a couple copies. This significantly changes the reward received for possibly the same state. 

When this is done, there is still a problem that in a over constrained environment, the area around the walls is seen as very negative value if the arm bumps into the walls there a couple of times. Then it can occur that the bootstrap value for those states becomes very low. This can create a feedback loop where the arm starts to bump into the wall immidiately from the startstate. To overcome this problem, the entropy was increase by a constant value:
self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 10e-4))

This helps to get the agent out of the feedback loop because it will act more randomly. 

IDEA: Add the average reward to the entropy to prevent the agent to converge onto a bad solution

Another solution to this problem could be to not bootstrap at all, or reduce the gamma value, or bootstrap when a terminal state is reached as well, or even increase the buffer size to a high value. 