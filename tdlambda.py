import numpy as np
class TD:
    """
    Discrete value function approximation via temporal difference learning.
    """
    def __init__(self, nstates, alpha, gamma, ld, init_val = 0.0):
        self.V = np.ones(nstates) * init_val
        self.e = np.zeros(nstates)
        self.nstates = nstates
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.ld = ld # lambda
    def value(self, state):
        return self.V[state]
    def delta(self, pstate, reward, state):
        """
        This is the core error calculation. Note that if the value
        function is perfectly accurate then this returns zero since by
        definition value(pstate) = gamma * value(state) + reward.
        """
        return reward + (self.gamma * self.value(state)) - self.value(pstate)
    def train(self, pstate, reward, state):
        """
        A single step of reinforcement learning.
        """
        delta = self.delta(pstate, reward, state)
        self.e[pstate] += 1.0
        #for s in range(self.nstates):
        self.V += self.alpha * delta * self.e
        self.e *= (self.gamma * self.ld)
        return delta
    def learn(self, nepisodes, env, policy, verbose = True):
        # learn for niters episodes with resets
        for i in range(nepisodes):
            self.reset()
            t = env.single_episode(policy) # includes env reset
            for (previous, action, reward, state, next_action) in t:
                self.train(previous, reward, state)
            if verbose:
                print i
    def reset(self):
        self.e = np.zeros(self.nstates)
