import numpy as np


class RandomPolicy:
    """
    A random policy, i.e., one that chooses an action uniformly at random from 
    those available in the current state.
    """
    def __init__(self, env, random_seed=None):
        """Initialize with an environment, and optionally a random seed."""
        self.env = env 
        self.RandomState = np.random.RandomState(random_seed)

    def __call__(self, state):
        actions = self.env.get_actions(state)
        return self.RandomState.choice(actions)

class Chain:
    """
    An environment consisting of a chain of states.
    Each state has two actions, `left` and `right`, except for the terminal 
    states at the "ends" of the chain. 
    """
    ACTIONS = ('left', 'right')
    def __init__(self, length, start=None, reward=None, **kwargs):
        if start is None:
            start = (length - 1) // 2
        if reward is not None:
            self.reward = reward

        self.length = length
        self.start = start
        self.state = start 

    def reset(self):
        self.state = self.start

    def is_terminal(self, s=None):
        if s is None:
            s = self.state
        return s == self.LEFTMOST or s == self.RIGHTMOST

    @property 
    def LEFTMOST(self):
        return 0

    @property 
    def RIGHTMOST(self):
        return self.length - 1

    @property
    def actions(self):
        return self.ACTIONS

    @property 
    def states(self):
        return list(range(self.LEFTMOST, self.RIGHTMOST + 1))

    @property
    def terminals(self):
        return [s for s in self.states if self.is_terminal(s)]

    @property
    def nonterminals(self):
        return [s for s in self.states if not self.is_terminal(s)]

    @property
    def num_states(self):
        return len(self.nonterminals) + 1
 
    def observe(self, s=None):
        if s is None:
            s = self.state 
        return s

    def get_actions(self, s=None):
        if s is None:
            s = self.state
        return self.actions

    def do(self, action):
        if self.is_terminal():
            next_state = self.state 
        elif action == 'left':
            next_state = self.state - 1
        elif action == 'right':
            next_state = self.state + 1
        else:
            raise Exception("Invalid action:", action)

        ret = self.reward(self.state, action, next_state)
        self.state = next_state
        return ret 

    def reward(self, s, a, sp):
        if sp == self.RIGHTMOST and not self.is_terminal(s) :
            return 1
        else:
            return 0

    @classmethod
    def transition_matrix(ns):
        """
        The transition matrix for a random walk with `n` states (including 
        two terminal states).
        """
        ret = np.zeros((n,n))
        # terminal state transitions
        ret[-2:, -2:] = np.eye(2) 
        # transient states that can terminate
        ret[0,-2] = p       # left side of chain
        ret[0,1] = (1-p)
        ret[-3,-4] = p      # right side of chain
        ret[-3,-1] = (1-p)
        # handle rest of transient states
        for i in range(1, n-3):
            ret[i][i-1] = p 
            ret[i][i+1] = (1-p)
        return ret


class ConveyorBelt:
    """
    An environment consisting of a series of states with a single action that
    leads to the next state until termination is reached.
    Reward is `-1` in all transitions, allowing for the value function to be 
    calculated exactly.

    TODO: Finish implementing slipperiness (and also RandomState)
    """
    ADVANCE = 0
    def __init__(self, n, slip=0.0, start=None, reward=None, **kwargs):
        if start is None:
            start = n
        if reward is not None:
            self.reward = reward

        # Set values
        self.length = n + 1
        self.slip = slip
        self.start = start 
        self.goal = 0 

        # Initialize the environment
        self.reset()

    def is_terminal(self, s=None):
        if s is None:
            s = self.state 
        return s == self.goal

    def reset(self):
        self._state = self.start

    @property 
    def state(self):
        return self._state

    @property 
    def states(self):
        return [i for i in range(self.length)] 

    @property 
    def nonterminals(self):
        return [s for s in self.states if not self.is_terminal(s)]

    @property
    def terminals(self):
        return [s for s in self.states if self.is_terminal(s)]

    @property
    def num_states(self):
        return len(self.nonterminals) + 1

    @property 
    def actions(self):
        return [self.ADVANCE,]

    def observe(self, s=None):
        """The observation associated with a state (default current state)."""
        if s is None:
            s = self.state 
        return s

    def get_actions(self, s=None):
        """The actions available in a state (default current state)."""
        if s is None:
            s = self.state
        return self.actions

    def do(self, action):
        """Take an action to the environment."""
        assert(action in self.get_actions())
        if self.is_terminal():
            nxt = self.goal
        else:
            nxt = self.state - 1
        
        ret = self.reward(self.state, action, nxt)
        self._state = nxt
        return ret

    def reward(self, s, a, sp):
        """The reward function for a given transition."""
        if self.is_terminal():
            return 0
        else:
            return -1

    @classmethod
    def transition_matrix(ns):
        transitions = []
        # non-terminal states
        for i in range(ns-1):
            tmp = np.zeros(ns)
            tmp[i+1] = 1
            transitions.append(tmp)
        # terminal state at end of conveyor belt
        tmp = np.zeros(ns)
        tmp[-1] = 1
        transitions.append(tmp)
        return np.array(transitions)