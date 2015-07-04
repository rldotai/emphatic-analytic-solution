import numpy as np 


class TD:
    """Temporal difference learning."""
    def __init__(self, n, **kwargs):
        """ Initialize the agent. """
        self.n = n
        self.z = np.zeros(n)
        self.theta = np.zeros(n)

    def update(self, **params):
        """ Perform an update for a single step of the algorithm. """
        fvec = params["s"]
        R = params["r"]
        fvec_p = params["sp"]
        alpha = params['alpha']
        gamma = params['gamma']
        lmbda = params['lmbda']
        
        delta = R + gamma*np.dot(self.theta, fvec_p) - np.dot(self.theta, fvec)
        self.z = fvec + (gamma*lmbda*self.z) # accumulating traces
        self.theta += alpha*delta*self.z
        return delta

    def reset(self):
        """Reset traces for the start of episode."""
        self.z[:] = 0

    @classmethod
    def from_weights(cls, weights):
        """Create and initialize an agent from a weight vector"""
        weights = np.array(weights)
        assert(weights.ndim == 1)
        # Initialize the object
        fvec_length = len(weights)
        obj = cls(fvec_length)
        # Set the weights from the weight vector
        obj.theta[:] = weights
        return obj


class EmphaticTD:
    """Temporal difference learning with emphasis."""
    def __init__(self, n, **kwargs):
        """ Initialize the agent. """
        self.n = n
        self.F = 0
        self.M = 0
        self.z = np.zeros(n)
        self.theta = np.zeros(n)

    def update(self, **params):
        """ Perform an update for a single step of the algorithm. """
        fvec = params["s"]
        R = params["r"]
        fvec_p = params["sp"]
        alpha = params['alpha']
        gamma = params['gamma']
        interest = params['interest']
        lmbda = params['lmbda']
        
        delta = R + gamma*np.dot(self.theta, fvec_p) - np.dot(self.theta, fvec)
        
        self.F = interest + gamma * self.F
        self.M = lmbda*interest + (1 - lmbda)*self.F 
        self.z = self.M*fvec + gamma*lmbda*self.z 
        self.theta += alpha*delta*self.z
        return delta

    def reset(self):
        """Reset traces for the start of episode."""
        self.F = 0
        self.M = 0
        self.z[:] = 0


class LSTD():
    def __init__(self, n, epsilon=0):
        self.n  = n                         # number of features
        self.z  = np.zeros(n)               # traces 
        self.A  = np.eye(n,n) * epsilon     # A^-1 . b = theta^*
        self.b  = np.zeros(n) 

    def reset(self):
        """Reset traces for the start of episode."""
        self.z[:] = 0

    @property 
    def theta(self):
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
        return _theta 

    def update(self, **params):
        # Should include rho and gamma_p to be completely correct
        fvec = params["s"]
        R = params["r"]
        fvec_p = params["sp"]
        gamma = params['gamma']
        lmbda = params['lmbda']
        self.z = gamma * lmbda * self.z + fvec 
        self.A += np.outer(self.z, (fvec - gamma*fvec_p))
        self.b += self.z * R


class ELSTD:
    """Emphatic least-squares temporal difference learning. """
    def __init__(self, n, epsilon=0, **kwargs):
        self.n = n
        self.z = np.zeros(n, dtype=np.float)
        self.A = np.eye(n, dtype=np.float) * epsilon
        self.b = np.zeros(n)
        self.F = 0
        self.M = 0

    def reset(self):
        """Reset traces for the start of episode."""
        self.z[:] = 0
        self.F = 0
        self.M = 0

    @property
    def theta(self):
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
        return _theta

    def update(self, params):
        # Should include rho and gamma_p to be completely correct
        fvec = params["s"]
        R = params["r"]
        fvec_p = params["sp"]
        gamma = params['gamma']
        interest = params['interest']
        lmbda = params['lmbda']
        self.F = gamma * self.F + interest
        self.M = (lmbda * interest) + ((1 - lmbda) * self.F)
        self.z = (gamma * lmbda * self.z + self.M * fvec)
        self.A += np.outer(self.z, (fvec - gamma * fvec_p))
        self.b += self.z * reward