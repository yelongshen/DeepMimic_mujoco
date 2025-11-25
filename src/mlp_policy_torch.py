'''
PyTorch version of the MLP policy
Converted from TensorFlow implementation in mlp_policy_trpo.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try gymnasium first, fall back to gym
try:
    import gymnasium as gym
    gym_spaces = gym.spaces
except ImportError:
    import gym
    gym_spaces = gym.spaces


class RunningMeanStd(nn.Module):
    """
    Running mean and standard deviation tracker
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-2, shape=()):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.shape = shape
        
        # Register buffers (non-trainable parameters that are part of state_dict)
        self.register_buffer('_sum', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('_sumsq', torch.full(shape, epsilon, dtype=torch.float64))
        self.register_buffer('_count', torch.tensor(epsilon, dtype=torch.float64))
    
    @property
    def mean(self):
        return (self._sum / self._count).float()
    
    @property
    def std(self):
        variance = (self._sumsq / self._count) - torch.square(self.mean.double())
        return torch.sqrt(torch.maximum(variance, torch.tensor(1e-2))).float()
    
    def update(self, x):
        """Update running statistics with new batch of data"""
        x = torch.as_tensor(x, dtype=torch.float64)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics"""
        delta = batch_mean - self.mean.double()
        tot_count = self._count + batch_count
        
        new_sum = self._sum + batch_mean * batch_count
        new_sumsq = self._sumsq + batch_var * batch_count + torch.square(delta) * self._count * batch_count / tot_count
        
        self._sum.copy_(new_sum)
        self._sumsq.copy_(new_sumsq)
        self._count.copy_(tot_count)


class DiagGaussianPd:
    """Diagonal Gaussian probability distribution"""
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = torch.chunk(flat, 2, dim=-1)
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)
    
    def mode(self):
        """Return the mode (mean for Gaussian)"""
        return self.mean
    
    def sample(self):
        """Sample from the distribution"""
        return self.mean + self.std * torch.randn_like(self.mean)
    
    def neglogp(self, x):
        """Negative log probability"""
        return 0.5 * torch.sum(torch.square((x - self.mean) / self.std), dim=-1) \
               + 0.5 * np.log(2.0 * np.pi) * x.shape[-1] \
               + torch.sum(self.logstd, dim=-1)
    
    def kl(self, other):
        """KL divergence to another Gaussian distribution"""
        return torch.sum(
            other.logstd - self.logstd + 
            (torch.square(self.std) + torch.square(self.mean - other.mean)) / (2.0 * torch.square(other.std)) - 0.5,
            dim=-1
        )
    
    def entropy(self):
        """Entropy of the distribution"""
        return torch.sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), dim=-1)


class DiagGaussianPdType:
    """Diagonal Gaussian probability distribution type"""
    def __init__(self, size):
        self.size = size
    
    def param_shape(self):
        return [2 * self.size]
    
    def sample_shape(self):
        return [self.size]
    
    def pdfromflat(self, flat):
        return DiagGaussianPd(flat)


def make_pdtype(ac_space):
    """Create probability distribution type from action space"""
    if isinstance(ac_space, gym_spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, gym_spaces.Discrete):
        raise NotImplementedError("Discrete action spaces not yet implemented in PyTorch version")
    elif isinstance(ac_space, gym_spaces.MultiDiscrete):
        raise NotImplementedError("MultiDiscrete action spaces not yet implemented in PyTorch version")
    elif isinstance(ac_space, gym_spaces.MultiBinary):
        raise NotImplementedError("MultiBinary action spaces not yet implemented in PyTorch version")
    else:
        raise NotImplementedError(f"Action space type {type(ac_space)} not implemented")


def normc_initializer(std=1.0):
    """
    Initialize weights with normalized columns
    """
    def _initializer(tensor):
        out = torch.randn_like(tensor)
        out *= std / torch.sqrt(torch.sum(torch.square(out), dim=0, keepdim=True))
        return out
    return _initializer


class MlpPolicy(nn.Module):
    """
    Multi-layer perceptron policy network
    
    Args:
        ob_space: Observation space
        ac_space: Action space
        hid_size: Hidden layer size
        num_hid_layers: Number of hidden layers
        gaussian_fixed_var: Whether to use fixed variance for Gaussian policies
    """
    recurrent = False
    
    def __init__(self, ob_space, ac_space, hid_size=64, num_hid_layers=2, gaussian_fixed_var=True):
        super(MlpPolicy, self).__init__()
        
        assert isinstance(ob_space, gym_spaces.Box)
        
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.gaussian_fixed_var = gaussian_fixed_var
        
        # Create probability distribution type
        self.pdtype = make_pdtype(ac_space)
        
        # Observation normalization
        self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        
        # Value function network
        vf_layers = []
        last_size = ob_space.shape[0]
        for i in range(num_hid_layers):
            fc = nn.Linear(last_size, hid_size)
            # Initialize with normalized columns
            with torch.no_grad():
                fc.weight.copy_(normc_initializer(1.0)(fc.weight))
                fc.bias.zero_()
            vf_layers.append(fc)
            vf_layers.append(nn.Tanh())
            last_size = hid_size
        
        # Value function final layer
        self.vf_net = nn.Sequential(*vf_layers)
        self.vf_final = nn.Linear(last_size, 1)
        with torch.no_grad():
            self.vf_final.weight.copy_(normc_initializer(1.0)(self.vf_final.weight))
            self.vf_final.bias.zero_()
        
        # Policy network
        pol_layers = []
        last_size = ob_space.shape[0]
        for i in range(num_hid_layers):
            fc = nn.Linear(last_size, hid_size)
            with torch.no_grad():
                fc.weight.copy_(normc_initializer(1.0)(fc.weight))
                fc.bias.zero_()
            pol_layers.append(fc)
            pol_layers.append(nn.Tanh())
            last_size = hid_size
        
        self.pol_net = nn.Sequential(*pol_layers)
        
        # Policy output layer
        if gaussian_fixed_var and isinstance(ac_space, gym_spaces.Box):
            # Mean output
            self.pol_mean = nn.Linear(last_size, self.pdtype.param_shape()[0] // 2)
            with torch.no_grad():
                self.pol_mean.weight.copy_(normc_initializer(0.01)(self.pol_mean.weight))
                self.pol_mean.bias.zero_()
            
            # Log standard deviation (learned parameter, not depending on input)
            self.pol_logstd = nn.Parameter(
                torch.zeros(1, self.pdtype.param_shape()[0] // 2)
            )
        else:
            self.pol_final = nn.Linear(last_size, self.pdtype.param_shape()[0])
            with torch.no_grad():
                self.pol_final.weight.copy_(normc_initializer(0.01)(self.pol_final.weight))
                self.pol_final.bias.zero_()
    
    def forward(self, ob, stochastic=True):
        """
        Forward pass
        
        Args:
            ob: Observations [batch_size, ob_dim]
            stochastic: Whether to sample stochastically or use mode
            
        Returns:
            actions: Selected actions [batch_size, ac_dim]
            vpred: Value predictions [batch_size]
            pd: Probability distribution object
        """
        # Normalize observations
        ob_normalized = torch.clamp(
            (ob - self.ob_rms.mean) / self.ob_rms.std,
            -5.0, 5.0
        )
        
        # Value function
        vf_out = self.vf_net(ob_normalized)
        vpred = self.vf_final(vf_out).squeeze(-1)
        
        # Policy
        pol_out = self.pol_net(ob_normalized)
        
        if self.gaussian_fixed_var and isinstance(self.ac_space, gym_spaces.Box):
            mean = self.pol_mean(pol_out)
            # Concatenate mean and logstd
            pdparam = torch.cat([mean, mean * 0.0 + self.pol_logstd], dim=1)
        else:
            pdparam = self.pol_final(pol_out)
        
        # Create probability distribution
        pd = self.pdtype.pdfromflat(pdparam)
        
        # Sample action
        if stochastic:
            ac = pd.sample()
        else:
            ac = pd.mode()
        
        return ac, vpred, pd
    
    def act(self, ob, stochastic=True):
        """
        Select action for a single observation
        
        Args:
            ob: Single observation [ob_dim]
            stochastic: Whether to sample stochastically
            
        Returns:
            action: Selected action [ac_dim]
            value: Value prediction (scalar)
        """
        with torch.no_grad():
            ob_tensor = torch.as_tensor(ob, dtype=torch.float32).unsqueeze(0)
            ac, vpred, _ = self.forward(ob_tensor, stochastic)
            return ac[0].cpu().numpy(), vpred[0].cpu().item()
    
    def get_value(self, ob):
        """Get value prediction for observation(s)"""
        with torch.no_grad():
            ob_tensor = torch.as_tensor(ob, dtype=torch.float32)
            if ob_tensor.ndim == 1:
                ob_tensor = ob_tensor.unsqueeze(0)
            
            ob_normalized = torch.clamp(
                (ob_tensor - self.ob_rms.mean) / self.ob_rms.std,
                -5.0, 5.0
            )
            vf_out = self.vf_net(ob_normalized)
            vpred = self.vf_final(vf_out).squeeze(-1)
            
            if vpred.ndim == 0:
                return vpred.cpu().item()
            return vpred.cpu().numpy()
    
    def get_value_train(self, ob_tensor):
        """Get value prediction with gradients enabled (for training)"""
        # Expects ob_tensor to already be a torch tensor on correct device
        if ob_tensor.ndim == 1:
            ob_tensor = ob_tensor.unsqueeze(0)
        
        ob_normalized = torch.clamp(
            (ob_tensor - self.ob_rms.mean) / self.ob_rms.std,
            -5.0, 5.0
        )
        vf_out = self.vf_net(ob_normalized)
        vpred = self.vf_final(vf_out).squeeze(-1)
        return vpred
    
    def get_initial_state(self):
        """Get initial recurrent state (empty for non-recurrent policy)"""
        return []


def test_policy():
    """Test the PyTorch policy implementation"""
    import gymnasium as gym
    
    # Create a simple environment
    env = gym.make('Hopper-v4')
    ob_space = env.observation_space
    ac_space = env.action_space
    
    print(f"Observation space: {ob_space}")
    print(f"Action space: {ac_space}")
    
    # Create policy
    policy = MlpPolicy(ob_space, ac_space, hid_size=64, num_hid_layers=2)
    
    # Test forward pass
    obs = torch.randn(5, ob_space.shape[0])
    actions, values, pd = policy(obs, stochastic=True)
    
    print(f"\nBatch test:")
    print(f"  Input shape: {obs.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Mean: {pd.mean.shape}")
    
    # Test single action
    obs_single = np.random.randn(ob_space.shape[0])
    action, value = policy.act(obs_single, stochastic=True)
    
    print(f"\nSingle test:")
    print(f"  Observation shape: {obs_single.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Value: {value}")
    
    # Test value function
    value = policy.get_value(obs_single)
    print(f"  Value (via get_value): {value}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_policy()
