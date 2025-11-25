"""
PyTorch implementation of Trust Region Policy Optimization (TRPO)
Converted from TensorFlow implementation in trpo.py
"""

import os
# Set environment variable BEFORE importing torch to prevent CUDA initialization
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA to avoid driver issues

import gym
import time
import argparse
import numpy as np

# Import torch - CPU only version required!
# If you get CUDA crashes, reinstall PyTorch with:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager
from collections import deque
from tqdm import tqdm

# Import PyTorch policy
from mlp_policy_torch import MlpPolicy

# Import utilities
from utils.misc_util import set_global_seeds, boolean_flag
from utils.console_util import colorize
from config import Config
from play_mocap import PlayMocap

# Try to import MPI, but make it optional
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI = None
    MPI_AVAILABLE = False
    print("Warning: MPI not available. Running in single-process mode.")


def conjugate_gradient(Ax_func, b, cg_iters=10, residual_tol=1e-10):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    for i in range(cg_iters):
        Ap = Ax_func(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        
        if new_rdotr < residual_tol:
            break
            
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
        
    return x


def flat_grad(output, parameters, retain_graph=False, create_graph=False):
    """Get flattened gradients"""
    if create_graph:
        retain_graph = True
    
    grads = torch.autograd.grad(output, parameters, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([grad.reshape(-1) for grad in grads])


def set_flat_params(model, flat_params, param_list=None):
    """Set model parameters from flattened vector"""
    if param_list is None:
        param_list = list(model.parameters())
    
    offset = 0
    for param in param_list:
        param_shape = param.shape
        param_size = param.numel()
        param.data.copy_(flat_params[offset:offset + param_size].reshape(param_shape))
        offset += param_size


def get_flat_params(model, param_list=None):
    """Get flattened model parameters"""
    if param_list is None:
        param_list = list(model.parameters())
    
    return torch.cat([param.data.reshape(-1) for param in param_list])


class TRPOAgent:
    """Trust Region Policy Optimization Agent"""
    
    def __init__(self, env, policy, max_kl=0.01, damping=0.1, 
                 gamma=0.995, lam=0.97, vf_iters=3, vf_lr=1e-3,
                 cg_iters=10, device='cpu'):
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        
        # TRPO hyperparameters
        self.max_kl = max_kl
        self.damping = damping
        self.gamma = gamma
        self.lam = lam
        
        # Value function training
        self.vf_iters = vf_iters
        self.vf_optimizer = torch.optim.Adam(
            [p for name, p in self.policy.named_parameters() if 'vf' in name],
            lr=vf_lr
        )
        
        self.cg_iters = cg_iters
        
    def select_action(self, obs, stochastic=True):
        """Select action from policy"""
        return self.policy.act(obs, stochastic=stochastic)
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        advantages = []
        gae = 0
        
        next_value = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        
        return advantages, returns
    
    def update_policy(self, obs, actions, advantages, old_log_probs):
        """Update policy using TRPO"""
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = advantages.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # DEBUG: Check advantages before normalization
        adv_mean_before = advantages.mean().item()
        adv_std_before = advantages.std().item()
        adv_min = advantages.min().item()
        adv_max = advantages.max().item()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # IMPORTANT: Get OLD policy outputs and store them (before any updates)
        # This is what we'll compare against after the policy update
        with torch.no_grad():
            _, _, old_pd = self.policy(obs, stochastic=True)
            old_log_probs_actual = -old_pd.neglogp(actions)
        
        # Compute surrogate loss with OLD policy (before update)
        _, _, pd = self.policy(obs, stochastic=True)
        log_probs = -pd.neglogp(actions)
        ratio = torch.exp(log_probs - old_log_probs_actual)  # Should be 1.0 initially
        surr_loss = -(ratio * advantages).mean()
        
        # DEBUG: Check if ratio is reasonable
        ratio_mean = ratio.mean().item()
        ratio_std = ratio.std().item()
        
        # Compute policy gradient
        policy_params = [p for name, p in self.policy.named_parameters() 
                        if 'pol' in name or 'logstd' in name]
        
        # DEBUG: Check if we found any policy parameters
        num_policy_params = len(policy_params)
        total_param_count = sum(p.numel() for p in policy_params)
        print(f"DEBUG: Found {num_policy_params} policy parameter tensors, {total_param_count} total params")
        if num_policy_params == 0:
            print("WARNING: No policy parameters found!")
            print("Available parameters:")
            for name, p in self.policy.named_parameters():
                print(f"  {name}: {p.shape}")
        else:
            # Print first few parameter values to check if they're zeros
            first_param = policy_params[0]
            print(f"DEBUG: First policy param shape: {first_param.shape}, mean: {first_param.mean().item():.6f}, std: {first_param.std().item():.6f}")
        
        grads = flat_grad(surr_loss, policy_params, retain_graph=True)
        
        # DEBUG: Check gradient magnitude
        grad_norm = torch.norm(grads).item()
        
        # Compute Fisher vector product
        def Fvp(v):
            kl = self._compute_kl(obs).mean()  # Must be scalar for grad computation
            grads = flat_grad(kl, policy_params, create_graph=True)
            flat_grad_kl = grads
            
            kl_v = (flat_grad_kl * v).sum()
            grads = flat_grad(kl_v, policy_params, retain_graph=True)
            
            return grads + v * self.damping
        
        # Compute step direction using conjugate gradient
        stepdir = conjugate_gradient(Fvp, -grads, cg_iters=self.cg_iters)
        
        # DEBUG: Check step direction
        stepdir_norm = torch.norm(stepdir).item()
        
        # Compute step size
        shs = 0.5 * torch.dot(stepdir, Fvp(stepdir))
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm
        
        # DEBUG: Check step size
        fullstep_norm = torch.norm(fullstep).item()
        
        # Line search - only update policy parameters, not value function
        # DEBUG: Check parameter values before get_flat_params
        print(f"DEBUG: About to call get_flat_params with {len(policy_params)} tensors")
        for i, p in enumerate(policy_params[:3]):  # Just first 3
            print(f"  Tensor {i}: shape={p.shape}, norm={torch.norm(p).item():.6f}")
        
        old_params = get_flat_params(self.policy, policy_params)
        print(f"DEBUG: get_flat_params returned tensor with shape {old_params.shape}, norm={torch.norm(old_params).item():.6f}")
        
        expected_improve = -torch.dot(grads, fullstep)
        
        success, new_params = self._line_search(
            old_params, fullstep, expected_improve, 
            obs, actions, advantages, old_log_probs, surr_loss,
            policy_params
        )
        
        set_flat_params(self.policy, new_params, policy_params)
        
        # Compute KL and entropy for logging
        with torch.no_grad():
            kl = self._compute_kl(obs).mean().item()
            _, _, pd = self.policy(obs, stochastic=True)
            entropy = pd.entropy().mean().item()
        
        return {
            'policy_loss': surr_loss.item(),
            'kl': kl,
            'entropy': entropy,
            'line_search_success': success,
            # DEBUG info
            'debug': {
                'adv_mean_before': adv_mean_before,
                'adv_std_before': adv_std_before,
                'adv_min': adv_min,
                'adv_max': adv_max,
                'ratio_mean': ratio_mean,
                'ratio_std': ratio_std,
                'grad_norm': grad_norm,
                'stepdir_norm': stepdir_norm,
                'fullstep_norm': fullstep_norm,
                'expected_improve': expected_improve.item()
                # param_change, old_params_norm, actual_param_change not yet implemented
                # final_ratio_mean, final_ratio_std not yet implemented
            }
        }
    
    def _compute_kl(self, obs):
        """Compute KL divergence between old and new policy"""
        with torch.no_grad():
            _, _, old_pd = self.policy(obs, stochastic=True)
            old_mean = old_pd.mean.clone()
            old_std = old_pd.std.clone()
        
        _, _, new_pd = self.policy(obs, stochastic=True)
        
        # KL divergence for diagonal Gaussian
        kl = torch.sum(
            torch.log(new_pd.std / old_std) + 
            (old_std.pow(2) + (old_mean - new_pd.mean).pow(2)) / (2.0 * new_pd.std.pow(2)) - 0.5,
            dim=-1
        )
        
        return kl
    
    def _line_search(self, old_params, fullstep, expected_improve, 
                     obs, actions, advantages, old_log_probs, old_loss,
                     policy_params, max_backtracks=10, accept_ratio=0.1):
        """Backtracking line search"""
        for step_frac in [0.5**x for x in range(max_backtracks)]:
            new_params = old_params + step_frac * fullstep
            set_flat_params(self.policy, new_params, policy_params)
            
            with torch.no_grad():
                _, _, pd = self.policy(obs, stochastic=True)
                log_probs = -pd.neglogp(actions)
                ratio = torch.exp(log_probs - old_log_probs)
                surr_loss = -(ratio * advantages).mean()
                
                improve = old_loss - surr_loss
                kl = self._compute_kl(obs).mean()
                
                if kl <= self.max_kl * 1.5 and improve > 0:
                    return True, new_params
        
        # Line search failed, return old parameters
        return False, old_params
    
    def update_value_function(self, obs, returns):
        """Update value function"""
        obs = torch.FloatTensor(obs).to(self.device)
        returns = returns.to(self.device)
        
        dataset = TensorDataset(obs, returns)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        total_loss = 0
        num_batches = 0
        
        for _ in range(self.vf_iters):
            for batch_obs, batch_returns in loader:
                self.vf_optimizer.zero_grad()
                
                # Use get_value_train which keeps gradients enabled
                values = self.policy.get_value_train(batch_obs)
                
                vf_loss = ((values - batch_returns) ** 2).mean()
                vf_loss.backward()
                self.vf_optimizer.step()
                
                total_loss += vf_loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


def collect_trajectories(policy, env, timesteps_per_batch, stochastic=True):
    """Collect trajectories from environment"""
    obs_list = []
    actions_list = []
    rewards_list = []
    values_list = []
    dones_list = []
    log_probs_list = []
    
    ep_rets = []
    ep_lens = []
    
    ob = env.reset()
    ob = env.reset_model_init()
    
    cur_ep_ret = 0
    cur_ep_len = 0
    t = 0
    
    while t < timesteps_per_batch:
        ac, vpred = policy.act(ob, stochastic=stochastic)
        
        # Get log probability
        with torch.no_grad():
            ob_tensor = torch.FloatTensor(ob).unsqueeze(0)
            _, _, pd = policy(ob_tensor, stochastic=stochastic)
            log_prob = -pd.neglogp(torch.FloatTensor(ac).unsqueeze(0)).item()
        
        obs_list.append(ob)
        actions_list.append(ac)
        values_list.append(vpred)
        log_probs_list.append(log_prob)
        
        ob, rew, done, _ = env.step(ac)
        
        rewards_list.append(rew)
        dones_list.append(done)
        
        cur_ep_ret += rew
        cur_ep_len += 1
        t += 1
        
        if done:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            ob = env.reset_model_init()
    
    return {
        'observations': np.array(obs_list),
        'actions': np.array(actions_list),
        'rewards': rewards_list,
        'values': values_list,
        'dones': dones_list,
        'log_probs': torch.tensor(log_probs_list),
        'ep_rets': ep_rets,
        'ep_lens': ep_lens
    }


def train(env, policy, args, task_name):
    """Train policy using TRPO"""
    # Force CPU usage to avoid CUDA driver issues
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # create trpo_agent.
    agent = TRPOAgent(
        env, policy,
        max_kl=args.max_kl,
        gamma=0.995,
        lam=0.97,
        vf_iters=3,
        vf_lr=1e-3,
        device=device
    )
    
    # Load pretrained weights if provided
    if args.pretrained_weight_path:
        print(f"Loading pretrained weights from {args.pretrained_weight_path}")
        agent.policy.load_state_dict(torch.load(args.pretrained_weight_path))
    
    # Load SFT pre-trained model if provided (recommended!)
    if args.load_sft_pretrain:
        print(f"\n{'='*60}")
        print(f"Loading SFT pre-trained policy from {args.load_sft_pretrain}")
        agent.policy.load_state_dict(torch.load(args.load_sft_pretrain, map_location=device))
        print(f"✓ SFT pre-training loaded successfully!")
        print(f"Starting RL fine-tuning from pre-trained policy...")
        print(f"{'='*60}\n")
    
    # Training loop
    timesteps_so_far = 0
    iters_so_far = 0
    
    lenbuffer = deque(maxlen=40)
    rewbuffer = deque(maxlen=40)
    
    print(f"\n{'='*60}")
    print(f"Starting TRPO Training")
    print(f"Task: {task_name}")
    print(f"Max timesteps: {args.num_timesteps}")
    print(f"{'='*60}\n")
    
    while timesteps_so_far < args.num_timesteps:
        # Collect trajectories
        traj = collect_trajectories(
            agent.policy, env, 
            timesteps_per_batch=2048,
            stochastic=True
        )
        
        # Compute advantages
        advantages, returns = agent.compute_advantages(
            traj['rewards'], 
            traj['values'], 
            traj['dones']
        )
        
        # DEBUG: Check if value function is overfitting
        values_tensor = torch.tensor(traj['values'], dtype=torch.float32)
        value_error = torch.abs(returns - values_tensor).mean().item()
        
        # Update policy
        policy_stats = agent.update_policy(
            traj['observations'],
            traj['actions'],
            advantages,
            traj['log_probs']
        )
        
        # Store value error for logging
        policy_stats['value_error'] = value_error
        
        # Update value function
        vf_loss = agent.update_value_function(
            traj['observations'],
            returns
        )
        
        # Update observation normalization
        agent.policy.ob_rms.update(traj['observations'])
        
        # Logging
        timesteps_so_far += len(traj['rewards'])
        iters_so_far += 1
        
        lenbuffer.extend(traj['ep_lens'])
        rewbuffer.extend(traj['ep_rets'])
        
        if iters_so_far % 10 == 0:
            print(f"Iteration {iters_so_far} | Timesteps {timesteps_so_far}")
            print(f"  Avg episode length: {np.mean(lenbuffer):.2f}")
            print(f"  Avg episode reward: {np.mean(rewbuffer):.4f}")
            print(f"  Min/Max reward: {np.min(traj['rewards']):.4f} / {np.max(traj['rewards']):.4f}")
            print(f"  Values vs Returns - error: {policy_stats.get('value_error', 0):.6f}")
            print(f"  Advantages - mean: {advantages.mean().item():.6f}, std: {advantages.std().item():.6f}")
            print(f"  Returns - mean: {returns.mean().item():.6f}, std: {returns.std().item():.6f}")
            print(f"  Policy loss: {policy_stats['policy_loss']:.4f}")
            print(f"  KL divergence: {policy_stats['kl']:.6f}")
            print(f"  Entropy: {policy_stats['entropy']:.4f}")
            print(f"  Value loss: {vf_loss:.4f}")
            print(f"  Line search: {'✓ Success' if policy_stats['line_search_success'] else '✗ Failed'}")
            
            # Debug information
            debug = policy_stats.get('debug', {})
            if debug:
                print(f"\n  [DEBUG]")
                print(f"    Advantages (raw): mean={debug.get('adv_mean_before', 0):.6f}, std={debug.get('adv_std_before', 0):.6f}")
                print(f"    Advantages range: [{debug.get('adv_min', 0):.4f}, {debug.get('adv_max', 0):.4f}]")
                print(f"    Ratio (BEFORE update): mean={debug.get('ratio_mean', 0):.6f}, std={debug.get('ratio_std', 0):.6f}")
                print(f"    Ratio (AFTER update): mean={debug.get('final_ratio_mean', 0):.6f}, std={debug.get('final_ratio_std', 0):.6f}")
                print(f"    Gradient norm: {debug.get('grad_norm', 0):.6f}")
                print(f"    Step direction norm: {debug.get('stepdir_norm', 0):.6f}")
                print(f"    Full step norm: {debug.get('fullstep_norm', 0):.6f}")
                print(f"    Expected improvement: {debug.get('expected_improve', 0):.6f}")
                print(f"    Param change (planned): {debug.get('param_change', 0):.6f}")
                print(f"    Param change (actual): {debug.get('actual_param_change', 0):.6f}")
                print(f"    Old params norm: {debug.get('old_params_norm', 0):.6f}")
            print()
        
        # Save checkpoint
        if iters_so_far % args.save_per_iter == 0:
            save_path = os.path.join(args.checkpoint_dir, task_name, f'iter_{iters_so_far}.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(agent.policy.state_dict(), save_path)
            print(f"💾 Saved checkpoint to {save_path}")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}\n")


def evaluate(env, policy, args):
    """Evaluate trained policy"""
    # Force CPU usage to avoid CUDA driver issues
    device = torch.device('cpu')
    policy = policy.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.load_model_path}")
    policy.load_state_dict(torch.load(args.load_model_path, map_location=device))
    policy.eval()
    
    # Setup video recording
    from VideoSaver import VideoSaver
    from datetime import datetime
    
    render_dir = './render'
    os.makedirs(render_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = os.path.join(render_dir, f'eval_{Config.motion}_{timestamp}.avi')
    video_saver = VideoSaver(video_path, fps=30)
    print(f"Recording video to: {video_path}")
    
    # Evaluation loop
    ep_rets = []
    ep_lens = []
    
    print(f"Evaluating {args.num_eval_episodes} episodes...")
    
    for ep in tqdm(range(args.num_eval_episodes), desc="Evaluating"):
        ob = env.reset()
        ob = env.reset_model_init()
        
        ep_ret = 0
        ep_len = 0
        done = False
        
        while not done and ep_len < args.max_episode_steps:
            with torch.no_grad():
                ac, vpred = policy.act(ob, stochastic=args.stochastic_policy)
            
            ob, rew, done, _ = env.step(ac)
            
            # Render and save frame
            frame = env.render(mode='rgb_array')
            if frame is not None:
                video_saver.add_frame(frame)
            
            ep_ret += rew
            ep_len += 1
        
        ep_rets.append(ep_ret)
        ep_lens.append(ep_len)
        
        if ep < 3:
            print(f"  Episode {ep+1}: length={ep_len}, return={ep_ret:.4f}")
    
    # Save video
    video_saver.save()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Average episode length: {np.mean(ep_lens):.2f} ± {np.std(ep_lens):.2f}")
    print(f"  Average return:         {np.mean(ep_rets):.4f} ± {np.std(ep_rets):.4f}")
    print(f"  Min return:             {np.min(ep_rets):.4f}")
    print(f"  Max return:             {np.max(ep_rets):.4f}")
    print(f"  Video saved to: {video_path}")
    print(f"{'='*60}\n")
    
    # Optionally save trajectories
    if args.save_sample:
        save_path = f"eval_{Config.motion}_{timestamp}.npz"
        np.savez(save_path, 
                 lengths=ep_lens,
                 returns=ep_rets)
        print(f"Saved evaluation data to {save_path}")


def main(args):
    """Main entry point"""
    set_global_seeds(args.seed)
    
    # Create environment
    from dp_env_v3 import DPEnv
    env = DPEnv()
    env.seed(args.seed)
    
    # If loading SFT pretrain, auto-detect hidden size from checkpoint
    actual_hidden_size = args.policy_hidden_size
    if args.load_sft_pretrain:
        print(f"\nDetecting hidden size from SFT checkpoint: {args.load_sft_pretrain}")
        checkpoint = torch.load(args.load_sft_pretrain, map_location='cpu')
        # Check the shape of the first hidden layer
        if 'pol_net.0.weight' in checkpoint:
            detected_size = checkpoint['pol_net.0.weight'].shape[0]
            print(f"✓ Detected hidden_size={detected_size} from checkpoint")
            actual_hidden_size = detected_size
        else:
            print(f"⚠ Warning: Could not detect hidden size from checkpoint, using default={actual_hidden_size}")
    
    # Create policy
    policy = MlpPolicy(
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hid_size=actual_hidden_size,
        num_hid_layers=2,
        gaussian_fixed_var=True
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Task name
    task_name = f"trpo-{Config.motion}-{args.seed}"
    
    if args.task == 'train':
        train(env, policy, args, task_name)
    elif args.task == 'evaluate':
        evaluate(env, policy, args)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    env.close()


def argsparser():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("PyTorch Implementation of TRPO for DeepMimic")
    
    # Environment
    parser.add_argument('--env_id', help='environment ID', default='DeepMimic')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    
    # Paths
    parser.add_argument('--checkpoint_dir', help='directory to save models', 
                       default='checkpoint_torch')
    parser.add_argument('--load_model_path', help='path to load model', 
                       type=str, default=None)
    parser.add_argument('--pretrained_weight_path', help='path to pretrained weights',
                       type=str, default=None)
    parser.add_argument('--load_sft_pretrain', help='path to SFT pre-trained model',
                       type=str, default=None)
    
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], 
                       default='train')
    
    # Training
    parser.add_argument('--num_timesteps', help='number of timesteps', 
                       type=int, default=5000000)
    parser.add_argument('--save_per_iter', help='save model every N iterations',
                       type=int, default=100)
    parser.add_argument('--g_step', help='number of policy update steps per iteration',
                       type=int, default=3)
    
    # Policy network
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    
    # TRPO hyperparameters
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficient', 
                       type=float, default=0)
    
    # Evaluation
    boolean_flag(parser, 'stochastic_policy', default=False,
                help='use stochastic policy for evaluation')
    boolean_flag(parser, 'save_sample', default=False,
                help='save evaluation trajectories')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                       help='number of episodes for evaluation')
    parser.add_argument('--max_episode_steps', type=int, default=2048,
                       help='maximum steps per episode')
    
    # Device
    boolean_flag(parser, 'cpu', default=False, help='force CPU usage')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = argsparser()
    main(args)
