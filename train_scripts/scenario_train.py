"""
Agents trained with this script are put into a Scenario with 4 reflesx agents that first clean for 200 steps,
then eat for 200 steps. We train 3 focal agents, whose goal is to exploit the cleaners' behaviour and eat as many apples as possible.
"""

import os
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.amp import GradScaler
from meltingpot import scenario as scenario_lib
import dm_env
from tqdm import tqdm
from collections import deque


SCENARIO_NAME = "clean_up_2"
N_FOCAL = 3
ACT_DIM = 9
OBS_H, OBS_W, OBS_C = 88, 88, 3

EPISODES = 3000
STEPS_EP = 700
TRAIN_EPOCHS = 6
MINIBATCH = 512

GAMMA = 0.99
LAM = 0.97
CLIP_RATIO = 0.2
VF_COEF = 0.5
ENT_COEF_START = 0.7
ENT_COEF_END = 0.3
ANNEAL_EPISODES = 3000
GRAD_CLIP = 0.5
LR = 1e-3
FC_HIDDEN = 256

SAVE_EVERY = 100
SAVE_PATH = "../checkpoints/scenario2_impl/"


ROI = (slice(48, 72), slice(40, 64))
DIRTY = np.array([48.0, 140.6, 141.6], dtype=np.float32)
CLEAN = np.array([35.0, 141.4, 130.0], dtype=np.float32)

QUEUE_LEN = 50
CLEAN_LOOKBACK = 20
SPIN_LOOKBACK = 5
SPIN_PENALTY = -2
NO_CLEAN_PENALTY = -0.5
ZAPPING_PENALTY = -2
NOOP_PENALTY = -1
APPLE_REWARD = 25
CLEAN_REWARD = 2
GOOD_CLEAN_REWARD = 15
NEAR_RIVER_REWARD = 0.05

NOOP = 0
FORWARD = 1
BACKWARD = 2
LEFT = 3
RIGHT = 4
TURN_LEFT = 5
TURN_RIGHT = 6
ZAP = 7
CLEAN_ACT  = 8

COMPARE_WINDOW = 50


class ActionQueue:
    def __init__(self, n_agents: int, maxlen: int, device):
        self.n_agents = n_agents
        self.maxlen   = maxlen
        self.device   = device
        self.buf      = torch.full((n_agents, maxlen), fill_value=-1,
                                   dtype=torch.int64, device=device)
        self.ptr      = 0

    def reset(self):
        self.buf.fill_(-1)
        self.ptr = 0

    def push(self, actions: torch.Tensor):
        self.buf[:, self.ptr] = actions
        self.ptr = (self.ptr + 1) % self.maxlen

    def last_k(self, k: int) -> torch.Tensor:
        k = min(k, self.maxlen)
        rolled = torch.roll(self.buf, -self.ptr, dims=1)
        return rolled[:, self.maxlen - k:]


class AgentNetwork(nn.Module):
    def __init__(self, n_agents: int, act_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim  = act_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(OBS_C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.cnn_out_dim = self.cnn(
                torch.zeros(1, OBS_C, OBS_H, OBS_W)
            ).shape[1]

        raw_in = self.cnn_out_dim + n_agents * act_dim

        self.policy_fc = nn.Sequential(
            nn.Linear(raw_in,    FC_HIDDEN), nn.ReLU(),
            nn.Linear(FC_HIDDEN, FC_HIDDEN), nn.ReLU(),
        )
        self.actor = nn.Linear(FC_HIDDEN, act_dim)
        self.critic = nn.Linear(FC_HIDDEN, 1)

    def encode(self, obs_chw: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs_chw)

    def forward_policy(self, cnn_feat, last_oh):
        raw = torch.cat([cnn_feat, last_oh], dim=-1)
        x = self.policy_fc(raw)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def prep_obs(rgb_hwc: np.ndarray, device) -> torch.Tensor:
    return (
        torch.tensor(rgb_hwc, dtype=torch.float32, device=device)
        .div_(255.0)
        .permute(2, 0, 1)
        .contiguous()
    )


def actions_to_onehot(actions: torch.Tensor, act_dim: int) -> torch.Tensor:
    B, N = actions.shape
    return F.one_hot(actions, num_classes=act_dim).float().view(B, N * act_dim)


def compute_gae(rewards, values, dones, v_last, gamma=GAMMA, lam=LAM):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = v_last if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    return adv, adv + values


def compute_action_queue_penalties(queue: ActionQueue, n_agents: int, device) -> torch.Tensor:
    penalties = torch.zeros(n_agents, device=device)

    recent_spin = queue.last_k(SPIN_LOOKBACK)
    is_turn = (recent_spin == TURN_LEFT) | (recent_spin == TURN_RIGHT)
    has_left = (recent_spin == TURN_LEFT).any(dim=1)
    has_right = (recent_spin == TURN_RIGHT).any(dim=1)
    valid = (recent_spin != -1).all(dim=1)
    spinning = valid & is_turn.all(dim=1) & (has_left | has_right)
    penalties[spinning] += SPIN_PENALTY

    recent_clean = queue.last_k(CLEAN_LOOKBACK)
    valid_clean = (recent_clean != -1).all(dim=1)
    cleaned = (recent_clean == CLEAN_ACT).any(dim=1)
    penalties[valid_clean & ~cleaned] += NO_CLEAN_PENALTY

    return penalties


def is_water_roi(rgb):
    x = rgb.astype(np.float32)
    water_mask = (x[..., 1] + x[..., 2]) > 1.5 * x[..., 0]
    patch = water_mask[ROI]
    if float(patch.mean()) > 2/3:
        return 1
    if float(water_mask.mean()) > 0.5:
        return 0
    return -1


def clean_fraction_roi(rgb):
    patch  = rgb[ROI].astype(np.float32)
    d_dirty = np.linalg.norm(patch - DIRTY, axis=2)
    d_clean = np.linalg.norm(patch - CLEAN, axis=2)
    return float((d_clean < d_dirty).mean())


def clean_success(obs_rgb, next_obs_rgb, delta=0.10):
    return (clean_fraction_roi(next_obs_rgb) - clean_fraction_roi(obs_rgb)) > delta


def clean_shaping(rewards_arr, actions_arr, obs_list, next_obs_list):
    for i in range(N_FOCAL):
        act = int(actions_arr[i])
        roi_score = is_water_roi(obs_list[i]["RGB"])
        if act == CLEAN_ACT and roi_score == 1:
            rewards_arr[i] += CLEAN_REWARD
            if clean_success(obs_list[i]["RGB"], next_obs_list[i]["RGB"]):
                rewards_arr[i] += GOOD_CLEAN_REWARD
        if roi_score >= 0 and act != NOOP:
            rewards_arr[i] += NEAR_RIVER_REWARD
        if roi_score >= 0 and act in (FORWARD, LEFT, RIGHT):
            rewards_arr[i] += 0.2
    return rewards_arr


def anti_degenerate_shaping(rewards_arr, actions_arr):
    for i in range(N_FOCAL):
        if int(actions_arr[i]) == ZAP:
            rewards_arr[i] += ZAPPING_PENALTY
        elif int(actions_arr[i]) == NOOP:
            rewards_arr[i] += NOOP_PENALTY
    return rewards_arr


def get_timestep_obs_list(timestep, n_focal):
    return [dict(timestep.observation[i]) for i in range(n_focal)]


def get_timestep_rewards(timestep, n_focal):
    arr = np.zeros(n_focal, dtype=np.float32)
    for i in range(n_focal):
        r = timestep.reward[i] if timestep.reward is not None else 0.0
        arr[i] = float(r) if r is not None else 0.0
    return arr


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(SAVE_PATH, exist_ok=True)

    env = scenario_lib.build(SCENARIO_NAME)

    timestep = env.reset()
    actual_focal = len(timestep.observation)
    print(f"Scenario '{SCENARIO_NAME}': {actual_focal} focal agents")
    assert actual_focal == N_FOCAL, (
        f"Expected {N_FOCAL} focal agents, got {actual_focal}. "
        f"Check SCENARIO_NAME or update N_FOCAL."
    )

    actual_act_dim = env.action_spec()[0].num_values
    assert actual_act_dim == ACT_DIM, (
        f"ACT_DIM mismatch: config={ACT_DIM}, env={actual_act_dim}. Update ACT_DIM."
    )
    act_dim = actual_act_dim
    print(f"Action dim: {act_dim}")

    nets = [AgentNetwork(N_FOCAL, act_dim).to(device) for _ in range(N_FOCAL)]

    optimizers = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]
    scalers = [GradScaler("cuda") for _ in range(N_FOCAL)]

    obs_buf = np.zeros((STEPS_EP, N_FOCAL, OBS_H, OBS_W, OBS_C), dtype=np.uint8)
    act_buf = np.zeros((STEPS_EP, N_FOCAL), dtype=np.int64)
    logp_buf = np.zeros((STEPS_EP, N_FOCAL), dtype=np.float32)
    val_buf = np.zeros((STEPS_EP, N_FOCAL), dtype=np.float32)
    rew_buf = np.zeros((STEPS_EP, N_FOCAL), dtype=np.float32)
    done_buf = np.zeros((STEPS_EP, N_FOCAL), dtype=np.float32)
    lastact_buf = np.zeros((STEPS_EP, N_FOCAL, N_FOCAL), dtype=np.int64)

    queue = ActionQueue(N_FOCAL, QUEUE_LEN, device)

    focal_return_history    = deque(maxlen=COMPARE_WINDOW)
    baseline_ep_returns     = deque(maxlen=COMPARE_WINDOW)

    total_env_steps = 0

    for ep in tqdm(range(EPISODES)):
        timestep = env.reset()
        queue.reset()

        last_actions = torch.zeros(N_FOCAL, dtype=torch.int64, device=device)
        ep_ret = np.zeros(N_FOCAL, dtype=np.float32)
        shaped_rew = np.zeros(N_FOCAL, dtype=np.float32)
        t_end = STEPS_EP

        obs_list = get_timestep_obs_list(timestep, N_FOCAL)

        for t in range(STEPS_EP):
            obs_tensors = torch.stack([
                prep_obs(obs_list[i]["RGB"], device) for i in range(N_FOCAL)
            ])

            with torch.no_grad(), torch.amp.autocast("cuda"):
                cnn_feats = torch.stack([
                    nets[i].encode(obs_tensors[i:i+1]).squeeze(0)
                    for i in range(N_FOCAL)
                ])

            last_oh = actions_to_onehot(
                last_actions.unsqueeze(0).expand(N_FOCAL, -1), act_dim
            )

            all_logits, all_values = [], []
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for i in range(N_FOCAL):
                    logits_i, val_i = nets[i].forward_policy(
                        cnn_feats[i:i+1], last_oh[i:i+1]
                    )
                    all_logits.append(logits_i.squeeze(0))
                    all_values.append(val_i.squeeze(0))

            all_logits_t = torch.stack(all_logits)
            all_values_t = torch.stack(all_values)

            dist    = Categorical(logits=all_logits_t)
            acts_t  = dist.sample()
            logps_t = dist.log_prob(acts_t)

            # Step environment
            acts_np  = acts_t.cpu().numpy()
            actions  = [int(acts_np[i]) for i in range(N_FOCAL)]
            timestep = env.step(actions)

            env_rewards  = get_timestep_rewards(timestep, N_FOCAL)
            next_obs_list = get_timestep_obs_list(timestep, N_FOCAL)

            # Apple reward boost
            for i in range(N_FOCAL):
                if env_rewards[i] >= 1.0:
                    env_rewards[i] = APPLE_REWARD

            shaped = env_rewards.copy()
            shaped = clean_shaping(shaped, acts_np, obs_list, next_obs_list)
            shaped = anti_degenerate_shaping(shaped, acts_np)

            queue.push(acts_t)
            queue_penalties = compute_action_queue_penalties(queue, N_FOCAL, device)
            shaped += queue_penalties.cpu().numpy()

            is_last = (timestep.step_type == dm_env.StepType.LAST)

            # Rollout
            for i in range(N_FOCAL):
                obs_buf[t, i] = obs_list[i]["RGB"]
                act_buf[t, i] = acts_np[i]
                logp_buf[t, i] = float(logps_t[i].item())
                val_buf[t, i] = float(all_values_t[i].item())
                rew_buf[t, i] = float(shaped[i])
                done_buf[t, i] = float(is_last)
                lastact_buf[t, i] = acts_np
                ep_ret[i] += float(env_rewards[i])
                shaped_rew[i] += float(shaped[i])

            last_actions = acts_t.detach()
            obs_list = next_obs_list
            total_env_steps += N_FOCAL

            if is_last:
                t_end = t + 1
                break

        v_last = np.zeros(N_FOCAL, dtype=np.float32)
        obs_last = torch.stack([prep_obs(obs_list[i]["RGB"], device) for i in range(N_FOCAL)])
        last_oh_f = actions_to_onehot(last_actions.unsqueeze(0).expand(N_FOCAL, -1), act_dim)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in range(N_FOCAL):
                feat_i = nets[i].encode(obs_last[i:i+1])
                _, val_i = nets[i].forward_policy(feat_i, last_oh_f[i:i+1])
                v_last[i] = float(val_i.item())

        # GAE
        adv_buf = np.zeros((t_end, N_FOCAL), dtype=np.float32)
        ret_buf = np.zeros((t_end, N_FOCAL), dtype=np.float32)
        for i in range(N_FOCAL):
            adv_buf[:, i], ret_buf[:, i] = compute_gae(
                rew_buf[:t_end, i], val_buf[:t_end, i], done_buf[:t_end, i], v_last[i]
            )

        adv_flat = adv_buf.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        adv_buf  = adv_flat.reshape(t_end, N_FOCAL)

        ent_coef = ENT_COEF_START - (ENT_COEF_START - ENT_COEF_END) * (ep / ANNEAL_EPISODES)

        # PPO update
        for i in range(N_FOCAL):
            net_i = nets[i]
            opt_i = optimizers[i]
            scaler_i = scalers[i]

            obs_i = obs_buf[:t_end, i]
            act_i = act_buf[:t_end, i]
            logp_old_i = logp_buf[:t_end, i]
            adv_i = adv_buf[:t_end, i]
            ret_i = ret_buf[:t_end, i]
            lastact_i = lastact_buf[:t_end, i]

            idxs = np.arange(t_end)
            net_i.train()

            for _ in range(TRAIN_EPOCHS):
                np.random.shuffle(idxs)
                for start in range(0, t_end, MINIBATCH):
                    mb = idxs[start : start + MINIBATCH]

                    obs_mb = torch.stack([prep_obs(obs_i[s], device) for s in mb])
                    act_mb = torch.as_tensor(act_i[mb], device=device, dtype=torch.int64)
                    logp_old_mb = torch.as_tensor(logp_old_i[mb], device=device, dtype=torch.float32)
                    adv_mb = torch.as_tensor(adv_i[mb], device=device, dtype=torch.float32)
                    ret_mb = torch.as_tensor(ret_i[mb], device=device, dtype=torch.float32)
                    lastact_mb = torch.as_tensor(lastact_i[mb], device=device, dtype=torch.int64)

                    last_oh_mb = actions_to_onehot(lastact_mb, act_dim)

                    with torch.amp.autocast("cuda"):
                        feat_mb = net_i.encode(obs_mb)
                        logits_mb, val_mb = net_i.forward_policy(feat_mb, last_oh_mb)

                        dist_mb = Categorical(logits=logits_mb)
                        logp_mb = dist_mb.log_prob(act_mb)
                        entropy = dist_mb.entropy().mean()

                        ratio = torch.exp(logp_mb - logp_old_mb)
                        surr1 = ratio * adv_mb
                        surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * adv_mb
                        pi_loss = -torch.min(surr1, surr2).mean()
                        v_loss = F.mse_loss(val_mb, ret_mb)
                        loss = pi_loss + VF_COEF * v_loss - ent_coef * entropy

                    opt_i.zero_grad()
                    scaler_i.scale(loss).backward()
                    scaler_i.unscale_(opt_i)
                    nn.utils.clip_grad_norm_(net_i.parameters(), GRAD_CLIP)
                    scaler_i.step(opt_i)
                    scaler_i.update()

        focal_mean = ep_ret.mean()
        focal_return_history.append(focal_mean)

        if ep < COMPARE_WINDOW:
            baseline_ep_returns.append(focal_mean)

        rolling_mean = np.mean(focal_return_history)
        baseline_mean = np.mean(baseline_ep_returns) if baseline_ep_returns else float('nan')
        vs_baseline = rolling_mean - baseline_mean

        print(
            f"ep {ep:04d}  "
            f"steps {t_end:4d}  "
            f"focal_env_ret {ep_ret.mean():.2f}  "
            f"focal_shaped {shaped_rew.mean():.2f}  "
            f"rolling{COMPARE_WINDOW} {rolling_mean:.2f}  "
            f"vs_baseline {vs_baseline:+.2f}  "
            f"ent_coef {ent_coef:.3f}  "
            f"total_steps {total_env_steps:,}"
        )

        if (ep + 1) % SAVE_EVERY == 0:
            for i, net in enumerate(nets):
                torch.save(
                    net.state_dict(),
                    os.path.join(SAVE_PATH, f"focal_agent_{i}_ep{ep+1}.pt"),
                )
            print(f"Saved checkpoints at episode {ep + 1}")

    env.close()
    for i, net in enumerate(nets):
        torch.save(net.state_dict(), os.path.join(SAVE_PATH, f"focal_agent_{i}_final.pt"))
    print("Training complete.")


if __name__ == "__main__":
    train()
