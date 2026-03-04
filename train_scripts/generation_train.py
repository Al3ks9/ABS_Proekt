"""
Agents are trained for 10 generations for 1000 episodes each, 700 steps per episode. Every generation,
a fitness tracker determines which agents were the most successful, picks the two best agents,
copies those nets to the other agents, and the copied nets have Gaussian noise so they don't converge to the same policy.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.amp import GradScaler
from shimmy import MeltingPotCompatibilityV0
from tqdm import tqdm


# Hyper-parameters

SUBSTRATE = "clean_up"
N_AGENTS = 7
ACT_DIM = 9
OBS_H, OBS_W, OBS_C = 88, 88, 3

GENERATIONS = 10
EPISODES = 1000
STEPS_EP = 700
TRAIN_EPOCHS = 6
MINIBATCH = 1024

GAMMA = 0.99
LAM = 0.97
CLIP_RATIO = 0.2
VF_COEF = 0.5
GRAD_CLIP = 0.5
LR_POLICY = 1e-3

ENT_COEF_START = 0.7
ENT_COEF_END = 0.3
ANNEAL_EPISODES = 10000

FC_HIDDEN = 256

SAVE_EVERY = 500
SAVE_PATH = "../checkpoints/generation_impl/"

ROI = (slice(48, 72), slice(40, 64))

DIRTY = np.array([48.0, 140.6, 141.6], dtype=np.float32)
CLEAN = np.array([35.0, 141.4, 130.0], dtype=np.float32)

QUEUE_LEN = 50
CLEAN_LOOKBACK = 20
WALKING_LOOKBACK = 30
SPIN_LOOKBACK = 5
SPIN_PENALTY = -2
NO_CLEAN_PENALTY = -0.5
WALKING_PENALTY = 0
ZAPPING_PENALTY = -2
NOOP_PENALTY = -1
APPLE_REWARD = 25
CLEAN_REWARD = 2
GOOD_CLEAN_REWARD = 15
NEAR_RIVER_REWARD = .05

NOOP = 0
FORWARD = 1
BACKWARD = 2
LEFT = 3
RIGHT = 4
TURN_LEFT  = 5
TURN_RIGHT = 6
ZAP = 7
CLEAN_ACT  = 8

NOISE_SMALL = 0.01
NOISE_LARGE = 0.05


class FitnessTracker:
    W_APPLES = 3.0
    W_GOOD_CLEANS = 2.0
    W_CLEAN_ATTEMPTS = 0.5
    W_RIVER_STEPS = 0.3
    W_ZAPS = -2.0
    W_NOOPS = -0.5

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.apples = np.zeros(self.n_agents, dtype=np.float32)
        self.good_cleans = np.zeros(self.n_agents, dtype=np.float32)
        self.clean_attempts = np.zeros(self.n_agents, dtype=np.float32)
        self.river_steps = np.zeros(self.n_agents, dtype=np.float32)
        self.zap_count = np.zeros(self.n_agents, dtype=np.float32)
        self.noop_count = np.zeros(self.n_agents, dtype=np.float32)
        self.total_steps = np.zeros(self.n_agents, dtype=np.float32)

    def update(
        self,
        agent_idx: int,
        action: int,
        reward: float,         # raw env reward BEFORE shaping
        roi_score: int,        # output of is_water_roi: -1, 0, or 1
        obs_rgb: np.ndarray,   # current obs, used for clean_success check
        next_obs_rgb: np.ndarray,
    ):
        self.total_steps[agent_idx] += 1

        # Apple eaten — env gives exactly 1.0 for eating an apple
        if reward >= 1.0:
            self.apples[agent_idx] += 1

        # Clean attempt — agent tried to clean while at the river
        if action == CLEAN_ACT and roi_score == 1:
            self.clean_attempts[agent_idx] += 1
            # Successful clean — river actually got cleaner
            if clean_success(obs_rgb, next_obs_rgb):
                self.good_cleans[agent_idx] += 1

        # River proximity — agent is near or at the river
        if roi_score >= 0:
            self.river_steps[agent_idx] += 1

        # Zapping
        if action == ZAP:
            self.zap_count[agent_idx] += 1

        # Nooping
        if action == NOOP:
            self.noop_count[agent_idx] += 1

    def compute_fitness(self) -> np.ndarray:
        steps = np.maximum(self.total_steps, 1)

        apple_rate = self.apples / steps
        good_clean_rate = self.good_cleans / steps
        clean_attempt_rate = self.clean_attempts / steps
        river_rate = self.river_steps / steps
        zap_rate = self.zap_count / steps
        noop_rate = self.noop_count / steps

        fitness = (
            self.W_APPLES * apple_rate
          + self.W_GOOD_CLEANS * good_clean_rate
          + self.W_CLEAN_ATTEMPTS * clean_attempt_rate
          + self.W_RIVER_STEPS * river_rate
          + self.W_ZAPS * zap_rate
          + self.W_NOOPS * noop_rate
        )

        return fitness.astype(np.float32)

    def summary(self, gen: int) -> str:
        fitness = self.compute_fitness()
        ranked = np.argsort(fitness)[::-1]

        lines = [
            f"\n{'='*65}",
            f"  Generation {gen} fitness summary",
            f"{'='*65}",
            f"  {'Agent':>6}  {'Fitness':>8}  {'Apples':>7}  {'GoodCln':>8}  "
            f"{'ClnAtt':>7}  {'RiverSt':>8}  {'Zaps':>6}  {'Noops':>6}",
            f"  {'-'*57}",
        ]
        for i in ranked:
            lines.append(
                f"{i:>6} {fitness[i]:>8.4f}  "
                f"{int(self.apples[i]):>7}  "
                f"{int(self.good_cleans[i]):>8}  "
                f"{int(self.clean_attempts[i]):>7}  "
                f"{int(self.river_steps[i]):>8}  "
                f"{int(self.zap_count[i]):>6}  "
                f"{int(self.noop_count[i]):>6}"
            )
        lines.append(f"{'='*65}\n")
        return "\n".join(lines)



class ActionQueue:
    def __init__(self, n_agents: int, maxlen: int, device):
        self.n_agents = n_agents
        self.maxlen = maxlen
        self.device = device
        self.buf = torch.full((n_agents, maxlen), fill_value=-1, 
                              dtype=torch.int64, device=device)
        self.ptr = 0

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
        self.act_dim = act_dim

        # CNN encoder
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
            nn.Linear(raw_in, FC_HIDDEN), nn.ReLU(),
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


# Utilities

def anti_degenerate_shaping(actions, shaped_rewards):
    for a, act in actions.items():
        if act == ZAP:
            shaped_rewards[a] += ZAPPING_PENALTY
        elif act == NOOP:
            shaped_rewards[a] += NOOP_PENALTY
    return shaped_rewards


def clean_fraction_roi(rgb):
    patch = rgb[ROI].astype(np.float32)
    d_dirty = np.linalg.norm(patch - DIRTY, axis=2)
    d_clean = np.linalg.norm(patch - CLEAN, axis=2)
    return float((d_clean < d_dirty).mean())


def clean_success(obs_rgb, next_obs_rgb, delta=0.10):
    before = clean_fraction_roi(obs_rgb)
    after = clean_fraction_roi(next_obs_rgb)
    return (after - before) > delta


def clean_shaping(rewards, actions, obs, next_obs, rois):
    for a, act in actions.items():
        roi_score = rois[a]
        if act == 8 and roi_score == 1:
            rewards[a] += CLEAN_REWARD
            if clean_success(obs[a]["RGB"], next_obs[a]["RGB"]):
                rewards[a] += GOOD_CLEAN_REWARD
        if roi_score >= 0 and act != NOOP:
            rewards[a] += NEAR_RIVER_REWARD
        if roi_score >= 0 and act in (FORWARD, LEFT, RIGHT):
            rewards[a] += 0.2
    return rewards


def is_water_roi(rgb):
    x = rgb.astype(np.float32)
    water_mask = (x[..., 1] + x[..., 2]) > 1.5 * x[..., 0]
    patch = water_mask[ROI]
    if float(patch.mean()) > 2/3:
        return 1
    if float(water_mask.mean()) > .5:
        return 0
    return -1


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

    # Spin detection
    recent_spin = queue.last_k(SPIN_LOOKBACK)
    is_turn = (recent_spin == TURN_LEFT) | (recent_spin == TURN_RIGHT)
    has_left_turn = (recent_spin == TURN_LEFT).any(dim=1)
    has_right_turn = (recent_spin == TURN_RIGHT).any(dim=1)
    valid = (recent_spin != -1).all(dim=1)
    spinning = valid & is_turn.all(dim=1) & (has_left_turn | has_right_turn)
    penalties[spinning] += SPIN_PENALTY

    # No-clean detection
    recent_clean = queue.last_k(CLEAN_LOOKBACK)
    valid_clean = (recent_clean != -1).all(dim=1)
    cleaned = (recent_clean == CLEAN_ACT).any(dim=1)
    penalties[valid_clean & ~cleaned] += NO_CLEAN_PENALTY

    return penalties


def evolve_population(nets, optimizers, ranked):
    elite_1 = ranked[0]
    elite_2 = ranked[1]

    # Agents ranked 3rd and 4th: copy from best + small noise
    for idx in ranked[2:4]:
        nets[idx].load_state_dict(nets[elite_1].state_dict())
        for param in nets[idx].parameters():
            param.data += NOISE_SMALL * torch.randn_like(param.data)
        optimizers[idx] = torch.optim.Adam(nets[idx].parameters(), lr=LR_POLICY)

    # Agents ranked 5th, 6th, 7th: copy from best + large noise
    for idx in ranked[4:]:
        nets[idx].load_state_dict(nets[elite_1].state_dict())
        for param in nets[idx].parameters():
            param.data += NOISE_LARGE * torch.randn_like(param.data)
        optimizers[idx] = torch.optim.Adam(nets[idx].parameters(), lr=LR_POLICY)

    print(f"Elites kept: agent_{elite_1}, agent_{elite_2}")
    print(f"Small noise copies: agent_{ranked[2]}, agent_{ranked[3]}")
    print(f"Large noise copies: agent_{ranked[4]}, agent_{ranked[5]}, agent_{ranked[6]}")


def save_generation(nets, gen, fitness_scores, save_path=SAVE_PATH):
    gen_dir = os.path.join(save_path, f"gen_{gen:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    for i, net in enumerate(nets):
        torch.save(
            net.state_dict(),
            os.path.join(gen_dir, f"agent_{i}_fitness{fitness_scores[i]:.4f}.pt")
        )
    print(f" Saved generation {gen} checkpoints → {gen_dir}")



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(SAVE_PATH, exist_ok=True)

    nets = [AgentNetwork(N_AGENTS, ACT_DIM).to(device) for _ in range(N_AGENTS)]

    optimizers = [
        torch.optim.Adam(net.parameters(), lr=LR_POLICY)
        for net in nets
    ]
    scalers = [GradScaler("cuda") for _ in range(N_AGENTS)]

    env = MeltingPotCompatibilityV0(
        substrate_name=SUBSTRATE,
        render_mode=None,
        max_cycles=STEPS_EP,
    )

    total_env_steps = 0

    obs_buf = np.zeros((STEPS_EP, N_AGENTS, OBS_H, OBS_W, OBS_C), dtype=np.uint8)
    act_buf = np.zeros((STEPS_EP, N_AGENTS), dtype=np.int64)
    logp_buf = np.zeros((STEPS_EP, N_AGENTS), dtype=np.float32)
    val_buf = np.zeros((STEPS_EP, N_AGENTS), dtype=np.float32)
    rew_buf = np.zeros((STEPS_EP, N_AGENTS), dtype=np.float32)
    done_buf = np.zeros((STEPS_EP, N_AGENTS), dtype=np.float32)
    lastact_buf = np.zeros((STEPS_EP, N_AGENTS, N_AGENTS), dtype=np.int64)
    queue = ActionQueue(N_AGENTS, QUEUE_LEN, device)

    fitness_tracker = FitnessTracker(7)

    for gen in range(GENERATIONS):
        scalers = [GradScaler("cuda") for _ in range(N_AGENTS)]
        fitness_tracker.reset()
        for ep in tqdm(range(EPISODES)):
            obs_dict, _ = env.reset()
            queue.reset()
            agents = list(env.agents)

            last_actions = torch.zeros(N_AGENTS, dtype=torch.int64, device=device)
            ep_ret = np.zeros(N_AGENTS, dtype=np.float32)
            shaped_rewards = np.zeros(N_AGENTS, dtype=np.float32)
            t_end = STEPS_EP

            global_ep = gen * EPISODES + ep
            ent_coef = ENT_COEF_START - (ENT_COEF_START - ENT_COEF_END) * min(1.0, global_ep / ANNEAL_EPISODES)

            for t in range(STEPS_EP):
                obs_tensors = torch.stack([
                    prep_obs(obs_dict[agents[i]]["RGB"], device)
                    for i in range(N_AGENTS)
                ])

                with torch.no_grad(), torch.amp.autocast("cuda"):
                    cnn_feats = torch.stack([
                        nets[i].encode(obs_tensors[i:i+1]).squeeze(0)
                        for i in range(N_AGENTS)
                    ])

                last_oh = actions_to_onehot(
                    last_actions.unsqueeze(0).expand(N_AGENTS, -1), ACT_DIM
                )

                # Policy forward for all agents
                all_logits, all_values = [], []
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    for i in range(N_AGENTS):
                        logits_i, val_i = nets[i].forward_policy(
                            cnn_feats[i:i+1], last_oh[i:i+1]
                        )
                        all_logits.append(logits_i.squeeze(0))
                        all_values.append(val_i.squeeze(0))

                all_logits_t = torch.stack(all_logits)
                all_values_t = torch.stack(all_values)

                dist = Categorical(logits=all_logits_t)
                acts_t = dist.sample()
                logps_t = dist.log_prob(acts_t)

                # Step environment
                acts_np = acts_t.cpu().numpy()
                actions_dict = {agents[i]: int(acts_np[i]) for i in range(N_AGENTS)}
                next_obs, raw_rewards, terminations, truncations, _ = env.step(actions_dict)

                queue.push(acts_t)
                queue_penalties = compute_action_queue_penalties(queue, N_AGENTS, device)

                rewards = {a: (APPLE_REWARD if (raw_rewards[a] >= 1) else raw_rewards[a]) for a in agents}

                rois = {a: is_water_roi(obs_dict[a]["RGB"]) for a in agents}

                rewards = clean_shaping(rewards, actions_dict, obs_dict, next_obs, rois)
                rewards = anti_degenerate_shaping(actions_dict, rewards)

                # Rollout
                for i in range(N_AGENTS):
                    a = agents[i]

                    fitness_tracker.update(
                        agent_idx=i,
                        action=acts_np[i],
                        reward=float(raw_rewards[a]),
                        roi_score=rois[a],
                        obs_rgb=obs_dict[a]["RGB"],
                        next_obs_rgb=next_obs[a]["RGB"]
                    )

                    obs_buf[t, i] = obs_dict[a]["RGB"]
                    act_buf[t, i] = acts_np[i]
                    logp_buf[t, i] = float(logps_t[i].item())
                    val_buf[t, i] = float(all_values_t[i].item())
                    rew_buf[t, i] = float(rewards[a]) + float(queue_penalties[i].item())
                    done_buf[t, i] = float(terminations[a] or truncations[a])
                    lastact_buf[t, i] = acts_np
                    ep_ret[i] += float(rewards[a])
                    shaped_rewards[i] += float(rewards[a]) + float(queue_penalties[i].item())

                last_actions = acts_t.detach()
                obs_dict = next_obs
                total_env_steps += N_AGENTS

                if all((terminations[a] or truncations[a]) for a in agents):
                    t_end = t + 1
                    break


            v_last = np.zeros(N_AGENTS, dtype=np.float32)
            obs_last = torch.stack([
                prep_obs(obs_dict[agents[i]]["RGB"], device) for i in range(N_AGENTS)
            ])
            last_oh_final = actions_to_onehot(
                last_actions.unsqueeze(0).expand(N_AGENTS, -1), ACT_DIM
            )
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for i in range(N_AGENTS):
                    feat_i = nets[i].encode(obs_last[i:i+1])
                    _, val_i = nets[i].forward_policy(feat_i, last_oh_final[i:i+1])
                    v_last[i] = float(val_i.item())

            # GAE per agent
            adv_buf = np.zeros((t_end, N_AGENTS), dtype=np.float32)
            ret_buf = np.zeros((t_end, N_AGENTS), dtype=np.float32)
            for i in range(N_AGENTS):
                adv_buf[:, i], ret_buf[:, i] = compute_gae(
                    rew_buf[:t_end, i], val_buf[:t_end, i], done_buf[:t_end, i], v_last[i]
                )

            # Normalise advantages jointly
            adv_flat = adv_buf.reshape(-1)
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            adv_buf = adv_flat.reshape(t_end, N_AGENTS)

            # PPO update
            for i in range(N_AGENTS):
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

                        last_oh_mb = actions_to_onehot(lastact_mb, ACT_DIM)

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

            print(
                f"ep {ep:04d}  steps {t_end:4d}  "
                f"env_ret {ep_ret.mean():.2f}  "
                f"shaped_rew {shaped_rewards.mean():.2f}  "
                f"ent_coef {ent_coef} "
                f"total_steps {total_env_steps:,}"
            )


            if (ep + 1) % SAVE_EVERY == 0:
                for i, net in enumerate(nets):
                    torch.save(
                        net.state_dict(),
                        os.path.join(SAVE_PATH, f"agent_{i}_ep{ep+1}.pt"),
                    )
                print(f"Saved checkpoints at episode {ep + 1}")

        fitness_scores = fitness_tracker.compute_fitness()
        ranked = np.argsort(fitness_scores)[::-1]

        print(fitness_tracker.summary(gen))
        evolve_population(nets, optimizers, ranked)
        save_generation(nets, gen, fitness_scores)


    env.close()
    for i, net in enumerate(nets):
        torch.save(net.state_dict(), os.path.join(SAVE_PATH, f"agent_{i}_final.pt"))
    print("Training complete.")


if __name__ == "__main__":
    train()
