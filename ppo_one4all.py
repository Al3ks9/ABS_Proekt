import torch
import numpy as np
from shimmy import MeltingPotCompatibilityV0
from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

# from AppKit import NSApplication
# NSApplication.sharedApplication().setActivationPolicy_(0)


class SharedActorCritic(nn.Module):
    def __init__(self, act_dim=9):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),   # -> ~21x21
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),  # -> ~9x9
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),  # -> ~7x7
            nn.Flatten(),
        )
        # infer feature size
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 3, 88, 88)).shape[1]

        self.pi = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, act_dim))
        self.v  = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, obs_chw_float01):
        z = self.cnn(obs_chw_float01)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value



def prep_rgb(rgb_hwc_uint8, device):
    x = torch.as_tensor(rgb_hwc_uint8, device=device)
    x = x.permute(2, 0, 1).contiguous()
    x = x.to(torch.float32) / 255.0
    return x


def train(
    episodes=200,                 # number of PPO "epochs" (rollout+update cycles)
    steps_per_episode=200,         # max_cycles in env
    gamma=0.99,
    lam=0.95,                      # GAE-lambda (Spinning Up uses lam=0.97 commonly)
    clip_ratio=0.2,                # PPO clip (Spinning Up: usually 0.1-0.3)
    vf_coef=0.5,
    ent_coef=0.01,
    lr=3e-4,
    train_epochs=4,
    minibatch_size=256,
    update_device="cpu",
    save_path="ppo_clean_up.pt",
):
    actor_critic = SharedActorCritic(act_dim=9).to(update_device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    actor_critic_cpu = SharedActorCritic(act_dim=9).to("cpu")
    actor_critic_cpu.load_state_dict(actor_critic.state_dict())
    actor_critic_cpu.eval()

    env = MeltingPotCompatibilityV0(
        substrate_name="clean_up",
        render_mode=None,
        max_cycles=steps_per_episode,
    )

    for ep in tqdm(range(episodes)):
        obs, _ = env.reset()
        agents = list(env.agents)  # e.g., ['player_0', ... 'player_6']
        n_agents = len(agents)

        # Rollout storage (store uint8 to save RAM; convert on update)
        obs_buf = np.zeros((steps_per_episode, n_agents, 88, 88, 3), dtype=np.uint8)
        act_buf = np.zeros((steps_per_episode, n_agents), dtype=np.int64)
        logp_buf = np.zeros((steps_per_episode, n_agents), dtype=np.float32)
        val_buf = np.zeros((steps_per_episode, n_agents), dtype=np.float32)
        rew_buf = np.zeros((steps_per_episode, n_agents), dtype=np.float32)
        done_buf = np.zeros((steps_per_episode, n_agents), dtype=np.float32)

        ep_ret_sum = np.zeros((n_agents,), dtype=np.float32)

        # --- collect on-policy rollout ---
        for t in range(steps_per_episode):
            # pack per-agent RGB obs
            for i, a in enumerate(agents):
                obs_buf[t, i] = obs[a]["RGB"]

            # build torch batch (n_agents,3,88,88)
            obs_t = torch.stack([prep_rgb(obs[a]["RGB"], 'cpu') for a in agents], dim=0)

            with torch.no_grad():
                logits, v = actor_critic_cpu(obs_t)
                dist = Categorical(logits=logits)
                act_t = dist.sample()
                logp_t = dist.log_prob(act_t)
                val_t = v

            actions = {agents[i]: int(act_t[i].item()) for i in range(n_agents)}
            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # store transition pieces
            for i, a in enumerate(agents):
                r = float(rewards[a])
                d = bool(terminations[a] or truncations[a])
                rew_buf[t, i] = r
                done_buf[t, i] = float(d)
                act_buf[t, i] = actions[a]
                logp_buf[t, i] = float(logp_t[i].item())
                val_buf[t, i] = float(val_t[i].item())
                ep_ret_sum[i] += r

            obs = next_obs

            if all((terminations[a] or truncations[a]) for a in agents):
                t_end = t + 1
                break
        else:
            t_end = steps_per_episode

        # bootstrap value for last state (for GAE)
        with torch.no_grad():
            obs_last = torch.stack([prep_rgb(obs[a]["RGB"], 'cpu') for a in agents], dim=0)
            _, v_last = actor_critic_cpu(obs_last)
            v_last = v_last.numpy()

        # --- compute GAE advantages + returns ---
        adv_buf = np.zeros((t_end, n_agents), dtype=np.float32)
        ret_buf = np.zeros((t_end, n_agents), dtype=np.float32)

        last_gae = np.zeros((n_agents,), dtype=np.float32)
        for t in reversed(range(t_end)):
            if t == t_end - 1:
                next_values = v_last
                next_nonterminal = 1.0 - done_buf[t]
            else:
                next_values = val_buf[t + 1]
                next_nonterminal = 1.0 - done_buf[t]

            delta = rew_buf[t] + gamma * next_values * next_nonterminal - val_buf[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + val_buf[:t_end]

        # flatten (B = t_end * n_agents)
        B = t_end * n_agents
        obs_flat = obs_buf[:t_end].reshape(B, 88, 88, 3)
        act_flat = act_buf[:t_end].reshape(B)
        logp_old_flat = logp_buf[:t_end].reshape(B)
        adv_flat = adv_buf.reshape(B)
        ret_flat = ret_buf.reshape(B)

        # normalize advantages (common trick)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        actor_critic.train()

        # --- PPO updates (multiple epochs on same rollout) ---
        idxs = np.arange(B)
        for _ in range(train_epochs):
            np.random.shuffle(idxs)
            for start in range(0, B, minibatch_size):
                mb = idxs[start : start + minibatch_size]

                obs_mb = torch.stack([prep_rgb(obs_flat[i], update_device) for i in mb], dim=0)
                act_mb = torch.as_tensor(act_flat[mb], device=update_device, dtype=torch.int64)
                logp_old_mb = torch.as_tensor(logp_old_flat[mb], device=update_device, dtype=torch.float32)
                adv_mb = torch.as_tensor(adv_flat[mb], device=update_device, dtype=torch.float32)
                ret_mb = torch.as_tensor(ret_flat[mb], device=update_device, dtype=torch.float32)

                logits, v = actor_critic(obs_mb)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_mb
                pi_loss = -(torch.min(surr1, surr2)).mean()

                v_loss = F.mse_loss(v, ret_mb)

                loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                optimizer.step()

        actor_critic_cpu.load_state_dict(actor_critic.state_dict())
        actor_critic_cpu.eval()

        print(f"ep {ep:04d}  steps {t_end:4d}  mean_return {ep_ret_sum.mean():.2f}")

    env.close()
    torch.save(actor_critic.state_dict(), save_path)


def run_trained_policy(
    ckpt_path="ppo_clean_up.pt",
    episodes=5,
    max_cycles=200,
    render=True,
    greedy=True,
    device="cpu",
):
    # Load model
    net = SharedActorCritic(act_dim=9).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    env = MeltingPotCompatibilityV0(
        substrate_name="clean_up",
        render_mode="human" if render else None,
        max_cycles=max_cycles,
    )

    for ep in range(episodes):
        obs, infos = env.reset()
        agents = list(env.agents)

        ep_return = {a: 0.0 for a in agents}

        while True:
            # Build batch (n_agents,3,88,88)
            obs_batch = torch.stack([prep_rgb(obs[a]["RGB"], device) for a in agents], dim=0)

            with torch.no_grad():
                logits, v = net(obs_batch)
                if greedy:
                    acts = torch.argmax(logits, dim=-1)
                else:
                    dist = Categorical(logits=logits)
                    acts = dist.sample()

            actions = {agents[i]: int(acts[i].item()) for i in range(len(agents))}
            obs, rewards, terminations, truncations, infos = env.step(actions)

            if render:
                env.render()  # Shimmy uses this to update the window in human mode [page:1]

            for a in agents:
                ep_return[a] += float(rewards[a])

            if all(terminations[a] or truncations[a] for a in agents):
                break

        print(f"Episode {ep}: mean_return={sum(ep_return.values())/len(ep_return):.2f} per_agent={ep_return}")

    env.close()



if __name__ == '__main__':
    train(update_device='mps', episodes=100, steps_per_episode=600, minibatch_size=512)
    # run_trained_policy()
