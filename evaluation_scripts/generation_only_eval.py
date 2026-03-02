"""
Agents evaluated with this script are trained for 10 generations in a clean environment,
where every generation of 1000 episodes, a fitness function is used to determine the two best agents,
and for the next generation those two stay the same, while the other 5 are copies of the main two,
but the neural networks have small and large amounts of gaussian noise, so that the agents 
don't converge to the same policies and don't end up interfering with eachouthers learning.

These agents are evaluated in a clean environment, no background agents.
"""

import torch
from shimmy import MeltingPotCompatibilityV0
import torch.nn as nn
import os
import numpy as np
from torch.functional import F
import glob


# APPLE SILICON PyGame not rendering fix
# Not required on non-apple systems
from AppKit import NSApplication
NSApplication.sharedApplication().setActivationPolicy_(0)


SUBSTRATE = "clean_up"
N_AGENTS = 7
ACT_DIM = 9
OBS_H, OBS_W, OBS_C = 88, 88, 3

FC_HIDDEN = 256

class AgentNetwork(nn.Module):
    def __init__(self, n_agents: int, act_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim  = act_dim

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

        # POLICY NET
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


def prep_obs(rgb_hwc: np.ndarray, device) -> torch.Tensor:
    return (
        torch.as_tensor(rgb_hwc, dtype=torch.float32, device=device)
        .div(255.0)
        .permute(2, 0, 1)
        .contiguous()
    )


def actions_to_onehot(actions: torch.Tensor, act_dim: int) -> torch.Tensor:
    B, N = actions.shape
    return F.one_hot(actions, num_classes=act_dim).float().view(B, N * act_dim)



def evaluate(ckpt_dir="gen_9/", tag="final", episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nets = []
    for i in range(N_AGENTS):
        matches = glob.glob(os.path.join(ckpt_dir, f"agent_{i}_fitness*.pt"))
        assert len(matches) == 1, f"Expected 1 checkpoint for agent_{i}, found {len(matches)}"
        net = AgentNetwork(N_AGENTS, ACT_DIM).to(device)
        net.load_state_dict(torch.load(matches[0], map_location=device))
        net.eval()
        nets.append(net)

    env = MeltingPotCompatibilityV0(
        substrate_name='clean_up',
        render_mode="human" if render else None,
        max_cycles=100,
    )

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        agents = list(env.agents)
        last_actions = torch.zeros(N_AGENTS, dtype=torch.int64, device=device)
        ep_ret = {a: 0.0 for a in agents}

        while True:
            obs_tensors = torch.stack([
                prep_obs(obs_dict[agents[i]]["RGB"], device)
                for i in range(N_AGENTS)
            ])
            last_oh = actions_to_onehot(
                last_actions.unsqueeze(0).expand(N_AGENTS, -1), ACT_DIM
            )
            acts = []
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for i in range(N_AGENTS):
                    feat_i = nets[i].encode(obs_tensors[i:i+1])
                    logits_i, _ = nets[i].forward_policy(
                        feat_i, last_oh[i:i+1]
                    )
                    acts.append(torch.argmax(logits_i, dim=-1).item())

            actions_dict = {agents[i]: acts[i] for i in range(N_AGENTS)}
            obs_dict, rewards, terminations, truncations, _ = env.step(actions_dict)

            for a in agents:
                ep_ret[a] += float(rewards[a])

            last_actions = torch.tensor(acts, dtype=torch.int64, device=device)

            if all(terminations[a] or truncations[a] for a in agents):
                break

        mean_r = sum(ep_ret.values()) / N_AGENTS
        print(f"Episode {ep}: mean_return={mean_r:.2f}  per_agent={ep_ret}")

    env.close()



if __name__ == '__main__':
    evaluate()
