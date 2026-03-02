"""
Agents evaluated with this script are trained for 10 generations just like in the generation_only
script, but are evaluated in Scenario 2.
"""

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from meltingpot import scenario as scenario_lib
import dm_env
import pygame
import glob

SCENARIO_NAME = 'clean_up_2'
N_FOCAL = 3
ACT_DIM = 9
OBS_H, OBS_W, OBS_C = 88, 88, 3
FC_HIDDEN = 256
SCALE = 3
FPS = 15

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
            self.cnn_out_dim = self.cnn(torch.zeros(1, OBS_C, OBS_H, OBS_W)).shape[1]
        raw_in = self.cnn_out_dim + n_agents * act_dim
        self.policy_fc = nn.Sequential(
            nn.Linear(raw_in,    FC_HIDDEN), nn.ReLU(),
            nn.Linear(FC_HIDDEN, FC_HIDDEN), nn.ReLU(),
        )
        self.actor  = nn.Linear(FC_HIDDEN, act_dim)
        self.critic = nn.Linear(FC_HIDDEN, 1)

    def encode(self, obs_chw):
        return self.cnn(obs_chw)

    def forward_policy(self, cnn_feat, last_oh):
        x = self.policy_fc(torch.cat([cnn_feat, last_oh], dim=-1))
        return self.actor(x), self.critic(x).squeeze(-1)


def prep_obs(rgb_hwc, device):
    return (
        torch.tensor(rgb_hwc, dtype=torch.float32, device=device)
        .div_(255.0)
        .permute(2, 0, 1)
        .contiguous()
    )

def actions_to_onehot(actions, act_dim):
    return F.one_hot(actions, num_classes=act_dim).float().view(1, -1)

def get_world_rgb(env, timestep):
    try:
        return np.array(env._substrate.observation()[0]['WORLD.RGB'])
    except Exception:
        return np.array(timestep.observation[0]['RGB'])

def get_actions(nets, obs_list, last_actions, device):
    obs_tensors = torch.stack([prep_obs(obs_list[i]['RGB'], device) for i in range(N_FOCAL)])
    
    # pad to 7 — background agent actions unknown, use zeros
    last_actions_7 = torch.zeros(7, dtype=torch.int64, device=device)
    last_actions_7[:N_FOCAL] = last_actions
    
    last_oh = actions_to_onehot(last_actions_7, ACT_DIM).expand(N_FOCAL, -1)

    actions = []
    with torch.no_grad():
        for i in range(N_FOCAL):
            feat = nets[i].encode(obs_tensors[i:i+1])
            logits, _ = nets[i].forward_policy(feat, last_oh[i:i+1])
            act = Categorical(logits=logits).sample()
            actions.append(int(act.item()))
    return actions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_dir = '../checkpoints/generation_impl'
nets = []
for i in [1, 2, 6]:
    matches = glob.glob(os.path.join(ckpt_dir, f"agent_{i}_fitness*.pt"))
    assert len(matches) == 1, f"Expected 1 checkpoint for agent_{i}, found {len(matches)}"
    net = AgentNetwork(7, ACT_DIM).to(device)
    net.load_state_dict(torch.load(matches[0], map_location=device))
    net.eval()
    nets.append(net)


env = scenario_lib.build(SCENARIO_NAME)
timestep = env.reset()
obs_list = [dict(timestep.observation[i]) for i in range(N_FOCAL)]

first_frame = get_world_rgb(env, timestep)
h, w, _ = first_frame.shape


pygame.init()
screen = pygame.display.set_mode((w * SCALE, h * SCALE))
pygame.display.set_caption(f"Melting Pot — {SCENARIO_NAME}")
clock = pygame.time.Clock()

def render(env, timestep):
    frame = get_world_rgb(env, timestep)
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    scaled = pygame.transform.scale(surface, (w * SCALE, h * SCALE))
    screen.blit(scaled, (0, 0))
    pygame.display.flip()
    clock.tick(FPS)

render(env, timestep)


# Run loop
last_actions = torch.zeros(N_FOCAL, dtype=torch.int64, device=device)
ep_ret = np.zeros(N_FOCAL, dtype=np.float32)
step = 0

running = True
while running and timestep.step_type != dm_env.StepType.LAST:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    actions = get_actions(nets, obs_list, last_actions, device)
    last_actions = torch.tensor(actions, dtype=torch.int64, device=device)

    timestep = env.step(actions)
    obs_list = [dict(timestep.observation[i]) for i in range(N_FOCAL)]

    if timestep.reward is not None:
        for i in range(N_FOCAL):
            r = timestep.reward[i]
            ep_ret[i] += float(r) if r is not None else 0.0

    render(env, timestep)
    step += 1

print(f"\nEpisode complete — {step} steps")
print(f"Per-agent returns: {ep_ret}")
print(f"Mean return: {ep_ret.mean():.2f}")

pygame.quit()
env.close()
