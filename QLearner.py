import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquad
from torchquad import Trapezoid
from torchquad import GaussLegendre
import itertools
from loguru import logger
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import yfinance as yf
logger.remove()
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

def Discrete_Action_Space(n_hedge, levels = [-1, 0, 1]):
    action_space = torch.tensor(list(itertools.product(levels, repeat = n_hedge)), dtype = torch.float64)
    return action_space

@dataclass
class HedgingEnvCfg:
    pos_cap: float | None        # absolute cap per instrument (None for no cap)
    trade_unit: List[float]      # size of one trade step per instrument
    tc: float                    # transaction cost rate (e.g., 0.001 = 10 bps)
    include_interest: bool = True

class HedgingEnv:
    """
    Offline episodic env over pre-simulated data:
    - X: [N, T+1, 3]  (S, v, tau)
    - PH: [N, T+1, M] hedge prices, with PH[:, -1, :] = payoffs at T
    - liability: provides .T, .r, .Payoff(ST) and .Price() (your class)
    - Action is an index into a discrete action table A: [A, M] trade deltas (multipliers)
    """
    def __init__(self, X, price_hedge, liability, action_space, cfg: HedgingEnvCfg):
        #assert X.shape[0] == price_hedge.shape[0] and X.shape[1] == price_hedge.shape[1]
        self.X = X
        self.price_hedge = price_hedge
        self.liability = liability
        self.A = action_space  # [A, M] trade multipliers
        self.cfg = cfg

        self.n_sim, self.n_step1, _ = X.shape
        self.n_hedge = price_hedge.shape[-1]  # number of hedge instruments
        self.n_step = self.n_step1 - 1
        self.dt = liability.T / self.n_step
        self.device = X.device
        self.dtype = X.dtype

        # runtime state
        self.ss = None  # path index
        self.tt = None
        self.pos = None
        self.cash = None

    def reset(self, path_index = None):
        self.ss = int(torch.randint(0, self.n_sim, ()).item()) if path_index is None else int(path_index)
        self.tt = 0
        self.pos = torch.zeros(self.n_hedge, dtype=self.dtype, device=self.device)
        # Start with liability price as initial cash (self-financing replication budget)
        self.cash = torch.as_tensor(self.liability.Price(), dtype=self.dtype, device=self.device)
        return self._state()
    
    def _state(self):
        # state = [S_t, v_t, tau_t, pos_1..pos_M]
        x_t = self.X[self.ss, self.tt, :]                   # [3]
        return torch.cat([x_t, self.pos], dim=0)          # [3+M]

    def step(self, action_index):
        # map action index → vector of trade deltas (multipliers * trade_unit)
        d_mult = self.A[action_index].to(self.device)       # [M]
        d_hedge = d_mult * torch.as_tensor(self.cfg.trade_unit, dtype=self.dtype, device=self.device)  # [M]

        # cap: ensure new position within [-cap, cap]
        if self.cfg.pos_cap is not None:
            new_pos_prop = self.pos + d_hedge
            cap = torch.full_like(new_pos_prop, self.cfg.pos_cap)
            new_pos = torch.clamp(new_pos_prop, -cap, cap)
            d_hedge = new_pos - self.pos
        else:
            new_pos = self.pos + d_hedge

        # prices at t
        price_hedge_t = self.price_hedge[self.ss, self.tt, :]  # [M]
        # transaction cost
        TC_t = (torch.abs(d_hedge) * price_hedge_t).sum() * self.cfg.tc

        # interest accrual then trade cash accounting (consistent with your PnL fix)
        if self.cfg.include_interest:
            self.cash = self.cash * (1 + self.liability.r * self.dt)
        proceed = (price_hedge_t * d_hedge).sum()
        self.cash = self.cash - proceed - TC_t
        self.pos = new_pos

        # advance time
        self.tt += 1
        done = (self.tt == self.n_step)  # next obs is terminal

        # reward: 0 until terminal; at terminal, realize PnL
        if done:
            payoff_hedge = self.price_hedge[self.ss, -1, :]  # [M], payoffs at T
            ST = payoff_hedge[0]                   # underlying payoff channel
            payoff_hedge = (payoff_hedge * self.pos).sum()
            payoff_liab = self.liability.Payoff(ST)
            if self.cfg.include_interest:
                pnl = self.cash * (1 + self.liability.r * self.dt) + payoff_hedge - payoff_liab
            else:
                pnl = self.cash + payoff_hedge - payoff_liab
            reward = torch.tensor(pnl, dtype=self.dtype, device=self.device)
        else:
            reward = torch.tensor(0.0, dtype=self.dtype, device=self.device)   

        return self._state(), reward, done
    
class QNet(nn.Module):
    # Small MLP; input = 3 + M, output = n_actions
    def __init__(self, in_dim, n_actions, hidden = 128, dtype = torch.float64, device = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=dtype, device=device), nn.ReLU(),
            nn.Linear(hidden, hidden, dtype=dtype, device=device), nn.ReLU(),
            nn.Linear(hidden, n_actions, dtype=dtype, device=device),
        )
    
    def forward(self, x):  # x: [B, in_dim]
        return self.net(x)

class Replay:
    def __init__(self, cap = 100000):
        self.s, self.a, self.r, self.ns, self.d = [], [], [], [], []
        self.cap = cap

    def push(self, s, a, r, ns, done):
        if len(self.s) >= self.cap:
            self.s.pop(0); self.a.pop(0); self.r.pop(0); self.ns.pop(0); self.d.pop(0)
        self.s.append(s); self.a.append(a); self.r.append(r); self.ns.append(ns); self.d.append(done)
        
        # ensure consistent shapes & types
        self.s.append(torch.as_tensor(s).detach())
        self.a.append(int(a))
        r_fixed = torch.as_tensor(r).reshape(1)        # <-- always [1]
        self.r.append(r_fixed.detach())
        self.ns.append(torch.as_tensor(ns).detach())
        self.d.append(bool(done))

    def sample(self, B, device, dtype):
        import random
        idx = random.sample(range(len(self.s)), B)
        S  = torch.stack([self.s[i]  for i in idx]).to(device=device, dtype=dtype)
        A  = torch.tensor([self.a[i] for i in idx], device=device, dtype=torch.long)
        r_vals = [torch.as_tensor(self.r[i]).reshape(()).item() for i in idx]
        R = torch.tensor(r_vals, device=device, dtype=dtype).unsqueeze(1)
        NS = torch.stack([self.ns[i] for i in idx]).to(device=device, dtype=dtype)
        D  = torch.tensor([self.d[i] for i in idx], device=device, dtype=torch.bool).unsqueeze(1)
        return S, A, R, NS, D
    
    def __len__(self):
        return len(self.s)

@torch.no_grad()
def epsilon_greedy(qnet, s, eps, n_actions):
    if torch.rand(()) < eps:
        return int(torch.randint(0, n_actions, ()).item())
    q = qnet(s.unsqueeze(0))  # [1, A]
    return int(q.argmax(dim=-1).item())

def dqn_train(env: HedgingEnv, n_actions, n_epoch = 10_000, batch_size = 256, gamma = 1.0, lr = 1e-3, eps_start = 0.5,
              eps_end = 0.05, eps_decay_episodes = 5_000, target_sync = 250, hidden = 128):
    device, dtype = env.device, env.dtype
    in_dim = 3 + env.n_hedge

    q = QNet(in_dim, n_actions, hidden, dtype, device)
    qt = QNet(in_dim, n_actions, hidden, dtype, device)
    qt.load_state_dict(q.state_dict())
    optimiser = torch.optim.Adam(q.parameters(), lr=lr)

    buffer = Replay(cap=200_000)

    def eps_schedule(epoch):
        if epoch >= eps_decay_episodes: return eps_end
        frac = epoch / max(1, eps_decay_episodes)
        return eps_start + (eps_end - eps_start) * frac

    for epoch in tqdm(range(n_epoch), ncols = 100):
        s = env.reset()  # [3+M]
        done = False
        while not done:
            a = epsilon_greedy(q, s, eps_schedule(epoch), n_actions)
            ns, r, done = env.step(a)
            buffer.push(s, a, r, ns, done)
            s = ns

            # learn
            if len(buffer) >= 2000:
                S, A, R, NS, D = buffer.sample(batch_size, device, dtype)
                with torch.no_grad():
                    # Double DQN
                    next_act = q(NS).argmax(dim=-1, keepdim=True)             # [B,1]
                    target_q = qt(NS).gather(1, next_act)                     # [B,1]
                    Y = R + (~D).to(dtype) * gamma * target_q                 # terminal has D=True

                Qsa = q(S).gather(1, A.unsqueeze(1))
                loss = F.mse_loss(Qsa, Y)

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                optimiser.step()

        if (epoch+1) % target_sync == 0:
            qt.load_state_dict(q.state_dict())

    return q

@torch.no_grad()
def policy_rollout_deltas(qnet, X, price_hedge, liability, action_table, cfg: HedgingEnvCfg):
    """
    Vectorized per-episode rollout (loop over paths) to produce d_hedge for each path.
    Returns: d_hedge [N, T, M]
    """
    n_sim, n_step1, _ = X.shape
    n_hedge = price_hedge.shape[-1]
    n_step = n_step1 - 1
    device, dtype = X.device, X.dtype

    d_all = torch.zeros(n_sim, n_step, n_hedge, dtype=dtype, device=device)

    for i in range(n_sim):
        position = torch.zeros(n_hedge, dtype=dtype, device=device)
        cash = torch.as_tensor(liability.Price(), dtype=dtype, device=device)
        dt = liability.T / n_step

        for tt in range(n_step):
            state = torch.cat([X[i, tt, :], position], dim=0).unsqueeze(0)     # [1, 3+M]
            qvals = qnet(state)
            a = int(qvals.argmax(dim=-1).item())
            d_mult = action_table[a].to(device=device, dtype=dtype)      # [M]
            d_hedge = d_mult * torch.as_tensor(cfg.trade_unit, dtype=dtype, device=device)

            # cap enforcement
            if cfg.pos_cap is not None:
                new_pos_prop = position + d_hedge
                cap = torch.full_like(new_pos_prop, cfg.pos_cap)
                new_pos = torch.clamp(new_pos_prop, -cap, cap)
                d_hedge = new_pos - position
            else:
                new_pos = position + d_hedge

            # book-keeping
            price_hedge_t = price_hedge[i, tt, :]
            TC_t = (torch.abs(d_hedge) * price_hedge_t).sum() * cfg.tc
            cash = cash * (1 + liability.r * dt) - (price_hedge_t * d_hedge).sum() - TC_t
            d_all[i, tt, :] = d_hedge
            position = new_pos

        # (terminal PnL is computed later via your PnL)
    return d_all
def main():
    n_hedge = price_hedge_train.shape[-1]
    trade_unit = 0.1
    levels = [-1.0, 0.0, 1.0]                 # 3-way per instrument; expand if you like
    Action_space = Discrete_Action_Space(n_hedge, levels)   # [A, M]
    cfg = HedgingEnvCfg(pos_cap = 50.0, trade_unit = [trade_unit] * n_hedge, tc = tc, include_interest=True)
    env = HedgingEnv(X_train, price_hedge_train, liability, Action_space, cfg)
    qnet = dqn_train(env, n_actions=Action_space.shape[0], n_epoch=5000, batch_size=512, gamma=1.0, lr=1e-3)

    # Rollout on test set, compute CVaR
    d_hedge_test = policy_rollout_deltas(qnet, X_test, price_hedge_test, liability, Action_space, cfg)  # [N_te, T, M]
    pnl_Q, pnl_NH = PnL(d_hedge_test, price_hedge_test, liability, transaction_cost=cfg.tc)
    for a in [0.01, 0.05, 0.10]:
        print(f"CVaR@{a}: {CVaR_PnL(pnl_Q, a).item():.4f}  (No-hedge: {CVaR_PnL(pnl_NH, a).item():.4f})")