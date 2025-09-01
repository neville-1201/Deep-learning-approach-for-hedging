import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog
import torchquad
from torchquad import Trapezoid
from torchquad import GaussLegendre
from loguru import logger
import pandas as pd

import time
from tqdm import tqdm
import os 
import HestonModel
import QLearner
import QLearnerHedging
import matplotlib.pyplot as plt
import gc
from dataclasses import dataclass
from typing import List, Tuple
gc.collect()
#torch.autograd.set_detect_anomaly(True)
current_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Neural Network Based Deep Hedging ###

# Define the class

class DeepHedging_NN_d(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.input_dim = input_dim #number of input features of each observation: St, vt, 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim #number of risky hedging instrument (No risk-free note)

        dtype = dtype
        device = device

        self.fc_in = nn.Linear(input_dim, hidden_dim, dtype = dtype, device = device)
        self.gru = nn.GRUCell(hidden_dim + output_dim, hidden_dim, dtype = dtype, device = device)
        self.fc_out = nn.Linear(hidden_dim, output_dim, dtype = dtype, device = device)

    def forward(self, X):
        n_sim, n_step, _ = X.shape
        output_dim = self.fc_out.out_features
        hidden = torch.zeros(n_sim, self.hidden_dim, dtype = X.dtype, device = X.device)
        d_hedge = torch.zeros(n_sim, output_dim, dtype = X.dtype, device = X.device)
        actions = []

        for tt in range(n_step):
            X_t = X[:, tt, :]
            input_t = self.fc_in(X_t)
            gru_input = torch.cat([input_t, d_hedge], dim = -1)
            hidden = self.gru(gru_input, hidden)
            d_hedge = self.fc_out(hidden)
            actions.append(d_hedge)
        
        return torch.stack(actions, dim = 1)
        
class DeepHedging_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, cap =  None, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.input_dim = input_dim #number of input features of each observation: St, vt, 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim #number of risky hedging instrument (No risk-free note)
        self.cap = cap
        dtype = dtype
        device = device

        self.gru = nn.GRUCell(input_dim + output_dim, hidden_dim, dtype = dtype, device = device)
        self.fc_out = nn.Linear(hidden_dim, output_dim, dtype = dtype, device = device)

    def forward(self, X):
        n_sim, n_step, _ = X.shape
        dtype = X.dtype
        device = X.device
        output_dim = self.fc_out.out_features
        hidden = torch.zeros(n_sim, self.hidden_dim, dtype = dtype, device = device)
        pos_hedge = torch.zeros(n_sim, output_dim, dtype = dtype, device = device)
        actions = []

        cap = None
        if getattr(self, "cap", None) is not None:
            cap = torch.as_tensor(self.cap, dtype = dtype, device = device)

        for tt in range(n_step - 1):
            X_t = X[:, tt, :]
            gru_input = torch.cat([X_t, pos_hedge], dim = -1)
            hidden = self.gru(gru_input, hidden)
            d_hedge = self.fc_out(hidden)
            
            if cap is not None:
                d_hedge = torch.clamp(d_hedge, -cap - pos_hedge, cap - pos_hedge) 
                
            pos_hedge = pos_hedge + d_hedge
            actions.append(d_hedge)
        
        return torch.stack(actions, dim = 1)

class DeepHedging_NN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, cap =  None, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.input_dim = input_dim #number of input features of each observation: St, vt, 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim #number of risky hedging instrument (No risk-free note)
        self.cap = cap
        dtype = dtype
        device = device

        self.ff_in = nn.Linear(input_dim + output_dim, hidden_dim, dtype = dtype, device = device)
        self.ff_mid = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, output_dim, dtype = dtype, device = device)

    def forward(self, X):
        n_sim, n_step, _ = X.shape
        dtype = X.dtype
        device = X.device
        output_dim = self.fc_out.out_features
        hidden = torch.zeros(n_sim, self.hidden_dim, dtype = dtype, device = device)
        pos_hedge = torch.zeros(n_sim, output_dim, dtype = dtype, device = device)
        actions = []

        cap = None
        if getattr(self, "cap", None) is not None:
            cap = torch.as_tensor(self.cap, dtype = dtype, device = device)

        for tt in range(n_step - 1):
            X_t = X[:, tt, :]
            z_t = torch.cat([X_t, pos_hedge], dim = -1)
            hidden = self.ff_mid(self.ff_in(z_t))
            d_hedge = self.fc_out(hidden)
            
            if cap is not None:
                d_hedge = torch.clamp(d_hedge, -cap - pos_hedge, cap - pos_hedge) 
                
            pos_hedge = pos_hedge + d_hedge
            actions.append(d_hedge)
        
        return torch.stack(actions, dim = 1)


def GreekHedging_0(X, liability, target_greeks, hedge_type, hedge_K, hedge_T = None, max_pos = None, min_pos = None):
    '''
    X as a matrix of n_sim simulation of n_step step of St, vt and tau
    liability as a class Derivative object, the liability or risk exposure to hedge
    hedge_type as a list of string, the type of hedging instuments
    '''
    device = X.device()
    if max_pos is None:
        max_pos = torch.full((n_hedge), float("inf"), dtype = torch.float64, device = device)
    if min_pos is None:
        min_pos = torch.full((n_hedge), -float("inf"), dtype = torch.float64, device = device)
    
    # liability
    T = liability.T
    r = liability.r
    q = liability.q
    v0 = liability.v0
    
    # hedging instrument
    n_hedge = len(hedge_type)
    n_sim, n_step, _ = X.shape
    S0 = X[0,0,0].item()
    hedging_instruments = []
    pos_hedge = torch.zeros(n_sim, n_step, n_hedge, dtype = torch.float64)
    d_hedge = torch.zeros(n_sim, n_step, n_hedge, dtype = torch.float64)
    
    for hh in range(n_hedge):
        hedge = Derivative(hedge_type[hh], S0, r, q, v0, T, hedge_K[hh])
        hedging_instruments.append(hedge)

    
    for tt in tqdm(range(n_step), ncols = 50, leave=False):
        tau = X[0, tt, 2].item()
        for ss in range(n_sim):
            St = X[ss, tt, 0]
            vt = X[ss, tt, 1]
            lia_greek = -liability.Greeks(St, vt, tau)[0:min(n_hedge, 4)]
            hedge_greek = torch.stack([hedging_instruments[hh].Greeks(St, vt, tau) for hh in range(n_hedge)], dim = 0)[:,0:min(n_hedge, 4)]
            result_greek = target_greeks - lia_greek
            reg = 1e-6 * torch.eye(hedge_greek.shape[0], device=hedge_greek.device)
            pos_hedge_ss = result_greek @ torch.linalg.inv(hedge_greek + reg)
            pos_hedge_ss = torch.max(torch.min(pos_hedge_ss, max_pos), min_pos)
            #print(d_hedge_ss @ hedge_greek + lia_greek)
            pos_hedge[ss, tt, :] = pos_hedge_ss.view(-1)
            if tt == 0:
                d_hedge[ss, tt, :] = pos_hedge_ss.view(-1)
            else:
                d_hedge[ss, tt, :] = pos_hedge[ss, tt, :] - pos_hedge[ss, tt - 1, :]
    return d_hedge

def GreekHedging(X, liability, target_greeks, hedge_type, hedge_K, hedge_T = None, max_pos = None, min_pos = None):
    '''
    X as a matrix of n_sim simulation of n_step step of St, vt and tau
    liability as a class Derivative object, the liability or risk exposure to hedge
    hedge_type as a list of string, the type of hedging instuments
    '''
    device = X.device
    dtype = X.dtype
    if max_pos is None:
        max_pos = torch.full((n_hedge), float("inf"), dtype = torch.float64, device = device)
    if min_pos is None:
        min_pos = torch.full((n_hedge), -float("inf"), dtype = torch.float64, device = device)
    
    # liability
    T = liability.T
    r = liability.r
    q = liability.q
    v0 = liability.v0
    
    # hedging instrument
    n_hedge = len(hedge_type)
    n_sim, n_step, _ = X.shape
    S0 = X[0,0,0].item()
    hedging_instruments = []
    pos_hedge = torch.zeros(n_sim, n_step, n_hedge, dtype = torch.float64)
    d_hedge = torch.zeros(n_sim, n_step, n_hedge, dtype = torch.float64)

    for hh in range(n_hedge):
        hedge = Derivative(hedge_type[hh], S0, r, q, v0, T, hedge_K[hh])
        hedging_instruments.append(hedge)

    n_greek = int(min(4, n_hedge))
    
    for tt in tqdm(range(n_step), ncols = 100):
        tau = X[0, tt, 2].item()
        for ss in range(n_sim):
            St = X[ss, tt, 0].item()
            vt = X[ss, tt, 1].item()
            prev_pos = pos_hedge[ss, tt-1, :] if tt > 0 else torch.zeros(n_hedge, dtype=dtype, device=device)

            lia_greek = -liability.Greeks(St, vt, tau)[0:n_greek]
            hedge_greek = torch.stack([hedging_instruments[hh].Greeks(St, vt, tau) for hh in range(n_hedge)], dim = 0)[:,0:n_greek]
            result_greek = target_greeks - lia_greek

            if n_hedge <= n_greek:
                reg = 1e-6 * torch.eye(hedge_greek.shape[0], device=hedge_greek.device)
                pos_hedge_ss = result_greek @ torch.linalg.inv(hedge_greek + reg)
                pos_hedge_ss = torch.max(torch.min(pos_hedge_ss, max_pos), min_pos)
            else:
                A = hedge_greek.T
                c_vec = (result_greek - (A @ prev_pos))
                price_hedge_t = [hedging_instruments[hh].Price(St = St, v0 = vt, T = tau) for hh in range(n_hedge)]
                p_np = np.asarray(price_hedge_t, dtype = np.float64)
                Aeq_np = A.detach().cpu().numpy()
                beq_np = c_vec.detach().numpy()
                
                lower_bound = (min_pos - prev_pos).detach().cpu().numpy()
                upper_bound = (max_pos - prev_pos).detach().cpu().numpy()
                bounds = list(zip(lower_bound, upper_bound))

                res = linprog(c=p_np, A_eq=Aeq_np, b_eq=beq_np, bounds=bounds, method="highs")
                if res.success:
                    delta = torch.from_numpy(res.x).to(device=device, dtype=dtype)
                    pos_hedge_ss = torch.clamp(prev_pos + delta, min_pos, max_pos)
                else:
                    # fallback: least-squares on positions, then clamp
                    # (min-norm solution to A pos = result_greek)
                    pos_ls = torch.linalg.lstsq(A, result_greek.unsqueeze(-1)).solution.squeeze(-1)
                    pos_hedge_ss = torch.clamp(pos_ls, min_pos, max_pos)

            pos_hedge[ss, tt, :] = pos_hedge_ss.view(-1)
            if tt == 0:
                d_hedge[ss, tt, :] = pos_hedge_ss.view(-1)
            else:
                d_hedge[ss, tt, :] = pos_hedge[ss, tt, :] - pos_hedge[ss, tt - 1, :]
    return d_hedge

class Derivative(nn.Module):
    def __init__(self, type, S0, r, q, v0, T, K, notional = 1.0, B = None, basket_member = None, basket_K = None, basket_position = None, Heston_Model = None):
    #def __init__(self, type, T, K, r, q, vt, notional = 1, B = None, basket_member = None, basket_K = None, basket_position = None, Heston_Model = None):
        super().__init__()
        self.type = type
        self.S0 =S0
        self.r = r
        self.q = q
        self.v0 = v0
        self.T = T
        self.K = K
        self.notional = notional
        self.B = B
        self.HM = Heston_Model
        self.basket_member = basket_member
        self.basket_K = basket_K
        self.basket_position = basket_position
        if self.type == "basket":
            self.n_comp = len(self.basket_member)
            self.basket = []
            for cc in range(self.n_comp):
                self.basket.append(Derivative(self.basket_member[cc], self.S0, self.r, self.q, v0, T, self.basket_K[cc]))

    def Price(self, St = None, r = None, q = None, v0 = None, T=None):
        if St is None:
            St = self.S0
        if r is None:
            r = self.r
        if q is None:
            q = self.q
        if v0 is None:
            v0 = self.v0
        if T is None:
            T = self.T
        d1 = (np.log(St / (self.K + 1e-12)) + (r - q + v0 / 2) * T) / np.sqrt(v0 * T + 1e-12)
        d2 = d1 - np.sqrt(v0 * T)
        N1 = norm.cdf(d1)
        N2 = norm.cdf(d2)
        if self.type == "call" or self.type == "Call":
            price = St * np.exp(-q * T) * N1 - self.K * np.exp(-r * T) * N2
        elif self.type == "digital_call":
            price =  self.notional * np.exp(-r * T) * N2
        elif self.type == "ELN":
            price = 0
        elif self.type == "put":
            price = - St * np.exp(-q * T) * (1 - N1) + self.K * np.exp(-r * T) * (1 - N2)
        elif self.type == "underlying" or self.type == "spot":
            price = St
        elif self.type == "basket":
            price = 0.0
            for cc in range(self.n_comp):
                price = price + self.basket[cc].Price(St, r, q, v0, T) * self.basket_position[cc]
        return price

    def Payoff(self, ST):
        '''
        ===Input Definition===
        ST (n_sim) as a tensor of underling price at T or (n_sim, n_step) as a tensor of underlying price at each time step for Knock-in ELN
        ===Output Definition===
        payoff (n_sim) as a tensor of payoff of longing the instrument at T 
        '''
        torch.save(ST, f"{current_dir}/ST.pt")
        if ST.ndim == 0:
            ST = ST.unsqueeze(0)
        if not torch.is_tensor(ST):
            ST = torch.tensor([ST], dtype = torch.float64).flatten()
        n_sim = ST.shape[0]
        if self.type == "call" or self.type == "Call":
            payoff = torch.clamp(ST - self.K, min = 0)
        elif self.type == "put":
            payoff = torch.clamp(self.K - ST, min = 0)
        elif self.type == "digital_call":
            payoff =  self.notional * (ST >= self.K)
        elif self.type == "ELN":
            payoff = torch.zeros(n_sim)
        elif self.type == "underlying" or self.type == "spot":
            payoff = ST
        elif self.type == "basket":
            payoff = torch.zeros(n_sim)
            for cc in range(self.n_comp):
                payoff = payoff + self.basket[cc].Payoff(ST) * self.basket_position[cc]
                    
        return payoff.to(dtype = torch.float64)
    
    # calculate Greeks
    def Greeks(self, ST, vt, T):
        if not torch.is_tensor(ST):
            ST = torch.tensor([ST], dtype = torch.float64).flatten()
        if not torch.is_tensor(T):
            T = torch.tensor([T], dtype = torch.float64).flatten()
        if not torch.is_tensor(vt):
            vt = torch.tensor([vt], dtype = torch.float64).flatten()

        d1 = (torch.log(ST / self.K) + (self.r - self.q + vt / 2) * T) / torch.sqrt(vt * T)
        d2 = d1 - torch.sqrt(vt * T)
        N1 = 0.5 * (1 + torch.erf(d1 / 2 ** 0.5))
        N2 = 0.5 * (1 + torch.erf(d2 / 2 ** 0.5))
        n1 = torch.exp(-0.5 * d1 ** 2) / torch.sqrt(2 * torch.tensor(torch.pi, dtype = torch.float64))
        n2 = torch.exp(-0.5 * d2 ** 2) / torch.sqrt(2 * torch.tensor(torch.pi, dtype = torch.float64))
        eps = 1e-12
        df_q = torch.exp(-self.q * T)
        df_r = torch.exp(-self.r * T)

        if self.type == "call" or self.type == "Call":
            delta = df_q * N1 + eps
            gamma = df_q * n1 / (ST * torch.sqrt(vt * T)) + eps
            vega = df_q * ST * n1 * torch.sqrt(T) + eps
            theta = -df_q * ST * n1 * torch.sqrt(vt / T) / 2 - self.r * df_r * self.K * N2 + self.q * df_q * ST * N1 + eps
        elif self.type == "put" or self.type == "Put":
            delta = df_q * (N1 - 1.0) - eps
            gamma = df_q * n1 / (ST * torch.sqrt(vt * T)) + eps
            vega = df_q * ST * n1 * torch.sqrt(T) + eps
            theta = -df_q * ST * n1 * torch.sqrt(vt / T) / 2 + self.r * df_r * self.K * (1 - N2) - self.q * df_q * ST * (1 - N1) + eps
        elif self.type == "digital_call":
            delta = (df_r * n2 / (ST * torch.sqrt(vt * T)) + eps) * self.notional
            gamma = (- df_r * (n2 * d2 / (ST ** 2 * vt * T)) + n2 / (ST ** 2 * torch.sqrt(vt * T)) + eps) * self.notional
            vega = (df_r * n2 * (-d1 / torch.sqrt(vt)) + eps) * self.notional
            d2_dT = ((self.r - self.q - 0.5 * vt) - torch.log(ST / self.K) / T) / (2 * torch.sqrt(vt * T)) + eps
            theta = (-self.r * df_r * N2 + df_r * n2 * d2_dT + eps) * self.notional
        elif self.type == "ELN":
            delta = eps
            gamma = eps
            vega = eps
            theta = eps
        elif self.type == "underlying" or self.type == "spot":
            delta = 1
            gamma = eps
            vega = eps
            theta = eps
        elif self.type == "basket":
            greek = torch.zeros(4, dtype = torch.float64)
            for cc in range(self.n_comp):
                greek = greek + self.basket[cc].Greeks(ST, vt, T) * self.basket_position[cc]
            greek = greek + eps
            delta = greek[0]
            gamma = greek[1]
            vega = greek[2]
            theta = greek[3]
        return torch.tensor([delta, gamma, vega, theta], dtype = torch.float64)

### Final PnL of each simulation with hedging actions from neural network ###
def PnL(d_hedge, price_hedge, liability, transaction_cost = 0.001, TC_output = False):
    '''
    ===Input Definition===
    d_hedge (n_sim, n_step, n_hedge) as a tensor of n_sim simulation of n_step (from t=0 to t=T-dt) of change of position of each hedging instrument (t = dt, 2dt, ..., T-dt)
    price_hedge (n_sim, n_step + 1, n_hedge) as a tensor of n_sim simulation of n_step + 1 (from t=0 to t=T) of price of each hedging instrument, *price at T = payoff at T*
    liability as a class/function object that give payoff of liability at T
    T as a scalar, time horizon
    r as a scalar, risk-free rate
    transaction_cost as a scalar, percentage transaction on the money value of the trades
    ===Output Definition===
    PnL (n_sim) as a tensor of final profit or loss at T
    '''
    n_sim, n_step, n_hedge = d_hedge.shape
    T = liability.T
    r = liability.r
    dt = T/n_step 

    cash_0 = liability.Price()
    cash = torch.ones(n_sim, device = price_hedge.device) * cash_0
    #cash = torch.zeros(n_sim, device = price_hedge.device)
    position = torch.zeros(n_sim, n_hedge, device = price_hedge.device)
    total_TC = torch.zeros(n_sim, device = price_hedge.device)
    total_interest = torch.zeros(n_sim, device = price_hedge.device)
    # from t=0 to t=T-dt
    for tt in range(n_step):
        # total proceed from buying and selling the hedging instruments
        price_hedge_t = price_hedge[:, tt, :]
        d_hedge_t = d_hedge[:, tt, :]
        position = position + d_hedge_t
        proceed_t = torch.sum(price_hedge_t * d_hedge_t, dim = -1)

        # transaction cost
        TC_t = torch.sum(torch.abs(d_hedge_t) * price_hedge_t, dim = -1) * transaction_cost
        total_TC = total_TC + TC_t

        # interest income or expense
        interest_t = cash * (r * dt)
        total_interest = total_interest + interest_t

        # update cash balance
        cash = cash - proceed_t - TC_t + interest_t
    
    # final step, at T
    payoff_hedge = price_hedge[:, -1, :] #payoff at T of each hedging instruments
    value_hedge = torch.sum(payoff_hedge * position, dim = -1)
    payoff_liability = liability.Payoff(payoff_hedge[:, 0])
    PnL = value_hedge + cash * (1 + r * dt) - payoff_liability
    PnL_NH = - payoff_liability + cash_0 * (1 + r * T)
    if TC_output:
        return PnL, PnL_NH , total_TC
    else:
        return PnL, PnL_NH

### Calculation of CVaR from the PnL ###
def CVaR_PnL(PnL, alpha):
    '''
    ===Input Definition===
    PnL (n_sim) as a tensor of final profit or loss at T
    alpha as a scalar, tail probability
    ===Output Definition===
    CVaR as a scalar
    '''
    alpha = float(torch.clamp(torch.tensor(alpha), 1e-8, 1 - 1e-8))

    loss = -PnL
    VaR = torch.quantile(loss.detach(), 1.0 - alpha)
    tail = F.relu(loss - VaR)
    CVaR = VaR + tail.mean() / alpha

    return CVaR

def CVar_Loss(d_hedge, price_hedge, liability, alpha, transaction_cost = 0.001, TC_output = False):
    pnl, _ = PnL(d_hedge, price_hedge, liability, transaction_cost)
    loss = CVaR_PnL(pnl, alpha)
    return loss

def Training(model, liability, Loss, X, price_hedge, T, r, tc, alpha, n_epoch = 100, lr = 1e-3, batch_size = 512, val_portion = 1/9 , clip = 1.0, verbose = True, early_terminate = False, patience = 10, min_improve = 1e-4):
    '''
    ===Input Definition===
    data: X as (St, vt, tau)
    ===Output Definition===
    '''

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    n_sim = X.shape[0]

    # Data Split
    X = X.to(device = device, dtype = dtype).detach()
    price_hedge = price_hedge.to(device = device, dtype = dtype).detach()
    index1 = torch.randperm(n_sim, device = device)
    n_Va = int(round(val_portion * n_sim))
    n_Tr = n_sim - n_Va
    Tr_index = index1[n_Va:]
    Va_index = index1[:n_Va]

    X_Tr = X[Tr_index]
    price_hedge_Tr = price_hedge[Tr_index]
    X_Va = X[Va_index]
    price_hedge_Va = price_hedge[Va_index]

    # Optimisation setup
    optimiser  = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.999), eps = 1e-8)
    #optimiser = torch.optim.SGD(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = max(n_epoch//5, 1), gamma = 0.5)
    best_loss_Va = float("inf")
    wait = 0

    cash_0 = 55.80

    # Training
    model.train()
    for epoch in tqdm(range(n_epoch), ncols = 100):
        index2 = torch.randperm(n_Tr, device = device)
        epoch_loss = 0.0
        n_batch = 0

        for start in range(0, n_Tr, batch_size):
            end = min(start + batch_size, n_Tr)
            batch_index =  index2[start:end]
            X_batch = X_Tr[batch_index]
            price_hedge_batch = price_hedge_Tr[batch_index]

            optimiser.zero_grad(set_to_none = True)

            d_hedge_batch = model(X_batch)

            loss = Loss(d_hedge_batch, price_hedge_batch, liability, alpha, transaction_cost = tc, TC_output = False)
            loss.backward()

            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimiser.step()

            epoch_loss = epoch_loss + float(loss.item())
            n_batch = n_batch + 1
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            d_hedge_Va = model(X_Va)
            loss_Va = float(Loss(d_hedge_Va, price_hedge_Va, liability, alpha, transaction_cost = tc, TC_output = False).item())
        
        if best_loss_Va - loss_Va > min_improve:
            best_loss_Va = loss_Va
            wait = 0
        else:
            wait = wait + 1
        
        if early_terminate and wait >= patience:
            if verbose:
                tqdm.write(f"###Early terminated without improvement greater than {min_improve} for {patience} epochs.")
                tqdm.write(f"Epoch {epoch + 1: 02d}/{n_epoch} | Training CVaR: {epoch_loss/n_batch: .6f} | Validation CVaR: {epoch_loss/n_batch: .6f} [Best: {best_loss_Va: .6f}] | LR: {cur_lr: .3g}")
            break

        if verbose:
            cur_lr = optimiser.param_groups[0]["lr"]
            if epoch % (n_epoch // 10) == 0 or epoch == n_epoch - 1:
                tqdm.write(f"Epoch {epoch + 1: 02d}/{n_epoch} | Training CVaR: {epoch_loss/n_batch: .6f} | Validation CVaR: {epoch_loss/n_batch: .6f} [Best: {best_loss_Va: .6f}] | LR: {cur_lr: .3g}")
        
        model.train()

def Load_CSV_Tensor(path, n_var, device = "cpu", dtype = torch.float64, standardised = False, standardised_ind = None):
    df = pd.read_csv(path)
    data_np = df.to_numpy(dtype = np.float64)
    n_sim, n_col = data_np.shape
    if n_col % n_var != 0:
        raise ValueError(f"Number of columns is not multiple of {n_var}.")
    n_step = n_col // n_var
    data_tensor = torch.from_numpy(data_np.reshape(n_sim, n_step, n_var)).to(device = device, dtype = dtype)

    if standardised:
        denominator = data_tensor[0,0,0].item()
        if standardised_ind is not None:
            for index in standardised_ind:
                data_tensor[:,:,index] = data_tensor[:,:,index] /  denominator * 100
    
    return data_tensor

def main():
    '''TESTING CASES'''
    LIAB_CASE = ["call", "digital_call", "basket"]
    LIAB_K_CASE = [100, 80, 120]
    HEDGE_TYPE = ["underlying", "call", "call", "call", "digital_call", "digital_call", "digital_call"]
    HEDGE_K = [100, 95, 100, 105, 95, 100, 105]
    HEDGE_T = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    HEDGE_CASE = [[1], #only underlying
             [0, 1, 2, 3], #only underlying and calls
             [0, 2, 5], #only underlying, ATM call and ATM digital call
             [1, 2, 3, 4, 5, 6, 7]] #"complete market"
    FREQ_CASE = [90, 45, 12, 3]
    TC_CASE = [0.0, 0.00001, 0.0001, 0.]
    '''SET UP'''
    # Market environment
    S0_sim = 100
    r = 0.051609817761459786
    q = 0.0168651076739148
    mu = 0.1301391936874255
    tc = 0.001
    # Hedging Instrument
    HH = 1
    hedge_type = [HEDGE_TYPE[ii] for ii in HEDGE_CASE[HH]]
    hedge_K = [HEDGE_K[ii] for ii in HEDGE_CASE[HH]]
    hedge_T = [HEDGE_T[ii] for ii in HEDGE_CASE[HH]]
    n_hedge = len(hedge_type)
    target_greeks = torch.zeros(min(4, n_hedge), dtype = torch.float64)
    max_pos, min_pos = 10, -10
    max_pos = torch.ones(n_hedge, dtype = torch.float64) * max_pos
    min_pos = torch.ones(n_hedge, dtype = torch.float64) * min_pos
    # Risk Exposure/Liability
    liab_type = "call"
    liab_K = S0_sim
    liab_T = 1/4
    basket_memeber = ["call", "call", "call", "put", "put", "put", "digital_call", "digital_call", "digital_call"]
    basket_K = [90, 100, 110, 90, 100, 110, 90, 100, 110]
    basket_pos = [1, -2, 3, -4, 5, -6, 7, -8, 9]
    # Simulation Parameters
    T_sim = liab_T
    n_step = 90
    n_total_sim = 1000
    n_sim = int(round(n_total_sim * 0.9))
    n_test = int(round(n_total_sim * 0.1))
    # Deep Hedging Parameters
    input_dim = 3
    hidden_dim = 64
    output_dim = n_hedge # 3
    v0 = 0.01573939062654972
    alpha = 0.05
    # Q Learner Parameters
    trade_unit = 0.1
    levels = [-2.0, -1.0, 0.0, 1.0, 2.0]   
    # Training Setting
    hedge_freq = n_step
    n_epoch = 1000
    lr = 1e-3
    batch_size = 512
    verbose, early_terminate, min_improve, patience = True, False, 1e-6, 10
    new_data, save_data, data_from_CSV = False, False, True
    BS_pricer = False
    train_deep_hedge = True
    save_deep_hedging, model_name = False, f"{liab_type}_{liab_K}_{liab_T}"
    train_greek_hedge = True
    save_greek_hedge = False
    # Testing Setting
    alpha_test = [0.01, 0.05, 0.10]
    same_liab = True
    liab2_type, liab2_K, liab2_T = liab_type, liab_K + 10, liab_T
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''DATA GENERATION'''
    if new_data: #True => Generate    
        heston_model = HestonModel.HestonModel(r, q)
        heston_model.load_state_dict(torch.load(f"{current_dir}/calibrated_heston2v.pt"))
        if BS_pricer:
            pricer = HestonModel.BSModel(r, q)
        else:
            pricer = None
        print("Data generation is started")
        t1 = time.time()
        X_train, price_hedge_train = HestonModel.StochasticData_Batch2(heston_model, S0_sim, T_sim, mu, n_step, hedge_type, hedge_K, hedge_T = hedge_T, n_sim = n_sim, batch_size = 128, pricer = pricer)
        X_test, price_hedge_test = HestonModel.StochasticData_Batch2(heston_model, S0_sim, T_sim, mu, n_step, hedge_type, hedge_K, hedge_T = hedge_T, n_sim = n_test, batch_size = 128, pricer = pricer)
        print(f"Data generation is complete with total running time: {time.time()-t1:.0f}s.")
        if save_data:
            torch.save(X_train, f"{current_dir}/X_train.pt")
            torch.save(price_hedge_train, f"{current_dir}/PH_train.pt")
            torch.save(X_test, f"{current_dir}/X_test.pt")
            torch.save(price_hedge_test, f"{current_dir}/PH_test.pt")
    else:
        if data_from_CSV:
            X_source = Load_CSV_Tensor(f"{current_dir}/sample data/project_data_3M/X_combined.csv", input_dim, standardised = True, standardised_ind = [0])
            price_hedge_source = Load_CSV_Tensor(f"{current_dir}/sample data/project_data_3M/PH_combined.csv", 7, standardised = True, standardised_ind = [0, 1, 2, 3])
            #print(X_source.shape)
            index_rand = torch.randperm(X_source.shape[0], device = device)
            X_train = X_source[index_rand[0:n_sim]]
            price_hedge_train = price_hedge_source[index_rand[0:n_sim]][:,:,HEDGE_CASE[HH]]
            X_test = X_source[index_rand[n_sim:n_total_sim]]
            price_hedge_test = price_hedge_source[index_rand[n_sim:n_total_sim]][:,:,HEDGE_CASE[HH]]
            del X_source
            del price_hedge_source

        else:
            X_train = torch.load(f"{current_dir}/X_train.pt")
            price_hedge_train = torch.load(f"{current_dir}/PH_train.pt")
            X_test = torch.load(f"{current_dir}/X_test.pt")
            price_hedge_test = torch.load(f"{current_dir}/PH_test.pt")
    # Adjustment for hedging frequency
    if hedge_freq != n_step:
        freq_index = [int(round(n_step / hedge_freq * tt)) for tt in range(hedge_freq)]
        X_train = X_train[:, freq_index, :]
        price_hedge_train = price_hedge_train[:, freq_index, :]
        X_test = X_test[:, freq_index, :]
        price_hedge_test = price_hedge_test[:, freq_index, :]
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''NEURAL NETWORK BASED DEEP HEDGING'''
    # Set Up
    NN_model = DeepHedging_NN(input_dim, hidden_dim, output_dim)
    liability = Derivative(liab_type, S0_sim, r, q, v0, T_sim, liab_K, notional = 100, basket_member=basket_memeber, basket_K=basket_K, basket_position=basket_pos)
    #liability = Derivative("call", S0_sim, r, q, v0, T_sim, liab_K, notional = 100, B = None)
    #print(liability.Price())
    if train_deep_hedge:
        Training(NN_model, liability, CVar_Loss, X_train, price_hedge_train, T_sim, r, tc, alpha, n_epoch = n_epoch, lr = lr, batch_size = batch_size, clip = 1.0, verbose = verbose, early_terminate=early_terminate, min_improve=min_improve, patience=patience)
        if save_deep_hedging:
            torch.save(NN_model.state_dict(), f"{current_dir}/{model_name}_NN.pt")
    else:
        NN_model.load_state_dict(torch.load(f"{current_dir}/{model_name}_NN.pt"))
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    '''Q LEARNER'''
    #Action_space = QLearner.Discrete_Action_Space(n_hedge, levels)   # [A, M]
    #cfg = QLearner.HedgingEnvCfg(pos_cap = 50.0, trade_unit = [trade_unit] * n_hedge, tc = tc, include_interest=True)
    #env = QLearner.HedgingEnv(X_train, price_hedge_train, liability, Action_space, cfg)
    #qnet = QLearner.dqn_train(env, n_actions=Action_space.shape[0], n_epoch = n_epoch, batch_size=512, gamma=1.0, lr=1e-3)

    if False:
        dnfkdsnfkdk = 0
        #hedging_instruments = [Derivative(hedge_type[hh], S0_sim, r, q, v0, T_sim, hedge_K[hh]) for hh in range(n_hedge)]
        #heston_model = HestonModel.HestonModel(r, q)
        #heston_model.load_state_dict(torch.load(f"{current_dir}/calibrated_heston2v.pt"))
        #HM = QLearnerHedging.Heston_Model_Q(r, q, heston_model.kappa.item(), heston_model.theta.item(), heston_model.sigma.item(), heston_model.v0.item(), heston_model.rho.item(), device, dtype = torch.float64)
        #levels = [-2,-1,0,1,2]
        #action_space = QLearnerHedging.Action_Space(n_hedge, levels)

        #env_Q = QLearnerHedging.QLearner_Env(S0_sim, v0, T_sim, mu, n_step, hedging_instruments, liability, HM, action_space, trade_unit, tc, device = device)
        #n_action = action_space.shape[0]
        #QL_model = QLearnerHedging.DQN_Train(env_Q, n_action, alpha, 64, n_epoch = n_epoch)
        #QLearnerHedging.rollout_PnL
        #####d_hedge_test_QL = QLearnerHedging.rollout_all_trades(env_Q, QL_model, n_action, 1024)

        ##### Rollout on test set, compute CVaR
        #####d_hedge_test_QL = QLearner.policy_rollout_deltas(qnet, X_test, price_hedge_test, liability, Action_space, cfg)  # [N_te, T, M]
        #pnl_QL, tc_QL = QLearnerHedging.rollout_PnL(env_Q, QL_model, n_action, 1024)
        #pnl = pnl_QL.detach().cpu().numpy()
        
        #plt.hist(pnl, bins=50, density=True, alpha=0.7, color="blue")
        #plt.xlabel("PNL")
        #plt.ylabel("Frequency")
        #plt.title("Histogram of PNL")
        #plt.show()
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    '''GREEKS BASED HEDGING'''
    if train_greek_hedge:
        d_hedge_Greek = GreekHedging(X_test, liability, target_greeks, hedge_type, hedge_K, hedge_T = None, max_pos = max_pos, min_pos = min_pos)
        if save_greek_hedge:
            torch.save(d_hedge_Greek, f"{current_dir}/{model_name}_Greek.pt")
    else:
        d_hedge_Greek = torch.load(f"{current_dir}/{model_name}_Greek.pt")
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''EVALUATION/COMPARISON'''
    d_hedge_test_NN = NN_model(X_test)
    
    if not same_liab:
        liability = Derivative(liab2_type, S0_sim, r, q, v0, liab2_T, liab2_K, notional = 100, B = None)
    
    pnl_NN, pnl_NH, tc_NN = PnL(d_hedge_test_NN, price_hedge_test, liability, transaction_cost=tc, TC_output = True)
    pnl_Greek, _, tc_Greek = PnL(d_hedge_Greek, price_hedge_test, liability, transaction_cost=tc, TC_output = True)
    tc_NN = tc_NN.mean().item()
    tc_Greek = tc_Greek.mean().item()
    print(f"=={liab_type} with T/C {tc}, hedging frequency {hedge_freq}, alpha {alpha}==")
    print(f"Total transaction cost | Neural network: {tc_NN} | Greeks: {tc_Greek} | QLearner: ? | Difference: {tc_NN - tc_Greek}")
    
    risk_NN = []
    risk_Greek = []
    risk_QL = []
    risk_NH = []


    pnl_NN_np = pnl_NN.detach().numpy()
    pnl_Greek_np = pnl_Greek.detach().numpy()
    bins = np.linspace(
    min(pnl_Greek_np.min(), pnl_NN_np.min()), 
    max(pnl_Greek_np.max(), pnl_NN_np.max()), 
    50   # number of bins
    )
    plt.figure(figsize=(6,3))
    plt.hist(pnl_Greek_np, bins=bins, alpha=0.5, label="Greek hedging", color="red", edgecolor="white")
    plt.hist(pnl_NN_np, bins=bins, alpha=0.5, label="Neural network", color="blue", edgecolor="white")
    #plt.title("Distribution of %Risk Reduction")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for aa in alpha_test:
        CVaR_NN = CVaR_PnL(pnl_NN, aa)
        CVaR_Greek = CVaR_PnL(pnl_Greek, aa)
        #CVaR_QL = CVaR_PnL(pnl_QL, aa)
        CVaR_NH = CVaR_PnL(pnl_NH, aa)
        risk_NN.append(CVaR_NN.item())
        risk_Greek.append(CVaR_Greek.item())
        #risk_QL.append(CVaR_QL.item())
        risk_NH.append(CVaR_NH.item())
        improve_NN = ((CVaR_NH - CVaR_NN)/torch.abs(CVaR_NH)).item()
        improve_Greek = ((CVaR_NH - CVaR_Greek)/torch.abs(CVaR_NH)).item()
        #improve_QL = ((CVaR_NH - CVaR_QL)/torch.abs(CVaR_NH)).item()
        CVaR_QL = torch.tensor([0.00], dtype=torch.float64)
        improve_QL = 0
        print(f"{(1-aa)*100}% CVaR | Neural network: {CVaR_NN.item():.4f}({improve_NN*100:.2f}%) | QLearner: {CVaR_QL.item():.4f}({improve_QL*100:.2f}%) | Greek: {CVaR_Greek.item():.4f}({improve_Greek*100:.2f}%) | No action: {CVaR_NH.item():.4f}")
    
    del X_train, price_hedge_train, X_test, price_hedge_test, NN_model, d_hedge_test_NN, d_hedge_Greek
    gc.collect()
    
    return tc_NN, pnl_NN, risk_NN, tc_Greek, pnl_Greek, risk_Greek, pnl_NH, risk_NH
    
main()
