import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquad
from torchquad import Trapezoid
from torchquad import GaussLegendre
from loguru import logger
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
import tqdm
import matplotlib.pyplot as plt
import yfinance as yf
logger.remove()
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

### Black Scholes Model ###
class BSModel(nn.Module):
    def __init__(self, r, q):
        super().__init__()
        self.r = r
        self.q = q

    def Option_Price(self, S0, K, T, vt, type = "call", integrator = None):
        '''
        ===Input Definition===
            S0 as a vector of spot price of underlying instrument S_t for the option at time t
            K as a vector of strike price K of each options
            T (tau) as a vector of time to maturity tau = T - t of each options
            *** Optional ***
            type as a list of type of option to price: call, put, digital_call, digital_put or all (gives price of all 4 options) <default = "call">
        ===Output Definition===
            price as a vector of price of specific option or a matrix of prices of call, put, digital_call and digital_put
        '''
        # input validation
        if not torch.is_tensor(S0):
            S0 = torch.tensor([S0], dtype = torch.float64).flatten()
        if not torch.is_tensor(K):
            K = torch.tensor([K], dtype = torch.float64).flatten()
        if not torch.is_tensor(T):
            T = torch.tensor([T], dtype = torch.float64).flatten()
        if not torch.is_tensor(vt):
            T = torch.tensor([vt], dtype = torch.float64).flatten()
        
        n = K.shape[0]
        
        d1 = (torch.log(S0 / K) + (self.r - self.q + vt / 2) * T) / torch.sqrt(vt * T)
        d2 = d1 - torch.sqrt(vt * T)
        N1 = 0.5 * (1 + torch.erf(d1 / 2 ** 0.5))
        N2 = 0.5 * (1 + torch.erf(d2 / 2 ** 0.5))

        if type == "all":
            price = torch.empty((n,4), dtype = torch.float64)
            for ii in range(n):
                if T[ii] != 0:
                    price[ii, 0] = S0[ii] * torch.exp(-self.q * T[ii]) * N1[ii] - K[ii] * torch.exp(-self.r * T[ii]) * N2[ii]
                    price[ii, 1] = - S0[ii] * torch.exp(-self.q * T[ii]) * (1 - N1[ii]) + K[ii] * torch.exp(-self.r * T[ii]) * (1 - N2[ii])
                    price[ii, 2] = torch.exp(-self.r * T[ii]) * N2[ii]
                    price[ii, 3] = torch.exp(-self.r * T[ii]) * (1 - N2[ii])
                else:
                    price[ii, 0] = torch.clamp(S0[ii] - K[ii], min = 0)
                    price[ii, 1] = torch.clamp(K[ii] - S0[ii], min = 0)
                    price[ii, 2] = (S0[ii] > K[ii])
                    price[ii, 3] = (S0[ii] <= K[ii])
            # otherwise, prices of specific option for each set of S0, K and T are calculated
        else:    
            price = torch.empty(n, dtype = torch.float64)
            for ii in range(n):
                if T[ii] != 0:
                    if type[ii] == "call":
                        price[ii] = S0[ii] * torch.exp(-self.q * T[ii]) * N1[ii] - K[ii] * torch.exp(-self.r * T[ii]) * N2[ii]
                    elif type[ii] == "put":
                        price[ii] = - S0[ii] * torch.exp(-self.q * T[ii]) * (1 - N1[ii]) + K[ii] * torch.exp(-self.r * T[ii]) * (1 - N2[ii])
                    elif type[ii] == "digital_call":
                        price[ii] = torch.exp(-self.r * T[ii]) * N2[ii]
                    elif type[ii] == "digital_put":
                        price[ii] = torch.exp(-self.r * T[ii]) * (1 - N2[ii])
                    elif type[ii] == "underlying":
                        price[ii] = S0[ii]
                    else:
                        price[ii] = 0
                else:  
                    if type[ii] == "call":
                        price[ii] = torch.clamp(S0[ii] - K[ii], min = 0)
                    elif type[ii] == "put":
                        price[ii] = torch.clamp(K[ii] - S0[ii], min = 0)
                    elif type[ii] == "digital_call":
                        price[ii] = (S0[ii] >= K[ii])
                    elif type[ii] == "digital_put":
                        price[ii] = (S0[ii] <= K[ii])
                    elif type[ii] == "underlying":
                        price[ii] = S0[ii]
                    else:
                        price[ii] = 0                
        return price
    
    def Greeks(self, S0, K, T, vt, type = "call"):
        '''
        ===Input Definition===
            S0 as a vector of spot price of underlying instrument S_t for the option at time t
            K as a vector of strike price K of each options
            T (tau) as a vector of time to maturity tau = T - t of each options
            *** Optional ***
            type as a list of type of option to price: call, put, digital_call, digital_put or all (gives price of all 4 options) <default = "call">
        ===Output Definition===
            greeks as a matrix of greeks of options: delta, gamma, 
        '''
        # input validation
        if not torch.is_tensor(S0):
            S0 = torch.tensor([S0], dtype = torch.float64).flatten()
        if not torch.is_tensor(K):
            K = torch.tensor([K], dtype = torch.float64).flatten()
        if not torch.is_tensor(T):
            T = torch.tensor([T], dtype = torch.float64).flatten()
        if not torch.is_tensor(vt):
            T = torch.tensor([vt], dtype = torch.float64).flatten()        
        
        n = K.shape[0]
        greeks = torch.empty((n,4), dtype = torch.float64)

        d1 = (torch.log(S0 / K) + (self.r - self.q + vt / 2) * T) / torch.sqrt(vt * T)
        d2 = d1 - torch.sqrt(vt * T)
        N1 = 0.5 * (1 + torch.erf(d1 / 2 ** 0.5))
        N2 = 0.5 * (1 + torch.erf(d2 / 2 ** 0.5))
        n1 = torch.exp(-0.5 * d1 ** 2) / torch.sqrt(2 * torch.pi)
        n2 = torch.exp(-0.5 * d2 ** 2) / torch.sqrt(2 * torch.pi)

        if type == "all":
            price = torch.empty((n,3), dtype = torch.float64)
            for ii in range(n):
                
                price[ii, 0] = S0[ii] * torch.exp(-self.q * T[ii]) * N1[ii] - K[ii] * torch.exp(-self.r * T[ii]) * N2[ii]
                price[ii, 1] = - S0[ii] * torch.exp(-self.q * T[ii]) * (1 - N1[ii]) + K[ii] * torch.exp(-self.r * T[ii]) * (1 - N2[ii])
                price[ii, 2] = torch.exp(-self.r * T[ii]) * N2[ii]
                price[ii, 3] = torch.exp(-self.r * T[ii]) * (1 - N2[ii])
        # otherwise, prices of specific option for each set of S0, K and T are calculated
        else:    
            price = torch.empty(n, dtype = torch.float64)
            for ii in range(n):
                if type[ii] == "call":
                    price[ii] = S0[ii] * torch.exp(-self.q * T[ii]) * N1[ii] - K[ii] * torch.exp(-self.r * T[ii]) * N2[ii]
                elif type[ii] == "put":
                    price[ii] = - S0[ii] * torch.exp(-self.q * T[ii]) * (1 - N1[ii]) + K[ii] * torch.exp(-self.r * T[ii]) * (1 - N2[ii])
                elif type[ii] == "digital_call":
                    price[ii] = torch.exp(-self.r * T[ii]) * N2[ii]
                elif type[ii] == "digital_put":
                    price[ii] = torch.exp(-self.r * T[ii]) * (1 - N2[ii])
                else:
                    price[ii] = 0
        return greeks

### Heston Model ###
class HestonModel(nn.Module):
    def __init__(self, r, q):
        super().__init__()
        self.r = r
        self.q = q
        # learnable parameters: kappa, theta, sigma, v0 and rho
        self.kappa = nn.Parameter(torch.empty(1).uniform_(0.5, 1.5))
        self.theta = nn.Parameter(torch.empty(1).uniform_(0, 0.1))
        self.sigma = nn.Parameter(torch.empty(1).uniform_(0, 0.1))
        self.v0 = nn.Parameter(torch.empty(1).uniform_(0, 0.1))
        self.rho = nn.Parameter(torch.empty(1).uniform_(-1, 1))

    def char_func(self, z, S0, T, vt = None):
        '''
        ===Input Definition===
            z as a variable for integration
            S0 as a vector of spot price of underlying instrument for the option at time t
            T (tau) as a vector of time to maturity tau = T - t of each options
        ===Output Definition===
            f1 and f2 as two vectors of characteristic function 1 and 2
        '''
        # input validation
        if not torch.is_tensor(S0):
            S0 = torch.tensor([S0], dtype = torch.float64).flatten()
        if not torch.is_tensor(T):
            T = torch.tensor([T], dtype = torch.float64).flatten()
        if S0.view(-1).shape != T.view(-1).shape:
            raise ValueError(f"S0 has shape {list(S0.shape)} and T has shape {list(T.shape)}. S0 and T should be vectors of same size.")
        if vt is None:
            vt = self.v0
        else:
            if not torch.is_tensor(vt):
                vt = torch.tensor([vt], dtype = torch.float64).flatten()
        # define the parameters for characteristic functions
        i = torch.tensor(1j, dtype = torch.cdouble)
        z = torch.as_tensor(z, dtype = torch.float64).to(torch.cdouble)
        S0 = S0.to(torch.cdouble)
        T = T.to(torch.cdouble)

        kappa = self.kappa.to(torch.float64)
        theta = self.theta.to(torch.float64)
        sigma = self.sigma.to(torch.float64)
        rho   = self.rho.to(torch.float64)
        r     = torch.as_tensor(self.r, dtype=torch.float64)
        q     = torch.as_tensor(self.q, dtype=torch.float64)

        b1 = kappa - rho * sigma
        b2 = kappa
        u1 = torch.tensor(0.5)
        u2 = torch.tensor(-0.5)
        
        # calculation of characteristic functions
        def _CD_(u, b):
            d = torch.sqrt((rho * sigma * z * i - b) ** 2 - (sigma ** 2) * (2 * u * i * z - z ** 2))
            g = (b - rho * sigma * z * i + d) / (b - rho * sigma * z * i - d)
            exp_dT = torch.exp(d * T)
            C = (r - q) * z * T * i + (kappa *  theta / (sigma ** 2)) *((b - rho * sigma * z * i + d) * T - 2.0 * torch.log((1 - g * exp_dT + 1e-8) / (1 - g + 1e-8)))
            D = ((b - rho * sigma * z * i + d) / (sigma ** 2)) * (1 - exp_dT) / (1 - g * exp_dT + 1e-8)
            return C, D
        C1, D1 = _CD_(u1, b1)
        C2, D2 = _CD_(u2, b2)
        
        f1 = torch.exp(C1  + D1 * vt.to(torch.cdouble) + z * i * torch.log(S0))
        f2 = torch.exp(C2  + D2 * vt.to(torch.cdouble) + z * i * torch.log(S0))

        return f1, f2

    def Integrand(self, z, S0, K, T, vt = None):
        '''
        ===Input Definition===
            z as a variable for integration
            S0 as a vector of spot price of underlying instrument S_t for the option at time t
            K as a vector of strike price K of each options
            T (tau) as a vector of time to maturity tau = T - t of each options
        ===Output Definition===
            intergrand1 and intergrand2 as two vectors of integrand 1 and 2 for integration and hence calculation of P1 and P2
        '''
        # input validation
        if not torch.is_tensor(S0):
            S0 = torch.tensor([S0], dtype = torch.float64).flatten()
        if not torch.is_tensor(K):
            K = torch.tensor([K], dtype = torch.float64).flatten()
        if not torch.is_tensor(T):
            T = torch.tensor([T], dtype = torch.float64).flatten()
        if (S0.view(-1).shape != T.view(-1).shape) or (S0.view(-1).shape != K.view(-1).shape):
            raise ValueError(f"S0 has shape {list(S0.shape)}, K has shape {list(K.shape)} and T has shape {list(T.shape)}. S0, K and T should be vectors of same size.")

        # definition the inputs
        i = 1j
        f1, f2 = self.char_func(z, S0, T, vt)
        integrand1 = (torch.exp(-z * i * torch.log(K)) * f1 / (z * i)).real
        integrand2 = (torch.exp(-z * i * torch.log(K)) * f2 / (z * i)).real
        
        return integrand1, integrand2

    def Prob(self, S0, K, T, vt = None, domain_max = 200, integrator = None):
        '''
        ===Input Definition===
            S0 as a vector of spot price of underlying instrument S_t for the option at time t
            K as a vector of strike price K of each options
            T (tau) as a vector of time to maturity tau = T - t of each options
            *** Optional ***
            domain_max as the upper bound of the integration <default = 1000>
        ===Output Definition===
            P1 and P2 as two vectors of P1 and P2 for option pricing
        '''
        # input validation
        if not torch.is_tensor(S0):
            S0 = torch.tensor([S0], dtype = torch.float64).flatten()
        if not torch.is_tensor(K):
            K = torch.tensor([K], dtype = torch.float64).flatten()
        if not torch.is_tensor(T):
            T = torch.tensor([T], dtype = torch.float64).flatten()
        if (S0.view(-1).shape != T.view(-1).shape) or (S0.view(-1).shape != K.view(-1).shape):
            raise ValueError(f"S0 has shape {list(S0.shape)}, K has shape {list(K.shape)} and T has shape {list(T.shape)}. S0, K and T should be vectors of same size.")
        
        # integration
        if integrator is None:
            integrator = GaussLegendre() #intergrator = Trapezoid()
        integral1 = integrator.integrate(fn = lambda z: self.Integrand(z, S0, K, T, vt = vt)[0], dim = 1, N = 256, integration_domain = [[1e-8, domain_max]])
        integral2 = integrator.integrate(fn = lambda z: self.Integrand(z, S0, K, T, vt = vt)[1], dim = 1, N = 256, integration_domain = [[1e-8, domain_max]])
        
        # calculation of P1 and P2
        P1 = 0.5 + integral1 / np.pi
        P2 = 0.5 + integral2 / np.pi
        if K.shape[0] == 1:
            P1 = torch.tensor([P1], dtype = torch.float64)
            P2 = torch.tensor([P2], dtype = torch.float64)
        
        return P1, P2

    def Option_Price(self, S0, K, T, type = "call", vt = None, integrator = None):
        '''
        ===Input Definition===
            S0 as a vector of spot price of underlying instrument S_t for the option at time t
            K as a vector of strike price K of each options
            T (tau) as a vector of time to maturity tau = T - t of each options
            *** Optional ***
            type as a list of type of option to price: call, put, digital_call, digital_put or all (gives price of all 4 options) <default = "call">
        ===Output Definition===
            price as a vector of price of specific option or a matrix of prices of call, put, digital_call and digital_put
        '''
        # input validation
        if not torch.is_tensor(S0):
            S0 = torch.tensor([S0], dtype = torch.float64).flatten()
        if not torch.is_tensor(K):
            K = torch.tensor([K], dtype = torch.float64).flatten()
        if not torch.is_tensor(T):
            T = torch.tensor([T], dtype = torch.float64).flatten()
        if vt is not None:
            if not torch.is_tensor(vt):
                vt = torch.tensor([vt], dtype = torch.float64).flatten()
        # calculate P1 and P2  
        P1, P2 = self.Prob(S0, K, T, vt, integrator = integrator)
        n = K.shape[0]
        # if type = "all", prices of all options for each set of S0, K and T are calculated
        if type == "all":
            price = torch.empty((n,4), dtype = torch.float64)
            for ii in range(n):
                if T[ii] != 0:
                    price[ii, 0] = S0[ii] * torch.exp(-self.q * T[ii]) * P1[ii] - K[ii] * torch.exp(-self.r * T[ii]) * P2[ii]
                    price[ii, 1] = - S0[ii] * torch.exp(-self.q * T[ii]) * (1 - P1[ii]) + K[ii] * torch.exp(-self.r * T[ii]) * (1 - P2[ii])
                    price[ii, 2] = torch.exp(-self.r * T[ii]) * P2[ii]
                    price[ii, 3] = torch.exp(-self.r * T[ii]) * (1 - P2[ii])
                else:
                    price[ii, 0] = torch.clamp(S0[ii] - K[ii], min = 0)
                    price[ii, 1] = torch.clamp(K[ii] - S0[ii], min = 0)
                    price[ii, 2] = (S0[ii] > K[ii])
                    price[ii, 3] = (S0[ii] <= K[ii])
            # otherwise, prices of specific option for each set of S0, K and T are calculated
        else:    
            price = torch.empty(n, dtype = torch.float64)
            for ii in range(n):
                if T[ii] != 0:
                    if type[ii] == "call":
                        price[ii] = S0[ii] * torch.exp(-self.q * T[ii]) * P1[ii] - K[ii] * torch.exp(-self.r * T[ii]) * P2[ii]
                    elif type[ii] == "put":
                        price[ii] = - S0[ii] * torch.exp(-self.q * T[ii]) * (1 - P1[ii]) + K[ii] * torch.exp(-self.r * T[ii]) * (1 - P2[ii])
                    elif type[ii] == "digital_call":
                        price[ii] = torch.exp(-self.r * T[ii]) * P2[ii]
                    elif type[ii] == "digital_put":
                        price[ii] = torch.exp(-self.r * T[ii]) * (1 - P2[ii])
                    elif type[ii] == "underlying":
                        price[ii] = S0[ii]
                    else:
                        price[ii] = 0
                else:
                    if type[ii] == "call":
                        price[ii] = torch.clamp(S0[ii] - K[ii], min = 0)
                    elif type[ii] == "put":
                        price[ii] = torch.clamp(K[ii] - S0[ii], min = 0)
                    elif type[ii] == "digital_call":
                        price[ii] = (S0[ii] >= K[ii])
                    elif type[ii] == "digital_put":
                        price[ii] = (S0[ii] <= K[ii])
                    elif type[ii] == "underlying":
                        price[ii] = S0[ii]
                    else:
                        price[ii] = 0
        return price
    
    def DataSim(self, S0, T, mu, n_step, n_sim = 10000):
        '''
        ===Input Definition===
            S0 as a scalar, the initial price of the simulation
            T as a scalar, the time horizon to simulate
            mu as a scalar, the average rate of return of the underlying instrument
            n_step as a scalar, the number of steps to simulate
            n_sim as a scalar, the number of simulated paths <default = 10000>
        ===Output Definition===
            S_paths as a matrix [n_sim x (n_step + 1)] of simulated prices of underlying instrument
            v_paths as a matrix [n_sim x (n_step + 1)] of simulated variance of underlying instrument
        '''
        # simulation setup
        dt = T/n_step
        sqrt_dt = torch.sqrt(torch.tensor(dt, dtype = torch.float64))
        S_paths = torch.empty((n_sim, n_step + 1), dtype = torch.float64)
        v_paths = torch.empty((n_sim, n_step + 1), dtype = torch.float64)
        S_paths[:,0] = S0
        v_paths[:,0] = self.v0
        
        # simulation
        for tt in range(1, n_step + 1):
            # generate two IID N(0,1)
            Z1 = torch.randn(n_sim, dtype = torch.float64)
            Z2 = torch.randn(n_sim, dtype = torch.float64)
            
            # correlated standard Brownian motions
            dW_S = Z1 * sqrt_dt
            dW_v = (self.rho * Z1 + torch.sqrt(1 - self.rho ** 2) * Z2) * sqrt_dt
            
            # calculate the paths of S and v
            log_S_prev = torch.log(S_paths[:, tt - 1])
            sqrt_v_prev = torch.sqrt(v_paths[:, tt - 1])
            S_paths[:, tt] = torch.exp(log_S_prev + (mu - self.q - 0.5 * v_paths[:, tt - 1]) * dt + sqrt_v_prev * dW_S)
            #S_paths[:, tt] = S_paths[:, tt - 1] * (1 + (mu - self.q) * dt + sqrt_v_prev * dW_S)
            v_paths[:, tt] = torch.clamp(v_paths[:, tt - 1] + self.kappa * (self.theta - v_paths[:, tt - 1]) * dt + self.sigma * sqrt_v_prev * dW_v, min = 1e-8)

        return S_paths, v_paths

    def SetParam(self, kappa = None, theta = None, sigma = None, v0 = None, rho = None):
        '''
        ===Input Definition===
        kappa, theta, sigma, v0, rho as scalars to set the parameters to certain value if the input is not None <default = None>
        '''
        # update the parameter(s) with the input value(s) if any
        with torch.no_grad():
            if kappa is not None:
                self.kappa.copy_(torch.tensor([kappa], dtype = torch.float64))
            if theta is not None:
                self.theta.copy_(torch.tensor([theta], dtype = torch.float64))
            if sigma is not None:
                self.sigma.copy_(torch.tensor([sigma], dtype = torch.float64))
            if v0 is not None:
                self.v0.copy_(torch.tensor([v0], dtype = torch.float64))
            if rho is not None:
                self.rho.copy_(torch.tensor([rho], dtype = torch.float64))

    def ClampParam(self):
        with torch.no_grad():
            self.kappa.clamp_(1e-6, 50.0)
            self.theta.clamp_(1e-8, 5.0)
            self.sigma.clamp_(1e-6, 5.0)
            self.v0.clamp_(1e-6, 5.0)
            self.rho.clamp_(-1, 1)

    def Loss(self, opt_data, integrator = None, diff = False):
        '''
        ===Input Definition===
        opt_data as a tensor of n_opt option: type (1 = call, 0 = put), time to maturity, spot price, strike price, and option price 
        '''
        type = opt_data[:, 0]
        type = ["call" if ty.item() == 1.0 else "put" for ty in type]
        T = opt_data[:, 1] 
        S0 = opt_data[:, 2]
        K = opt_data[:, 3]
        market_price = opt_data[:, 4]
        model_price = self.Option_Price(S0, K, T, type, integrator = integrator)
        model_price = torch.nan_to_num(model_price, nan = 0.0, posinf = 1e12, neginf = -1e12)
        loss = torch.sum((market_price - model_price) ** 2)
        return loss
    
### Generate Data ###
def StochasticData(heston_model, S0, T, mu, n_step, K_hedge, n_sim = 10000, pricer = None):
    '''
    ===Input Definition===
    heston_model as a class objected, a calibrated Heston model
    S0 as a scalar, spot price of the underlying instrument at t=0
    T as a scalar, time horizon of simulation
    n_step as a scalar, number of step in the simulation
    K_hedge (n_hedge - 1) as a vector of the strike prices of each hedging instruments
    n_sim as a scalar, number of simulation
    ===Output Definition===
    X (n_sim, n_step + 1, 3) as a tensor of data that feeds into the neural network: St, vt, T-t
    price_hedge (n_sim, n_step + 1, n_hedge) as a tensor of prices of hedging instrument at different time step 
    '''
    # setup
    dt = T/n_step
    
    # data simulation
    St, vt = heston_model.DataSim(S0, T, mu, n_step, n_sim)
    tau = T - torch.arange(n_step, dtype = torch.float64) * dt
    tau = tau.unsqueeze(0).expand(n_sim, -1)
    # generate X
    X = torch.stack((St[:, :-1], vt[:, :-1], tau), dim = -1)

    # generate price_hedge
    if pricer is None:
        pricer = heston_model
    price_hedge = torch.zeros(n_sim, n_step + 1, 3, dtype = torch.float64)
    for tt in range(n_step):
        print(tt)
        price_hedge[:, tt, 0] = St[:, tt]
        price_hedge[:, tt, 1] = pricer.Option_Price(St[:, tt], torch.ones(n_sim, dtype = torch.float64) * K_hedge[0], tau[:, tt], vt = vt[:, tt], type = ["call"] * n_sim)
        price_hedge[:, tt, 2] = pricer.Option_Price(St[:, tt], torch.ones(n_sim, dtype = torch.float64) * K_hedge[1], tau[:, tt], vt = vt[:, tt], type = ["digital_call"] * n_sim)
    price_hedge[:, n_step, 0] = St[:, -1]
    price_hedge[:, n_step, 1] = torch.clamp(St[:, -1] - K_hedge[0], 0)
    price_hedge[:, n_step, 2] = 1 * (St[:, -1] >= K_hedge[1])

    return X, price_hedge

def StochasticData_Batch(heston_model, S0, T, mu, n_step, K_hedge, n_sim = 10000, batch_size = 128, pricer = None, K = None):
    '''
    ===Input Definition===
    heston_model as a class objected, a calibrated Heston model
    S0 as a scalar, spot price of the underlying instrument at t=0
    T as a scalar, time horizon of simulation
    n_step as a scalar, number of step in the simulation
    K_hedge (n_hedge - 1) as a vector of the strike prices of each hedging instruments
    n_sim as a scalar, number of simulation
    batch_size as a scalar, size of batch
    pricer as a class object that calculate the pricing of hedging instruments at each step <default = None -> heston_model>
    K as a strick price of the liability <default = None -> Not included>
    ===Output Definition===
    X (n_sim, n_step + 1, 3) as a tensor of data that feeds into the neural network: St, vt, T-t
    price_hedge (n_sim, n_step + 1, n_hedge) as a tensor of prices of hedging instrument at different time step 
    '''
    # Data validation
    if torch.tensor(K_hedge).flatten().shape[0] == 1:
        single_K = True
    else:
        single_K = False
    # setup
    dt = T/n_step
    tau = T - torch.arange(n_step, dtype = torch.float64) * dt
    X_chunks = []
    price_hedge_chunks = []
    integrator = GaussLegendre()
    # data simulation
    with torch.no_grad():
        for start in range(0, n_sim, batch_size):
            bs = min(batch_size, n_sim - start)
            
            St_batch, vt_batch = heston_model.DataSim(S0, T, mu, n_step, bs)
            # generate X
            tau_batch = tau.unsqueeze(0).expand(bs, -1)
            if K is None:
                X_batch = torch.stack((St_batch[:, :-1], vt_batch[:, :-1], tau_batch), dim = -1)
            else:
                K_batch = torch.tensor(K, dtype = torch.float64).expand(bs, n_step)
                X_batch = torch.stack((St_batch[:, :-1], vt_batch[:, :-1], tau_batch, K_batch), dim = -1)
            # generate price_hedge
            if pricer is None:
                pricer = heston_model
            price_hedge_batch = torch.zeros(bs, n_step + 1, 3, dtype = torch.float64)
            for tt in range(n_step):
                price_hedge_batch[:, tt, 0] = St_batch[:, tt]
                if single_K:
                    opt_price = pricer.Option_Price(St_batch[:, tt], torch.ones(bs, dtype = torch.float64) * K_hedge, tau_batch[:, tt], vt = vt_batch[:, tt], type = "all", integrator = integrator)
                    price_hedge_batch[:, tt, 1] = opt_price[:, 0]
                    price_hedge_batch[:, tt, 2] = opt_price[:, 2]
                else:
                    price_hedge_batch[:, tt, 1] = pricer.Option_Price(St_batch[:, tt], torch.ones(bs, dtype = torch.float64) * K_hedge[0], tau_batch[:, tt], vt = vt_batch[:, tt], type = ["call"] * bs, integrator = integrator)
                    price_hedge_batch[:, tt, 2] = pricer.Option_Price(St_batch[:, tt], torch.ones(bs, dtype = torch.float64) * K_hedge[1], tau_batch[:, tt], vt = vt_batch[:, tt], type = ["digital_call"] * bs, integrator = integrator)
            price_hedge_batch[:, n_step, 0] = St_batch[:, -1]
            if single_K:
                price_hedge_batch[:, n_step, 1] = torch.clamp(St_batch[:, -1] - K_hedge, 0)
                price_hedge_batch[:, n_step, 2] = 1 * (St_batch[:, -1] >= K_hedge)
            else:
                price_hedge_batch[:, n_step, 1] = torch.clamp(St_batch[:, -1] - K_hedge[0], 0)
                price_hedge_batch[:, n_step, 2] = 1 * (St_batch[:, -1] >= K_hedge[1])

            X_chunks.append(X_batch)
            price_hedge_chunks.append(price_hedge_batch)

    X = torch.cat(X_chunks, dim = 0)
    price_hedge = torch.cat(price_hedge_chunks, dim = 0)
    return X, price_hedge

def StochasticData_Batch2(heston_model, S0, T, mu, n_step, hedge_type, hedge_K, hedge_T = None, n_sim = 10000, batch_size = 128, pricer = None, liab_K = None):
    '''
    ===Input Definition===
    heston_model as a class objected, a calibrated Heston model
    S0 as a scalar, spot price of the underlying instrument at t=0
    T as a scalar, time horizon of simulation
    n_step as a scalar, number of step in the simulation
    K_hedge (n_hedge - 1) as a vector of the strike prices of each hedging instruments
    n_sim as a scalar, number of simulation
    batch_size as a scalar, size of batch
    pricer as a class object that calculate the pricing of hedging instruments at each step <default = None -> heston_model>
    K as a strick price of the liability <default = None -> Not included>
    ===Output Definition===
    X (n_sim, n_step + 1, 3) as a tensor of data that feeds into the neural network: St, vt, T-t
    price_hedge (n_sim, n_step + 1, n_hedge) as a tensor of prices of hedging instrument at different time step 
    '''
    # setup
    n_hedge = len(hedge_type)
    dt = T/n_step
    tau = T - torch.arange(n_step, dtype = torch.float64) * dt
    X_chunks = []
    price_hedge_chunks = []
    integrator = GaussLegendre()
    # data simulation
    with torch.no_grad():
        for start in range(0, n_sim, batch_size):
            bs = min(batch_size, n_sim - start)
            
            St_batch, vt_batch = heston_model.DataSim(S0, T, mu, n_step, bs)
            # generate X
            tau_batch = tau.unsqueeze(0).expand(bs, -1)
            if liab_K is None:
                X_batch = torch.stack((St_batch[:, :-1], vt_batch[:, :-1], tau_batch), dim = -1)
            else:
                K_batch = torch.tensor(liab_K, dtype = torch.float64).expand(bs, n_step)
                X_batch = torch.stack((St_batch[:, :-1], vt_batch[:, :-1], tau_batch, K_batch), dim = -1)
            # generate price_hedge
            if pricer is None:
                pricer = heston_model
                print("Pricer is Heston Model.")
            price_hedge_batch = torch.zeros(bs, n_step + 1, n_hedge, dtype = torch.float64)
            for tt in range(n_step + 1):
                for hh in range(n_hedge):
                    if hedge_type[hh] == "underlying" or hedge_type[hh] == "spot" or hedge_type[hh] == 0:
                        price_hedge_batch[:, tt, hh] = St_batch[:, tt]
                    hedge_K_tensor = hedge_K[hh] * torch.ones(bs, dtype = torch.float64)
                    hedge_tau_tensor = (hedge_T[hh] - tt * dt) * torch.ones(bs, dtype = torch.float64)
                    hedge_type_list = [hedge_type[hh]] * bs
                    price_hedge_batch[:, tt, hh] = pricer.Option_Price(St_batch[:, tt], hedge_K_tensor, hedge_tau_tensor, vt = vt_batch[:, tt], type = hedge_type_list, integrator = integrator)
            
            X_chunks.append(X_batch)
            price_hedge_chunks.append(price_hedge_batch)

    X = torch.cat(X_chunks, dim = 0)
    price_hedge = torch.cat(price_hedge_chunks, dim = 0)
    return X, price_hedge

### Fetching Market Data for Calibration ###
def Market_Data(ticker = None, max_expiration = None, TD_limit = 7, OI_limit = 10):
    if ticker is None:
        ticker = "^SPX"
    ticker = yf.Ticker(ticker)
    expirations = ticker.options
    if max_expiration:
        expirations = expirations[:max_expiration]
    now = datetime.now(timezone.utc)
    min_TD = now - timedelta(days = TD_limit)
    S0 = ticker.fast_info.get("lastPrice", None)

    output_data = []
    for exp in expirations:
        chain = ticker.option_chain(exp)
        exp_T = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo = timezone.utc)
        T = int((exp_T - now).total_seconds() / (24 * 3600)) / 365
        if T <= 0:
            continue
        
        for type, df in [("call", chain.calls), ("put", chain.puts)]:
            if df is None or df.empty:
                continue

            for col in ["strike", "lastPrice", "lastPrice", "openInterest", "lastTradeDate"]:
                if col not in df.columns:
                    df[col] = pd.NA

            price_last = pd.to_numeric(df["lastPrice"], errors = "coerce")
            TD_last = pd.to_datetime(df["lastTradeDate"], errors = "coerce")
            open_int = pd.to_numeric(df["openInterest"], errors = "coerce")
            strike = pd.to_numeric(df["strike"], errors = "coerce")

            valid = (TD_last >= min_TD) & (TD_last <= now) & (open_int >= OI_limit) & strike.notna() & price_last.notna()

            if not valid.any():
                continue

            K_list = strike[valid].to_numpy(dtype = "float64")
            P_list = price_last[valid].to_numpy(dtype = "float64")
            S0_list = np.full(len(K_list), S0, dtype = "float64") 
            type_list = np.full(len(K_list), 1.0 if type == "call" else 0.0, dtype = "float64")    
            T_list = np.full(len(K_list), T, dtype = "float64")

            output_data.append(np.column_stack([type_list, T_list, S0_list, K_list, P_list]))

    if not output_data:
        return torch.empty((0, 4), dtype = torch.float64)
    
    output_data = np.vstack(output_data)
    return torch.from_numpy(output_data).to(dtype = torch.float64)

def YieldCurve(opt_data, T = None):
    raw_data = opt_data.cpu().numpy()
    df = pd.DataFrame(raw_data, columns = ["type", "T", "S0", "K", "P"])
    
    S0 = df["S0"].dropna().unique()[0]
    calls = df[df["type"] == 1.0][["T", "K", "P"]].rename(columns = {"P":"call"})
    puts = df[df["type"] == 0.0][["T", "K", "P"]].rename(columns = {"P":"put"})
    paired_df = pd.merge(calls, puts, on = ["T","K"], how = "inner").dropna()
    
    if paired_df.empty:
        return torch.empty((0, 2), dtype = torch.float64)
    
    rates = []

    for TT, grouped_df in paired_df.groupby("T"):
        y = (grouped_df["call"]-grouped_df["put"]).to_numpy()
        K = grouped_df["K"].to_numpy()
        if y.size < 10:
            continue
        
        X = np.column_stack([np.ones_like(K), K])
        A, B = np.linalg.lstsq(X, y, rcond = None)[0]
        A = max(A, 1e-12)
        B = min(B, -1e-12)

        # -K exp (-q*T) = B >>> -ln(A/S)/T
        q = -np.log(A / S0) / TT 
        r = -np.log(-B) / TT

        rates.append([TT, r, q])

    return torch.tensor(rates,dtype= torch.float64)

def Yield(opt_data):
    raw_data = opt_data.cpu().numpy()
    df = pd.DataFrame(raw_data, columns = ["type", "T", "S0", "K", "P"])
    
    S0 = df["S0"].dropna().unique()[0]
    calls = df[df["type"] == 1.0][["T", "K", "P"]].rename(columns = {"P":"call"})
    puts = df[df["type"] == 0.0][["T", "K", "P"]].rename(columns = {"P":"put"})
    paired_df = pd.merge(calls, puts, on = ["T","K"], how = "inner").dropna()
    
    if paired_df.empty:
        return torch.empty((0, 2), dtype = torch.float64)
    
    paired_tensor = torch.from_numpy(paired_df.to_numpy()).to(dtype = torch.float64)
    T = paired_tensor[:,0]
    K = paired_tensor[:,1]
    LHS = paired_tensor[:,2] - paired_tensor[:,3]

    r = nn.Parameter(torch.zeros(1, dtype = torch.float64))
    q = nn.Parameter(torch.zeros(1, dtype = torch.float64))
    optimiser  = torch.optim.Adam([r, q], lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8)

    for epoch in range(10000):
        optimiser.zero_grad()
        RHS = S0 * torch.exp(-q * T) - K * torch.exp(-r * T)
        loss = torch.mean((LHS - RHS) ** 2)
        loss.backward()
        optimiser.step()

    return r.item(), q.item()

def Rates_from_curve(opt_data, T):
    yield_curve = YieldCurve(opt_data)

    match_index = (yield_curve[:, 0] == T).nonzero(as_tuple=True)[0]
    if match_index.numel() != 0:
        T_index = match_index.item()
        return yield_curve[T_index, 1], yield_curve[T_index, 2]
    
    if T < yield_curve[0, 0]:
        return yield_curve[0, 1], yield_curve[0, 2]
    if T > yield_curve[-1, 0]:
        return yield_curve[-1, 1], yield_curve[-1, 2]
    
    upper_index = torch.searchsorted(yield_curve[:,0], T)
    lower_index = upper_index - 1
    upper_T = yield_curve[upper_index, 0]
    upper_rates = yield_curve[upper_index, 1:]
    lower_T = yield_curve[lower_index, 0]
    lower_rates = yield_curve[lower_index, 1:]
    interpolated_rates = lower_rates + (upper_rates - lower_rates) * (T - lower_T) / (upper_T - lower_T)
    return interpolated_rates[0], interpolated_rates[1] 

def Var_YF(ticker, T, trading_days = 252):
    T = int(T)
    data = yf.Ticker(ticker).history(period = str(T)+"y", interval = "1d", auto_adjust = True)
    close = data["Close"].dropna()

    price = torch.tensor(close.values, dtype = torch.float64)
    log_ret = torch.log(price[1:] / price[:-1])
    var_daily = torch.var(log_ret, unbiased = True)
    var_annual = var_daily * trading_days

    return var_annual

def Mean_YF(ticker, T, trading_days = 252):
    T = int(T)
    data = yf.Ticker(ticker).history(period = str(T)+"y", interval = "1d", auto_adjust = True)
    close = data["Close"].dropna()

    price = torch.tensor(close.values, dtype = torch.float64)
    log_ret = torch.log(price[1:] / price[:-1])
    mean_daily = torch.mean(log_ret)
    mean_annual = mean_daily * trading_days

    return mean_annual

#print(Mean_YF("^SPX", 5).item())
### Calibration ###
def Calibration(heston_model, opt_data, n_epoch = 100, lr = 1e-3, batch_size = 512, _scheduler = True, verbose = True, result = False):
    device = next(heston_model.parameters()).device
    dtype = next(heston_model.parameters()).dtype

    n_opt = opt_data.shape[0]
    optimiser  = torch.optim.Adam(heston_model.parameters(), lr = lr, betas = (0.9, 0.999), eps = 1e-8)
    if _scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = max(n_epoch//5, 1), gamma = 0.5)
    integrator = GaussLegendre()

    heston_model.train()
    for epoch in range(n_epoch):
        index = torch.randperm(n_opt, device = device)
        epoch_loss = 0.0

        for start in range(0, n_opt, batch_size):
            end = min(start + batch_size, n_opt)
            batch_index = index[start: end]
            opt_data_batch = opt_data[batch_index]

            optimiser.zero_grad(set_to_none = True)

            loss = heston_model.Loss(opt_data_batch)
            loss.backward()
            optimiser.step()
            heston_model.ClampParam()

            epoch_loss = epoch_loss + float(loss.item())
        if _scheduler:
            scheduler.step()
        if verbose:
            cur_lr = optimiser.param_groups[0]["lr"]
            if epoch % (n_epoch // 10) == 0 or epoch == n_epoch - 1:
                print(f"Epoch {epoch + 1: 02d}/{n_epoch} | RMSE Loss: {np.sqrt(epoch_loss/n_opt): .6f} | LR: {cur_lr: .3g}")
    
    if result:
        type = opt_data[:, 0]
        type = ["call" if ty.item() == 1.0 else "put" for ty in type]
        T = opt_data[:, 1] 
        S0 = opt_data[:, 2]
        K = opt_data[:, 3]
        market_price = opt_data[:, 4]
        model_price = heston_model.Option_Price(S0, K, T, type, integrator = integrator)
        
        diff = torch.abs(model_price-market_price)
        percent_diff = diff/market_price
        diff = diff.detach().numpy()
        percent_diff = percent_diff.detach().numpy()
        market_price = market_price.detach().numpy()
        model_price = model_price.detach().numpy()
        T = T.detach().numpy()

        print(f"Market price min: {market_price.min()} and max: {market_price.max()}")
        print(f"Discrepancy min: {diff.min()} and max: {diff.max()} | mean: {diff.mean()}")
        print(f"% Discrepancy min: {percent_diff.min()} and max: {percent_diff.max()}| mean: {percent_diff.mean()}")

        plt.hist(diff, bins=50, density=True, alpha=0.7, color="blue")
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.title("Histogram of Difference")
        plt.show()

        plt.figure(figsize = (10, 5))
        plt.scatter(T, percent_diff, label = "%Diff", alpha = 0.7, s = 20)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Scatter Plot of Two Datasets Over Time")
        plt.legend()
        plt.show()

### MAIN ###
def main():
    # fetch market data
    new_opt_data = False
    if new_opt_data:
        market_data_option = Market_Data(OI_limit = 100)
        print(market_data_option.shape[0])
        torch.save(market_data_option, f"{current_dir}/data_1508_H.pt")
    else:
        market_data_option = torch.load(f"{current_dir}/data_1508_H.pt")
    T_list = market_data_option[:, 1].unique()*365
    T_filter = (market_data_option[:, 1] >= 1/12) & (market_data_option[:, 1] <= 1.25)
    market_data_option = market_data_option[T_filter]
    print(f"S0: {market_data_option[0,2]}")
    print(market_data_option.shape)
    # estimate r and q
    #r, q = Rates_from_curve(market_data_option, 181/365)
    r, q = Yield(market_data_option)
    print(f"r: {r}, q: {q}")
    #torch.save(torch.tensor([r, q], dtype = torch.float64), f"{current_dir}/rates_1508.pt") 
    
    kappa = 1.5
    theta = 0.04
    sigma = 0.3
    v0 = 0.04
    rho = -0.9

    # initialise Heston Model
    heston_model = HestonModel(r, q)
    t1 = time.time()
    #heston_model.SetParam(kappa, theta, sigma, v0, rho)
    #loss = heston_model.Loss(market_data_option)
    #print(loss)
    # calibration
    save_heston = False
    if save_heston:
        v0 = Var_YF("^SPX", 1).item()
        heston_model.SetParam(v0 = v0)
        Calibration(heston_model, market_data_option, n_epoch = 1000, result = False, _scheduler = False)
        torch.save(heston_model.state_dict(), f"{current_dir}/calibrated_heston1v.pt")
        Calibration(heston_model, market_data_option, n_epoch = 1000, result = True, _scheduler = True)
        torch.save(heston_model.state_dict(), f"{current_dir}/calibrated_heston2v.pt")
    else:
        heston_model.load_state_dict(torch.load(f"{current_dir}/calibrated_heston1v.pt"))

    print(f"Running time: {time.time()-t1:.0f}s")
    print(f"Model parameters: kappa = {heston_model.kappa.item():.4f}, theta = {heston_model.theta.item():.4f}, sigma = {heston_model.sigma.item():.4f}, v0 = {heston_model.v0.item():.4f}, rho = {heston_model.rho.item():.4f}")

main()
#print(Var_YF("^SPX", 1))
#### TESTING CODE ####
if not True:
    # Market environment
    S0 = torch.tensor([100,100])
    r = 0.02
    q = 0.00
    T = torch.tensor([0.25,0.25])
    K = torch.tensor([100, 100])
    type = ["digital_call","call"]
    kappa = 1.5
    theta = 0.04
    sigma = 0.3
    v0 = 0.04
    rho = -0.9
    mu = 0.04
    K_hedge = [100, 100]
    vt = torch.tensor([0.04, 0.0399])
    z = 1

    model = HestonModel(r, q)
    model.SetParam(kappa=kappa, theta=theta, sigma=sigma, v0=v0, rho=rho)
    P = model.Option_Price(100, 100, 0.25, "digital_call", v0) *100
    print(P)
    ### test char_func ###
    #print(model.char_func(z, S0, T, vt))
    ### test Integrand ###
    #print(model.Integrand(z, S0, K, T, vt))
    ### test Prob ###
    #print(model.Prob(S0, K, T, vt))
    ### test OptionPrice ###
    print(model.Option_Price(S0, K, T, type, vt))
    #StochasticData(heston_model, S0, T, mu, n_step, K_hedge, n_sim = 10000)
    #X, H = StochasticData(model, 100, 1, mu, 5, K_hedge, 10)
    #print(X)
    #model.Integrand(0.1, K, T)
    #print(model.Prob(K[0], T[0]))
    #model.SetParam(kappa = 0, theta = 0, sigma = 1e-8, v0 = 1, rho=0)
    #StochasticData(model, 100, 0.5, mu, 5, K_hedge, n_sim = 5)
    #Price = model.Option_Price(S0, K, T, type, vt)
    #print(Price)
    n_step = 10
    n_sim = 10
    T_sim = 0.5
    dt = T_sim/n_step
    tau = T_sim - torch.arange(n_step, dtype = torch.float64) * dt
    tau = tau.unsqueeze(0).expand(n_sim, -1)
    #SS, vv = model.DataSim(100, T_sim, r, n_step, n_sim)
    tt = 5
    #A = model.Option_Price(SS[:, tt], torch.ones(n_sim, dtype = torch.float64) * K_hedge[1], tau[:, tt], type = ["digital_call"] * n_sim)
    #print(A)
    #P = torch.mean(torch.clamp(SS[:,-1]-K[0].item(), min = 0)) * np.exp(-r*T[0].item())
    #print(P.item()) 
    #SS = SS.detach().numpy()
    #vv = vv.detach().numpy()
    #time = np.linspace(0, 1, n_step +1)

    if False:
        # Plot asset price paths
        plt.figure(figsize=(12, 5))
        for i in range(n_sim):
            plt.plot(time, SS[i], lw=0.7, alpha=0.7)
        plt.title('Simulated Asset Price Paths ($S_t$)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot variance paths
        plt.figure(figsize=(12, 5))
        for i in range(n_sim):
            plt.plot(time, vv[i], lw=0.7, alpha=0.7)
        plt.title('Simulated Variance Paths ($v_t$)')
        plt.xlabel('Time')
        plt.ylabel('Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

