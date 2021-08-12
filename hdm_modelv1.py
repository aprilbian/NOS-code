import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class HDM_modelbin(nn.Module):

    def __init__(self,opt):
        super(HDM_modelbin, self).__init__()

        self.V = opt['V']
        self.M = opt['M']
        self.D = opt['D']
        self.NN_enc = opt['NN_enc']
        self.NN_dec = opt['NN_dec']
        self.device = opt['device']

        enc_list, dec_list = [],[]
        for _ in range(self.V):

            enc = [nn.Linear(self.M,self.NN_enc),nn.ReLU(),nn.BatchNorm1d(self.NN_enc),
                    nn.Linear(self.NN_enc,self.NN_enc),nn.ReLU(),nn.BatchNorm1d(self.NN_enc),
                        nn.Linear(self.NN_enc,self.D)] 
            enc_list.append(nn.Sequential(*enc))
        self.enc = nn.ModuleList(enc_list)
        
        for _ in range(self.V):

            dec = [nn.Linear(self.D,self.NN_dec),nn.ReLU(),nn.BatchNorm1d(self.NN_dec),
                    nn.Linear(self.NN_dec,self.NN_dec),nn.ReLU(),nn.BatchNorm1d(self.NN_dec),
                        nn.Linear(self.NN_dec,self.M)]
            dec_list.append(nn.Sequential(*dec))
        self.dec = nn.ModuleList(dec_list)
    
    def encoder(self, one_hots):

        enc_sig = torch.zeros([one_hots.shape[0],self.D]).to(self.device)

        for v in range(self.V):
            x_v = one_hots[:,v,:]
            enc_v = self.enc[v](x_v)

            enc_v = self.normalize(enc_v, self.D/self.V)
            enc_sig = enc_sig + enc_v

        trans_sig = self.normalize(enc_sig, self.D)
        return trans_sig

    def normalize(self, x, pwr=1):
        '''Normalization function'''
        power = torch.sum(x**2, -1, True)
        alpha = np.sqrt(pwr)/torch.sqrt(power)
        return alpha*x


    def forward(self, x, SNR):
        mini_batch = x.shape[0]

        enc_sig = self.encoder(x)

        ## awgn channel
        N0 = 10**(SNR/10)
        n = (torch.randn((mini_batch, self.D))/math.sqrt(N0)).to(self.device)
        y = enc_sig + n

        P_ = torch.zeros((mini_batch, self.V, self.M)).to(self.device)
        for v in range(self.V):
            p_v = self.dec[v](y)
            P_[:,v,:] = p_v

        return P_, y
