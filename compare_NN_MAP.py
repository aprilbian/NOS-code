from tools import map_fun
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from hdm_modelv1 import HDM_modelbin
from tools import *
import seaborn as sns

### hyper parameters setting
V = 3
M = 2048
D = 128
I = int(np.log2(M))
NN_enc = 4*D
NN_dec = 4*D
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = {
    'V' : V,
    'M' : M,
    'D' : D,
    'NN_enc' : NN_enc,
    'NN_dec' : NN_dec,
    'device' : device}

save_path = 'V3M11D128.pth'

model = HDM_modelbin(opt)
model.load_state_dict(torch.load(save_path))
model.eval().to(device)

def construct_table():
    '''maintain a lookup table in the memory'''
    table = torch.zeros((V, M, D)).to(device)
    one_hots = np.zeros((1,M, M))                   # batch_size = 1
    one_hots[0,:,:] = np.eye(M)
    one_hots = torch.from_numpy(one_hots).float().to(device)
    
    for v in range(V):
        for m in range(M):
            enc_vm = model.enc[v](one_hots[:,m,:])
            table[v,m,:] = model.normalize(enc_vm, pwr = D/V)
    
    lookup = table.detach().view((-1,D))
    lookup = lookup.transpose(1,0)     # (D, V*M)

    return lookup

def gen_data_bin(num):
    '''generate data for training and testing'''
    raw_bits = np.random.randint(2,size = (num, V, I))
    one_hots = np.zeros((num, V, M))
    idx = np.zeros((num, V))

    for n in range(num):
        for v in range(V):
            de_val = bi2de(raw_bits[n,v,:])
            idx[n,v] = de_val
            one_hots[n,v,de_val] = 1

    return raw_bits, one_hots, idx

lookup_table = construct_table()

test_num = 100000
batch_size = 10000

def test_model():
    model.eval()

    test_SNR = [i for i in range(5)]
    de2bin_map = map_fun(I)

    test_bits, testset, _ = gen_data_bin(test_num)
    testset = torch.from_numpy(testset).float()

    with torch.no_grad():

        for snr in test_SNR:
            Iter = int(test_num/batch_size)
            berr = 0
            berr1 = 0
            sigma = 10**(-snr/10)

            for iter in range(Iter):
                raw_data = testset[iter*batch_size:(iter+1)*batch_size,:,:].to(opt['device'])
                bit_data = test_bits[iter*batch_size:(iter+1)*batch_size,:,:]
                prob,y = model(raw_data, snr) # (batch, V, M)
                prob = torch.nn.Softmax(dim = -1)(prob)

                # directly calculate the inner-product
                inner_pro = torch.matmul(y, lookup_table)    # (batch, V*M)
                inner_pro = inner_pro.view((-1, V,M))
                inner_pro = torch.exp(inner_pro/sigma)
                total_pro = torch.sum(inner_pro, dim = -1).unsqueeze(-1)   # (batch, V, 1)
                
                corr_prob = inner_pro/total_pro 

                # hard decision
                pred = torch.argmax(prob,dim = -1).cpu()
                cur_err = np.sum(abs(bit_data-de2bin_map[pred]))
                berr += cur_err

                pred1 = torch.argmax(corr_prob.cpu(),dim = -1)
                cur_err = np.sum(abs(bit_data-de2bin_map[pred1]))
                berr1 += cur_err
            
            ber = berr/(Iter*V*batch_size*I)
            ber1 = berr1/(Iter*V*batch_size*I)
            
            print('for snr = ', snr, ' dB:')
            print('NN BER : ', ber); print('MAP BER ', ber1)

test_model()