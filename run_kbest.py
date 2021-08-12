import numpy as np
import torch 
import torch.nn as nn
from tools import *
from hdm_modelv1 import HDM_modelbin
#from stat_ber import stat_matrix, revise_prob
#from cnn_model import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

### hyper parameters setting
V = 3
M = 2048
D = 128
L = 4
I = int(np.log2(M))

NN_enc = 4*D 
NN_dec = 4*D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = {
    'V' : V,
    'M' : M,
    'D' : D,
    'I' : I,
    'NN_enc' : NN_enc,
    'NN_dec' : NN_dec,
    'device' : device
}

info_bit = int(V*(np.log2(M)))

### K_best setting
EbN0 = np.array([1+0.5*i for i in range(5)])
SNR_db = EbN0 + 10*np.log10(2*V*np.log2(M)/D)
k_tree = 128
threshold = 64
max_iter = 5000

# crc check settings
crc_len = 11
crc_str = '111000100001'

# load the NN_Model
save_path = 'V3M11D128.pth'

model = HDM_modelbin(opt = opt)
model.load_state_dict(torch.load(save_path))
model.eval().to(device)


def hdm_crc(prob, dec, k_best, is_order):
    '''crc for hdm
    prob:  (V,M); dec: (V,I)'''
    # First construct the candidate list:
    id_list = np.zeros((V, k_best))

    # deciding the order
    max_element = np.max(prob,axis=-1)
    if is_order:
        order = np.argsort(-max_element)
    else:
        order = np.arange(V)

    u = order[0]
    idx = np.argsort(-prob[u,:])[0:k_best]
    id_list[0,:] = idx
    cur_metric = prob[u,idx]
    cur_metric = cur_metric.reshape(1,-1)

    for i in range(1,V):
        v = order[i]
        id_v = np.argsort(-prob[v,:])[0:k_best]
        metric_v = prob[v,id_v].reshape(-1,1)

        metric_sum = cur_metric + metric_v         # numpy boardcast
        metric_sum = metric_sum.reshape(-1)
        
        id_2d =  np.argsort(-metric_sum)[0:k_best]
        id_new, id_old = ind2sub(id_2d,  k_best)

        cur_metric = metric_sum[id_2d].reshape(1,-1)           # worries here; if the shape still (1,-1)

        id_list[i,:] = id_v[id_new]
        id_list[0:i,:] = id_list[0:i, id_old]

    id_list = id_list[np.argsort(order),:]

    # Traverse the candidate list for crc
    crc_pass = False
    test_bits = np.zeros((V,int(np.log2(M))))

    for k in range(k_best):
        candidate = id_list[:, k]

        for v in range(V):
            m_bit =  de2bi(candidate[v], I) 
            test_bits[v,:] = m_bit
        
        # crc
        crc_bit = test_bits.reshape(-1)
        crc_bit = crc_bit.astype(int)

        str_bit = bi2str(crc_bit)
        crc_pass = CRC_Decoding(str_bit, crc_str)

        if crc_pass:
            break

    test_bits = test_bits.reshape((V,-1))

    if crc_pass:
        return crc_pass, test_bits, id_list
    else:
        return crc_pass, dec, id_list

def gen_data(num):
    '''generate data for training and testing'''
    test_bits = np.zeros((num,V,I),dtype=int)
    testset = np.zeros((num,V,M))
    for n in range(num):
        raw_bits = np.random.randint(2, size = info_bit - crc_len)
        str_bits = bi2str(raw_bits)

        str_codeword = CRC_Encoding(str_bits, crc_str)

        raw_bits = str2bi(str_codeword)
        raw_bits = raw_bits.reshape((V,I))
        test_bits[n,:,:] = raw_bits

        for v in range(V):
            de_val = bi2de(test_bits[n,v,:])
            testset[n,v,de_val] = 1

    return test_bits,testset

def construct_table():

    '''maintain a lookup table in the memory'''
    table = torch.zeros((V, M, D)).to(device)
    one_hots = np.zeros((1,M, M))                   # batch_size = 1
    one_hots[0,:,:] = np.eye(M)
    one_hots = torch.from_numpy(one_hots).float().to(device)
    

    for v in range(V):
        for m in range(M):
            enc_vm = model.enc[v](one_hots[:,m,:])
            table[v,m,:] = model.normalize(enc_vm, pwr = D/opt['V'])
    
    lookup = table.detach()
    lookup = lookup.view((-1,D))
    lookup = lookup.transpose(1,0)     # (D, V*M)

    return lookup

lookup_table = construct_table()

# ML or not
is_ml = True

if __name__ == '__main__':

    for snr in SNR_db:

        perr = 0
        perr1 = 0
        err_min = 0 # loop until err_min exceeds threshold
        count = 0   # count the number of iterations

        sigma = 10**(-snr/10)

        with torch.no_grad():
            
            while (perr <= threshold):

                test_bits,testset = gen_data(1)
                test_bits = test_bits[0,:,:]

                testset = torch.from_numpy(testset).float().to(device)

                prob,y = model(testset, snr) # (batch, V, M)
                
                prob_iter = nn.Softmax(dim = -1)(prob)
                prob_iter = prob_iter.squeeze().cpu().numpy()
                prob_iter = np.log2(prob_iter)
                '''
                # directly calculate the inner-product
                inner_pro = torch.matmul(y, lookup_table)    # (batch, V*M)
                inner_pro = inner_pro.view((-1, V,M))
                inner_pro = torch.exp(inner_pro/sigma)
                total_pro = torch.sum(inner_pro, dim = -1).unsqueeze(-1)   # (batch, V, 1)
                
                corr_prob = inner_pro/total_pro 
                corr_prob = corr_prob.squeeze().cpu().numpy()
                corr_prob = np.log2(corr_prob)'''

                count = count + 1

                # hard decision before k-best decoding
                pred_iter = np.argmax(prob_iter,axis = -1)
                #pred_iter = np.argmax(corr_prob,axis = -1)

                oneshot_bits = np.zeros((V,I))
                for v in range(V):
                    oneshot_bits[v,:] = de2bi(pred_iter[v],I)
                
                # k-best decoding
                _, dec_b, _ = hdm_crc(prob_iter, oneshot_bits, k_tree, True)
                #_, dec_b, _ = hdm_crc(corr_prob, oneshot_bits, k_tree, True)
                
                if not np.array_equal(test_bits[:,:],dec_b):
                    perr += 1
                    if(perr%20==0): print(perr)

        print('SNR = ', snr)
        print(perr/count); print(count)