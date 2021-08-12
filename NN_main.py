import numpy as np 
import torch 
import torch.nn as nn
from torch.nn import init
from hdm_modelv1 import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## System settings
V = 3
M = 2048
D = 128
I = int(np.log2(M))

train_snr = -1.5

# NN settings
NN_enc = 4*D
NN_dec = 4*D

# Training settings
lr = 2e-4
Epoches = 10001
Epoches = 0
batch_size = 1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_num = 50000
test_num = 1000000

name = 'SPARC'+'_V'+str(V) + '_M'+str(M)+'_D'+str(D)+'_SNR_M1p5'

save_path = 'V3M11D128.pth'
log_path = name + '/log_loss.txt'
test_path = name + '/log_per.txt'
loss_path = name + '/loss.npy'
f_log = open(log_path, "a+")
f_test = open(test_path, "a+")

opt = {
    'V' : V,
    'M' : M,
    'D' : D,
    'NN_enc' : NN_enc,
    'NN_dec' : NN_dec, 
    'device': device,}

print(opt)

de2bin_map = torch.tensor([[0], [1]])
for i in range(I-1):
    de2bin_map_top = de2bin_map.clone()
    de2bin_map_down = de2bin_map.clone()

    de2bin_map_top = torch.cat((torch.zeros(2**(i+1), 1), de2bin_map_top), 1)
    de2bin_map_down = torch.cat((torch.ones(2**(i+1), 1), de2bin_map_down), 1)
    
    de2bin_map = torch.cat((de2bin_map_top, de2bin_map_down), 0)

de2bin_map = de2bin_map.numpy()

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=2e-2)


bin_vec = np.zeros((I,1), dtype= int)
for i in range(I):
    bin_vec[i, 0] = 2**(I-i-1)

def gen_bindata_vec(num):
    '''vectorize the calculatioin'''
    raw_bits = np.random.randint(2, size = (num, V, I))
    one_hots = np.zeros((num, V, M))

    idx = np.zeros((num, V))
    idx = np.dot(raw_bits, bin_vec)[:,:,0]   # (num, V)

    one_hots[np.arange(num)[:,None], np.arange(V)[None,:], idx] = 1

    return raw_bits, one_hots, idx

def loss_fun(prob, label):
    loss = torch.zeros(1).to(opt['device'])

    for v in range(V):
        target_v = torch.argmax(label[:,v,:], dim = -1).long()
        loss = loss + nn.CrossEntropyLoss()(prob[:,v,:],target_v)
    
    return loss

model = HDM_modelbin(opt = opt).to(opt['device'])
model.apply(initNetParams)
model.load_state_dict(torch.load(save_path))

def test_model(epoch):
    '''Test the model every 1000 epochs'''
    #model.load_state_dict(torch.load(save_path))
    model.eval()
    test_SNR = [-2+i for i in range(9)]
    ber_list = []; per_list = [] 

    test_bits, testset, _ = gen_bindata_vec(test_num)
    testset = torch.from_numpy(testset).float()

    with torch.no_grad():
        f_test.write('epoch is '+ str(epoch) + '\n')

        for snr in test_SNR:
            Iter = int(test_num/batch_size)
            berr = 0; perr = 0

            for iter in range(Iter):
                raw_data = testset[iter*batch_size:(iter+1)*batch_size,:,:].to(opt['device'])
                bit_data = test_bits[iter*batch_size:(iter+1)*batch_size,:,:]
                prob,_ = model(raw_data, snr) # (batch, V, M)

                # hard decision
                pred = torch.argmax(prob,dim = -1).cpu()
                cur_err = np.sum(abs(bit_data-de2bin_map[pred]))
                berr += cur_err

                # cal the per
                label = torch.argmax(raw_data, dim = -1).cpu()
                cur_err = abs(label-pred)   # (batch,V)
                perr += torch.sum(torch.eq(cur_err, 0))
            
            ber = berr/(Iter*V*batch_size*I)
            per = 1 - perr/(batch_size*Iter*V)
            ber_list.append(ber); per_list.append(per)
            
            print('for snr = ', snr, ' dB:')
            print('BER = ', ber); print('PER = ', per)
            
            f_test.write('for snr = '+ str(snr) + ' dB:\n')
            f_test.write('BER = ' + str(ber) + '\n')

    print(ber_list); print(per_list)

if __name__ == '__main__':

    ## training......
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.4,verbose=1,min_lr=1e-6,patience=40)

    for epoch in range(Epoches):

        _, trainset, idx = gen_bindata_vec(train_num)
        trainset = torch.from_numpy(trainset).float()

        Iter = int(train_num/batch_size)
        train_loss = 0.0

        for iter in range(Iter):
            raw_data = trainset[iter*batch_size:(iter+1)*batch_size,:,:].to(opt['device'])
            prob, _ = model(raw_data, opt['train_snr'])

            optimizer.zero_grad()
            loss = loss_fun(prob, raw_data)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss/Iter
        scheduler.step(train_loss)    #update the learning rate


        if epoch % 10 == 0:
            print('epoch is', epoch)
            print('loss:', train_loss)

            ber = test_model(epoch)
            model.train()
            
            f_log.write('epoch is '+str(epoch)+'\n')
            f_log.write('loss is ' +str(train_loss)+'\n')
        
        if epoch % 1000 ==0:
            save_epoch = name + '/epoch' + str(epoch) + '.pth'
            print('save the model at epoch ' + str(epoch) +'\n')
            torch.save(model.state_dict(), save_epoch)

    test_model(0)

    f_log.close()
    f_test.close()