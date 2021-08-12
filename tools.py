import numpy as np 
import torch

def bi2str(raw_bits):
    '''bit sequence 2 string'''
    Len = raw_bits.shape[0]

    bits_list = [str(raw_bits[i]) for i in range(Len)]
    str_bits = ''.join(bits_list)

    return str_bits

def str2bi(str_bits):
    '''string 2 bit sequence(numpy)'''
    Len = len(str_bits)

    bit_seq = list(str_bits)
    bit_seq = [int(bit_seq[i]) for i in range(Len)]

    return np.array(bit_seq)

def ind2sub(id_2d, k_best):
    '''Python version of Matlab function: ind2sub'''

    Len = id_2d.shape[0]
    id_new, id_old = np.zeros(Len,dtype = int), np.zeros(Len,dtype = int)

    for idx in range(Len):

        id_old[idx] = id_2d[idx] % k_best
        id_new[idx] = int(id_2d[idx]/ k_best)
    
    return id_new, id_old

'''
def ind2sub(id_2d, keepnode, k_best):
    #Python version of Matlab function: ind2sub

    Len = id_2d.shape[0]
    id_new, id_old = np.zeros(Len,dtype = int), np.zeros(Len,dtype = int)

    for idx in range(Len):

        id_new[idx] = id_2d[idx] % k_best
        id_old[idx] = int(id_2d[idx]/ k_best)
    
    return id_old, id_new
'''

def de2bi(num,K):

    bit_seq = [0 for _ in range(K)]
    for i in range(K):
        s = num%2
        num = num//2
        bit_seq[K-i-1] = s 
        if num == 0:
            break
    return bit_seq

def bi2de(array):
    k = array.shape[0]

    num = 0
    for j in range(k):
        num += array[j]*2**(k-j-1)

    return num

def map_fun(I):
    de2bin_map = torch.tensor([[0], [1]])
    for i in range(I-1):
        de2bin_map_top = de2bin_map.clone()
        de2bin_map_down = de2bin_map.clone()

        de2bin_map_top = torch.cat((torch.zeros(2**(i+1), 1), de2bin_map_top), 1)
        de2bin_map_down = torch.cat((torch.ones(2**(i+1), 1), de2bin_map_down), 1)
        
        de2bin_map = torch.cat((de2bin_map_top, de2bin_map_down), 0)

    de2bin_map = de2bin_map.numpy()
    return de2bin_map



############# CRC Part #############


def XOR(str1, str2):
    ans = ''
    if str1[0] == '0':
        return '0', str1[1:]
    else:
        for i in range(len(str1)):
            if (str1[i] == '0' and str2[i] == '0'):
                ans = ans + '0'
            elif (str1[i] == '1' and str2[i] == '1'):
                ans = ans + '0'
            else:
                ans = ans + '1'
    return '1', ans[1:]
                

def CRC_Encoding(str1,str2):
    lenght = len(str2)
    str3 = str1 + '0'*(lenght-1)
    ans = ''
    yus = str3[0:lenght]
    for i in range(len(str1)):
        str4,yus = XOR(yus, str2)
        ans = ans+str4
        if i == len(str1)-1:
            break
        else:
            yus = yus+str3[i+lenght]
    ans = str1 + yus
    return ans

def CRC_Decoding(str1,str2): 
    lenght = len(str2)
    str3 = str1 + '0'*(lenght-1)
    ans = ''
    yus = str3[0:lenght]
    for i in range(len(str1)):
        str4,yus = XOR(yus, str2)
        ans = ans+str4
        if i == len(str1)-1:
            break
        else:
            yus = yus+str3[i+lenght]
    return yus == '0'*len(yus)


if __name__ == "__main__":

    raw_bits = "10011000000000000000011"
    crc8 = "100000111"
    codeword = CRC_Encoding(raw_bits,crc8)
    print(codeword)
    code_list = list(codeword)
    code_list[10] = str((1+int(code_list[10]))%2)
    codeword = ''.join(code_list)
    print(CRC_Decoding(codeword,crc8))

