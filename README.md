# DNM
Source code for paper "Deep Learning Based Near-OrthogonalSuperposition Codes for Short Packets"

"NN_main.py"
Trains the deep learning model and tests the BER performance.
(You may need to use 'mkdir xxx' to save the models per 1000 epoch, just follow the
instructions when excuting the program.)

"hdm_modelv1.py"
Defination of the neural encoders and the decoders.

"compare_NN_MAP.py"
Compare the MAP decoder and the NN decoder in terms of BER.

"run_kbest.py"
Combine the NN/MAP decoder output with K-best decoding algorithm.
It simulates the final PER performance.

"tools.py"
A collection of functions; such as: de2bi, sub2ind, CRC_encoding. 
