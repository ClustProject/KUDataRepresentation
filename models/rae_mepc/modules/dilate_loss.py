import torch
import numpy as np
from . import soft_dtw
from . import path_soft_dtw 

def dilate_loss(outputs, targets, gamma, device):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    target_output = targets.shape[1]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, target_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k,:,:],outputs[k,:,:])
        # Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
        D[k:k+1,:,:] = Dk     
    loss_shape = softdtw_batch(D,gamma)
    
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D,gamma)           
    Omega =  soft_dtw.pairwise_distances(torch.range(1,target_output).view(target_output,1), torch.range(1,N_output).view(N_output,1)).to(device)
    loss_temporal =  torch.sum( path*Omega ) / (target_output*N_output) 
            
    alpha = 0.5
    loss = alpha*loss_shape+ (1-alpha)*loss_temporal

    return loss_shape, D