import torch
import kaldiio
import torch.nn as nn
import torch.nn.functional as F
from nets.crn import CRNModel
import scipy.io as sio


if __name__ == '__main__':
    cp_path = ""
    nf_path = ""
    cf_path = ""

    device = torch.device('cpu')
    crn = CRNModel(dim=83, causal=False, units=256, conv_channels=8, use_batch_norm=False, pitch_dims=3)
    crn = nn.DataParallel(crn)
    checkpoint = torch.load(cp_path)
    crn.load_state_dict(checkpoint["model"])
    crn = crn.module
    crn.to(device).eval()

    noisy_feat = kaldiio.load_mat(nf_path)
    clean_feat = kaldiio.load_mat(cf_path)
    enhanced_feat = crn.inference(torch.as_tensor(noisy_feat, torch.float32, device))
    sio.savemat('debug1.mat', {
        "noisy_feat": noisy_feat,
        "clean_feat": clean_feat,
        "enhanced_feat": enhanced_feat.numpy()
    })
