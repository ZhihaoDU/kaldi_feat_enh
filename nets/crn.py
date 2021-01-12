#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class CRNModel(torch.nn.Module):
    """Convolutional recurrent network.

    This is a module of CRN, which is described in `A Convolutional Recurrent Neural Network for
    Real-Time Speech Enhancement`_. This module enhances the features of corrupted speech to their
    clean counterparts.

    .. _`A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement`:
       http://web.cse.ohio-state.edu/~tan.650/doc/papers/Tan-Wang1.interspeech18.pdf

    """

    def __init__(
            self,
            dim: int,
            causal: bool = False,
            units: int = 512,
            lstm_layers: int = 2,
            conv_channels: int = 16,
            use_batch_norm: bool = True,
            pitch_dims: int = 3,
    ):
        """Initialize Tacotron2 encoder module.

        Args:
            dim (int): Dimension of the inputs.
            causal (bool): The causality of module.
            units (int): The number of hidden units in LSTM layer.
            lstm_layers (int): The number of LSTM layers.
            conv_channels (int): The number of convolution channels.
            use_batch_norm (bool): Whether to use batch normalization for output.
            pitch_dims (int): The number of pitch dimension.
        """

        super(CRNModel, self).__init__()
        # store the hyperparameters
        self.dim = dim
        ngf = conv_channels
        self.noncausal = (not causal)
        self.pitch_dims = pitch_dims
        # 1 x T x 80
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, ngf, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf),
            nn.ELU(),
        )
        # ngf x T x 40
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(),
        )
        # ngf*2 x T x 20
        self.conv3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(),
        )
        # ngf*4 x T x 10
        self.conv4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 8),
            nn.ELU(),
        )
        # ngf*8 x T x 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 16, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 16),
            nn.ELU(),
        )
        # ngf*16 x T x 2

        # T x ngf*16*2
        self.lstm = nn.LSTM(ngf * 16 * 2 + (pitch_dims * ngf), units, lstm_layers, batch_first=True,
                            bidirectional=self.noncausal)
        self.fc = nn.Sequential(
            nn.Linear(units * (self.noncausal + 1), ngf * 16 * 2 + (pitch_dims * ngf)),
            nn.BatchNorm1d(ngf * 16 * 2 + (pitch_dims * ngf)),
            nn.ELU(),
        )
        # T x ngf*16*2
        # ngf*16 x T x 2
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16 * 2, ngf * 8, (1, 5), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 8),
            nn.ELU(),
        )
        # ngf*8 x T x 5
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(),
        )
        # ngf*4 x T x 10
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(),
        )
        # ngf*2 x T x 20
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * 2, ngf * 1, (1, 4), (1, 2), (0, 1)),
            nn.BatchNorm2d(ngf * 1),
            nn.ELU(),
        )
        # ngf*1 x T x 40
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1 * 2, 1, (1, 4), (1, 2), (0, 1)),
        )
        if use_batch_norm:
            self.deconv1.add_module("out_bn", nn.BatchNorm2d(1))
        # 1 x T x 80
        if pitch_dims > 0:
            self.pitch_enc = nn.ModuleList([nn.Sequential(
                nn.ConstantPad1d((1, 1) if self.noncausal else (2, 0), 0),
                nn.Conv1d(pitch_dims, pitch_dims * ngf, 3, 1),
                nn.BatchNorm1d(pitch_dims * ngf),
                nn.ELU(),
            )])
            for _ in range(4):
                self.pitch_enc.append(nn.Sequential(
                    nn.ConstantPad1d((1, 1) if self.noncausal else (2, 0), 0),
                    nn.Conv1d(pitch_dims * ngf, pitch_dims * ngf, 3, 1),
                    nn.BatchNorm1d(pitch_dims * ngf),
                    nn.ELU(),
                ))
            self.pitch_enc = nn.Sequential(*self.pitch_enc)

            self.pitch_dec = nn.ModuleList([nn.Sequential(
                nn.ConvTranspose1d(pitch_dims * ngf, pitch_dims * ngf, 3, 1),
                nn.BatchNorm1d(pitch_dims * ngf),
                nn.ELU(),
            ) for _ in range(4)])
            self.pitch_dec.append(nn.Sequential(
                nn.ConvTranspose1d(pitch_dims * ngf, pitch_dims, 3, 1),
            ))
            if use_batch_norm:
                self.pitch_dec[-1].add_module("out_bn", nn.BatchNorm1d(pitch_dims))
        else:
            self.pitch_enc = None
            self.pitch_dec = None

    def forward(self, xs, lens=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the feature sequence (B, Lmax, fdim).
            lens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the enhanced feature sequences (B, Lmax, eunits).
        """
        if self.pitch_dims > 0:
            spec = xs[:, :, :-3].unsqueeze(1)
            ps = xs[:, :, -3:].permute(0, 2, 1)
        else:
            spec = xs.unsqueeze(-1)
            ps = None

        c1 = self.conv1(spec)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        b, c, t, d = c5.size()
        lin = c5.permute(0, 2, 1, 3).contiguous().view(b, t, c*d)

        if self.pitch_dims > 0:
            ep = self.pitch_enc(ps).transpose(1, 2)
            lstm_in = torch.cat([lin, ep], 2)
        else:
            lstm_in = lin

        lstm_in = pack_padded_sequence(lstm_in, lens, batch_first=True, enforce_sorted=False)
        h1, _ = self.lstm(lstm_in)
        h1, _ = pad_packed_sequence(h1, batch_first=True, total_length=t)
        lstm_out = self.fc(h1.view(-1, h1.size(-1))).view(b, t, -1)

        if self.pitch_dims > 0:
            d5 = lstm_out[:, :, :c*d].view(b, t, c, d).contiguous().permute(0, 2, 1, 3)
            dp = lstm_out[:, :, c*d:].view(b, t, -1).permute(0, 2, 1)
        else:
            d5 = lstm_out.view(b, t, c, d).contiguous().permute(0, 2, 1, 3)
            dp = None

        d4 = self.deconv5(torch.cat([d5, c5], 1))
        d3 = self.deconv4(torch.cat([d4, c4], 1))
        d2 = self.deconv3(torch.cat([d3, c3], 1))
        d1 = self.deconv2(torch.cat([d2, c2], 1))
        d0 = self.deconv1(torch.cat([d1, c1], 1)).squeeze(1)
        if self.pitch_dims > 0:
            for module in self.pitch_dec:
                dp = module(dp)
                dp = dp[:, :, 1: -1] if self.noncausal else dp[:, :, :-2]
            outs = torch.cat([d0, dp.transpose(1, 2)], 2)
        else:
            outs = d0
        return outs

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequence of noisy features (L, fdim).

        Returns:
            Tensor: The sequences of enhanced features (L, fdim).

        """
        xs = x.unsqueeze(0)
        ilens = torch.tensor([x.size(0)])

        return self.forward(xs, ilens)[0]


if __name__ == '__main__':
    crn = CRNModel(dim=83, causal=False, units=16, conv_channels=2, use_batch_norm=False, pitch_dims=3)
    xs = torch.rand(2, 30, 83)
    lens = torch.LongTensor([25, 28])
    outs = crn.forward(xs, lens)
    print(outs.size())
