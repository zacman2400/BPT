import torch
import numpy as np
import random
import imageio
from torch.utils import checkpoint
import sigpy as sp
import cupy
from math import ceil
import cupy
import sigpy
import torch
import argparse
import logging
import numpy as np
import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
from interpol import grid_pull
import random
import os
#from multi_scale_low_rank_image import MultiScaleLowRankImage
print('a')
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

import argparse
import numpy as np
import sigpy as sp
import logging
import h5py
import torch
import cupy as xp

##functions:
#1. warp1:
#2. rescale_flow
#3.save_images_for
#4.save_images_inv
#5.rescale_flow
#6.gen_MSLR_for
#7.gen_MSLR_adj
#8.for_field_solver
#9.inv_field_solver
#10.inv_field_solver_refine
#11.normalize
#12.kspace_scaling
#13.gen_template
#14.MultiScaleLowRankRecona
#15.gen
#16.flows
#17.MoCo_MSLR
#18. warp1
import argparse
import logging
import numpy as np
import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
#from multi_scale_low_rank_image import MultiScaleLowRankImage

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


class MultiScaleLowRankRecon:
    r"""Multi-scale low rank reconstruction.

    Considers the objective function,

    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)

    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.

    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        mps (array): sensitivity maps of shape (C, N_D, ..., N_1).
            where (N_D, ..., N_1) represents the image shape.
        T (int): number of frames.
        lamda (float): regularization parameter.
        blk_widths (tuple of ints): block widths for multi-scale low rank.
        beta (float): step-size decay factor.
        sgw (None or array): soft-gating weights.
            Shape should be compatible with dcf.
        device (sp.Device): computing device.
        comm (None or sp.Communicator): distributed communicator.
        seed (int): random seed.
        max_epoch (int): maximum number of epochs.
        decay_epoch (int): number of epochs to decay step-size.
        max_power_iter (int): maximum number of power iteration.
        show_pbar (bool): show progress bar.

    """
    def __init__(self, ksp, coord, dcf, mps, T, lamda,
                 blk_widths=[32, 64, 128], alpha=1, beta=0.5, sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=60, decay_epoch=20, max_power_iter=5,
                 show_pbar=True):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.T = T
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)

        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        self.img_shape = self.mps.shape[1:]
        self.D = len(self.img_shape)
        self.J = len(self.blk_widths)
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        self.B = [self._get_B(j) for j in range(self.J)]
        self.G = [self._get_G(j) for j in range(self.J)]

       # self._normalize()

    def _get_B(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.D, oshift=[0] * self.D)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        with self.device:
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def _get_G(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]
        n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

        M_j = sp.prod(b_j)
        P_j = sp.prod(n_j)
        return M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5

    def _normalize(self):
        with self.device:
            # Estimate maximum eigenvalue.
            coord_t = sp.to_device(self.coord[:self.tr_per_frame], self.device)
            dcf_t = sp.to_device(self.dcf[:self.tr_per_frame], self.device)
            F = sp.linop.NUFFT(self.img_shape, coord_t)
            W = sp.linop.Multiply(F.oshape, dcf_t)

            max_eig = sp.app.MaxEig(F.H * W * F, max_iter=self.max_power_iter,
                                    dtype=self.dtype, device=self.device,
                                    show_pbar=self.show_pbar).run()
            self.dcf /= max_eig

            # Estimate scaling.
            img_adj = 0
            dcf = sp.to_device(self.dcf, self.device)
            coord = sp.to_device(self.coord, self.device)
            for c in range(self.C):
                mps_c = sp.to_device(self.mps[c], self.device)
                ksp_c = sp.to_device(self.ksp[c], self.device)
                img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, self.img_shape)
                img_adj_c *= self.xp.conj(mps_c)
                img_adj += img_adj_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            img_adj_norm = self.xp.linalg.norm(img_adj).item()
            self.ksp /= img_adj_norm

    def _init_vars(self):
        self.L = []
        self.R = []
        for j in range(self.J):
            L_j_shape = self.B[j].ishape
            L_j = sp.randn(L_j_shape, dtype=self.dtype, device=self.device)
            L_j_norm = self.xp.sum(self.xp.abs(L_j)**2,
                                   axis=range(-self.D, 0), keepdims=True)**0.5
            L_j /= L_j_norm

            R_j_shape = (self.T, ) + L_j_norm.shape
            R_j = self.xp.zeros(R_j_shape, dtype=self.dtype)
            self.L.append(L_j)
            self.R.append(R_j)

    def _power_method(self):
        for it in range(self.max_power_iter):
            # R = A^H(y)^H L
            with tqdm(desc='PowerIter R {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for t in range(self.T):
                    self._AHyH_L(t)
                    pbar.update()

            # Normalize R
            for j in range(self.J):
                R_j_norm = self.xp.sum(self.xp.abs(self.R[j])**2,
                                       axis=0, keepdims=True)**0.5
                self.R[j] /= R_j_norm

            # L = A^H(y) R
            with tqdm(desc='PowerIter L {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for j in range(self.J):
                    self.L[j].fill(0)

                for t in range(self.T):
                    self._AHy_R(t)
                    pbar.update()

            # Normalize L.
            self.sigma = []
            for j in range(self.J):
                L_j_norm = self.xp.sum(self.xp.abs(self.L[j])**2,
                                       axis=range(-self.D, 0), keepdims=True)**0.5
                self.L[j] /= L_j_norm
                self.sigma.append(L_j_norm)

        for j in range(self.J):
            self.L[j] *= self.sigma[j]**0.5
            self.R[j] *= self.sigma[j]**0.5

    def _AHyH_L(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj= self.B[j].H(AHy_t)
            self.R[j][t] = self.xp.sum(AHy_tj * self.xp.conj(self.L[j]),
                                       axis=range(-self.D, 0), keepdims=True)

    def _AHy_R(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj = self.B[j].H(AHy_t)
            self.L[j] += AHy_tj * self.xp.conj(self.R[j][t])

    def run(self):
        with self.device:
            self._init_vars()
            self._power_method()
            self.L_init = []
            self.R_init = []
            for j in range(self.J):
                self.L_init.append(sp.to_device(self.L[j]))
                self.R_init.append(sp.to_device(self.R[j]))

            done = False
            while not done:
                try:
                    self.L = []
                    self.R = []
                    for j in range(self.J):
                        self.L.append(sp.to_device(self.L_init[j], self.device))
                        self.R.append(sp.to_device(self.R_init[j], self.device))

                    self._sgd()
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        tqdm.write('\nReconstruction diverged. '
                                   'Scaling step-size by {}.'.format(self.beta))

            if self.comm is None or self.comm.rank == 0:
                return self.L,self.R,self.B

    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.T
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                loss = 0
                for i, t in enumerate(np.random.permutation(self.T)):
                    loss += self._update(t)
                    pbar.set_postfix(loss=loss * self.T / (i + 1))
                    pbar.update()

    def _update(self, t):
        # Form image.
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # Data consistency.
        e_t = 0
        loss_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # L gradient.
            g_L_j = self.B[j].H(e_t)
            g_L_j *= self.xp.conj(self.R[j][t])
            g_L_j += lamda_j / self.T * self.L[j]
            g_L_j *= self.T

            # R gradient.
            g_R_jt = self.B[j].H(e_t)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            g_R_jt += lamda_j * self.R[j][t]

            # Precondition.
            g_L_j /= self.J * self.sigma[j] + lamda_j
            g_R_jt /= self.J * self.sigma[j] + lamda_j

            # Add.
            self.L[j] -= self.alpha * self.beta**(self.epoch // self.decay_epoch) * g_L_j
            self.R[j][t] -= self.alpha * g_R_jt

        loss_t /= 2
        return loss_t

def pca_cc(kdata=None, axis=0, target_channels=None):

    logger = logging.getLogger('PCA_CoilCompression')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')
        kdata_cc = kdata[0]
    else:
        kdata_cc = kdata

    logger.info(f'Compressing to {target_channels} channels, along axis {axis}')
    logger.info(f'Initial  size = {kdata_cc.shape} ')

    # Put channel to first axis
    kdata_cc = np.moveaxis(kdata_cc, axis, -1)
    old_channels = kdata_cc.shape[-1]
    logger.info(f'Old channels =  {old_channels} ')

    # Subsample to reduce memory for SVD
    mask_shape = np.array(kdata_cc.shape)
    mask = np.random.choice([True, False], size=mask_shape[:-1], p=[0.05, 1-0.05])

    # Create a subsampled array
    kcc = np.zeros( (old_channels, np.sum(mask)),dtype=kdata_cc.dtype)
    logger.info(f'Kcc Shape = {kcc.shape} ')
    for c in range(old_channels):
        ktemp = kdata_cc[...,c]
        kcc[c,:] = ktemp[mask]

    kdata_cc = np.moveaxis(kdata_cc, -1, axis)

    #  SVD decomposition
    logger.info(f'Working on SVD of {kcc.shape}')
    u, s, vh = np.linalg.svd(kcc, full_matrices=False)

    logger.info(f'S = {s}')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

        for e in range(len(kdata)):
            kdata[e] = np.moveaxis(kdata[e], axis, -1)
            kdata[e] = np.expand_dims(kdata[e],-1)
            logger.info(f'Shape = {kdata[e].shape}')
            kdata[e]= np.matmul(u,kdata[e])
            kdata[e] = np.squeeze(kdata[e], axis=-1)
            kdata[e] = kdata[e][..., :target_channels]
            kdata[e] = np.moveaxis(kdata[e],-1,axis)

        for ksp in kdata:
            logger.info(f'Final Shape {ksp.shape}')
    else:
        # Now iterate over and multiply by u
        kdata = np.moveaxis(kdata,axis,-1)
        kdata = np.expand_dims(kdata,-1)
        kdata = np.matmul(u, kdata)
        logger.info(f'Shape = {kdata.shape}')

        # Crop to target channels
        kdata = np.squeeze(kdata,axis=-1)
        kdata = kdata[...,:target_channels]

        # Put back
        kdata = np.moveaxis(kdata,-1,axis)
        logger.info(f'Final shape = {kdata.shape}')

    return kdata
    
def convert_utea(h5_file, max_coils=20, dsfSpokes=1.0, compress_coils=False):
    with h5py.File(h5_file, "r") as hf:
        try:
            num_encodes = np.squeeze(hf["Kdata"].attrs["Num_Encodings"])
            num_coils = np.squeeze(hf["Kdata"].attrs["Num_Coils"])
            num_frames = np.squeeze(hf["Kdata"].attrs["Num_Frames"])
            trajectory_type = [
                np.squeeze(hf["Kdata"].attrs["trajectory_typeX"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeY"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeZ"]),
            ]
            dft_needed = [
                np.squeeze(hf["Kdata"].attrs["dft_neededX"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededY"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededZ"]),
            ]
            logging.info(f"Frames {num_frames}")
            logging.info(f"Coils {num_coils}")
            logging.info(f"Encodings {num_encodes}")
            logging.info(f"Trajectory Type {trajectory_type}")
            logging.info(f"DFT Needed {dft_needed}")
        except Exception:
            logging.info("Missing H5 Attributes...")
            num_coils = 0
            while f"KData_E0_C{num_coils}" in hf["Kdata"]:
                num_coils += 1
            logging.info(f"Number of coils: {num_coils}")
            num_encodes = 0
            while f"KData_E{num_encodes}_C0" in hf["Kdata"]:
                num_encodes += 1
            logging.info(f"Number of encodes: {num_encodes}")
        # if max_coils is not None:
        #     num_coils = min(max_coils, num_coils)
        coords = []
        dcfs = []
        kdata = []
        ecgs = []
        resps = []
        print('test')
        for encode in range(num_encodes):
            print(num_encodes)
           
            try:
                time = np.squeeze(hf["Gating"][f"time"])
                order = np.argsort(time)
            except Exception:
                time = np.squeeze(hf["Gating"][f"TIME_E{encode}"])
                order = np.argsort(time)
            try:
                resp = np.squeeze(hf["Gating"][f"resp"])
                resp = resp[order]
            except Exception:
                resp = np.squeeze(hf["Gating"][f"RESP_E{encode}"])
                resp = resp[order]
            print('coord')
            coord = []
            for i in ["Z", "Y", "X"]:
                # logging.info(f"Loading {i} coords.")
                coord.append(hf["Kdata"][f"K{i}_E{encode}"][0][order])
            coord = np.stack(coord, axis=-1)
           
            dcf = np.array(hf["Kdata"][f"KW_E{encode}"][0][order])
          
            try:
                ecg = np.squeeze(hf["Gating"][f"ecg"])
                ecg = ecg[order]
            except Exception:
                ecg = np.squeeze(hf["Gating"][f"ECG_E{encode}"])
                ecg = ecg[order]
            # Get k-space
            ksp = []
            print('coils')
            for c in range(num_coils):
                print(c)
                ksp.append(
                    hf["Kdata"][f"KData_E{encode}_C{c}"]["real"][0][order]
                    + 1j * hf["Kdata"][f"KData_E{encode}_C{c}"]["imag"][0][order]
                )
          
            ksp = np.stack(ksp, axis=0)
            logging.info("num_coils {}".format(num_coils))
            import sigpy.mri as mr
            print('noisy')
           
          
          #  noise = hf["Kdata"]["Noise"]["real"] + 1j * hf["Kdata"]["Noise"]["imag"]
          #  logging.info("Whitening ksp.")
          #  cov = mr.get_cov(noise)
          #  ksp = mr.whiten(ksp, cov)
          
           
           # logging.warning(f"{err}. Scaling k-space by max value.")
            ksp /= np.abs(ksp).max()
            if compress_coils:
                logging.info("Compressing to {} channels.".format(max_coils))
                ksp = pca_cc(kdata=ksp, axis=0, target_channels=10)
            # Append to list
            coords.append(coord)
            dcfs.append(dcf)
            kdata.append(ksp)
            ecgs.append(ecg)
            resps.append(resp)
         
    # Stack the data along projections (no reason not to keep encodes separate in my case)
    print('stack')
    kdata = np.concatenate(kdata, axis=1)
    dcfs = np.concatenate(dcfs, axis=0)
    coords = np.concatenate(coords, axis=0)
    ecgs = np.concatenate(ecgs, axis=0)
    resps = np.concatenate(resps, axis=0)
    # crop empty calibration region (1800 spokes for ipf)
    kdata = kdata[:, :, :]
    coords = coords[:, :, :]
    dcfs = dcfs[:, :]
    resps = resps[:]
    ecgs = ecgs[:]
    # crop to desired number of spokes (all by default)
    totalSpokes = kdata.shape[1]
    nSpokes = int(totalSpokes // dsfSpokes)
    kdata = kdata[:, :nSpokes, :]
    coords = coords[:nSpokes, :, :]
    dcfs = dcfs[:nSpokes, :]
    resps = resps[:nSpokes]
    print('complete')
   
    # Get TR
    d_time = time[order]
    tr = d_time[1] - d_time[0]
    return kdata, coords, dcfs, resps / resps.max(), tr


def autofova(coords, dcf,kdata,device=None,
            thresh=0.15, scale=1, oversample=2.0):
    #logger = logging.getLogger('autofov')
    import cupy as xp
    # Set to GPU
    if device is None:
        device = sp.Device(0)
  #  logger.info(f'Device = {device}')
    xp = device.xp

    with device:
        # Put on GPU
        coord = 2.0 * coords
       # dcf = mri_raw.dcf[0]

        # Low resolution filter
        res = 64
        lpf = np.sum(coord ** oversample, axis=-1)
        lpf = np.exp(-lpf / (2.0 * res * res))

        # Get reconstructed size
        img_shape = sp.estimate_shape(coord)
        img_shape = [int(min(i, 64)) for i in img_shape]
        images = xp.ones([20] + img_shape, dtype=xp.complex64)
        kdata = kdata

        sos = xp.zeros(img_shape, dtype=xp.float32)

      #  logger.info(f'Kdata shape = {kdata[0].shape}')
      #  logger.info(f'Images shape = {images.shape}')

        coord_gpu = sp.to_device(coord, device=device) # coord needs to be push to device in new sigpy version

        for c in range(kdata.shape[0]):
            print(c)
        #    logger.info(f'Reconstructing  coil {c}')
            ksp_t = np.copy(kdata[c, ...])
            ksp_t *= np.squeeze(dcf)
            ksp_t *= np.squeeze(lpf)
            ksp_t = sp.to_device(ksp_t, device=device)

            sos += xp.square(xp.abs(sp.nufft_adjoint(ksp_t, coord_gpu, img_shape)))

        # Multiply by SOS of low resolution maps
        sos = xp.sqrt(sos)

    sos = sp.to_device(sos)


    # Spherical mask
    zz, xx, yy = np.meshgrid(np.linspace(-1, 1, sos.shape[0]),
                             np.linspace(-1, 1, sos.shape[1]),
                             np.linspace(-1, 1, sos.shape[2]),indexing='ij')
    rad = zz ** 2 + xx ** 2 + yy ** 2
    idx = ( rad >= 1.0)
    sos[idx] = 0.0

    # Export to file
    out_name = 'AutoFOV.h5'
   # logger.info('Saving autofov to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("IMAGE", data=np.abs(sos))

    boxc = sos > thresh * sos.max()
    boxc_idx = np.nonzero(boxc)
    boxc_center = np.array(img_shape) // 2
    boxc_shape = np.array([int(2 * max(c - min(boxc_idx[i]), max(boxc_idx[i]) - c) * scale)
                           for i, c in zip(range(3), boxc_center)])

    #  Due to double FOV scale by 2
    target_recon_scale = boxc_shape / img_shape
   # logger.info(f'Target recon scale: {target_recon_scale}')

    # Scale to new FOV
    target_recon_size = sp.estimate_shape(coord) * target_recon_scale

    # Round to 16 for blocks and FFT
    target_recon_size = 16*np.ceil( target_recon_size / 16 )

    # Get the actual scale without rounding
    ndim = coord.shape[-1]

    with sp.get_device(coord):
        img_scale = [(2.0*target_recon_size[i]/(coord[..., i].max() - coord[..., i].min())) for i in range(ndim)]

    # fix precision errors in x dir
    for i in range(ndim):
        round_img_scale = round(img_scale[i], 6)
        if round_img_scale - img_scale[i] < 0:
            round_img_scale += 0.000001
        img_scale[i] = round_img_scale

  #  logger.info(f'Target recon size: {target_recon_size}')
  #  logger.info(f'Kspace Scale: {img_scale}')

    for e in range(len(coord)):
        coords[e] *= img_scale

    new_img_shape = sp.estimate_shape(coords)
    print(sp.estimate_shape(coords))
    #print(sp.estimate_shape(mri_raw.coords[1]))
    #print(sp.estimate_shape(mri_raw.coords[2]))
    #print(sp.estimate_shape(mri_raw.coords[3]))
    #print(sp.estimate_shape(mri_raw.coords[4]))

   # logger.info('Image shape: {}'.format(new_img_shape))
    return coords

def warp1(img,flow,mps,complex=True):
    img=img.cuda()
    #img=torch.reshape(img,[1,1,304, 176, 368])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])

    spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
    size=shape
    vectors=[]
    vectors = [ torch.arange(0, s) for s in size ] 
    grids = torch.meshgrid(vectors)
    grid  = torch.stack(grids) # y, x, z
    grid  = torch.unsqueeze(grid, 0)  #add batch
    grid = grid.type(torch.FloatTensor)
    new_locs=grid.cuda()+flow
    shape=(mps.shape[1],mps.shape[2],mps.shape[3])
  #  for i in range(len(shape)):
  #      new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
    new_locs = new_locs.permute(0, 2, 3, 4, 1) 
   # new_locs = new_locs[..., [2,1,0]]
    new_locsa = new_locs[..., [0,1,2]]
    if complex==True:
        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=1,bound='zero',extrapolate=False,prefilter=True)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=1,bound='zero',extrapolate=False,prefilter=True)
        im_out=torch.complex(ima_real,ima_imag)
    else:
        im_out=grid_pull(torch.squeeze((img)),torch.squeeze(new_locsa),interpolation=1,bound='zero',extrapolate=False,prefilter=True)

    return im_out




def rescale_flow(deforma, new_res, old_res):
    """
    This function rescales the flow according to the new resolution.
    """
    flow = torch.nn.functional.interpolate(deforma, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
    flow[:,0] = flow[:,0] * new_res.shape[1] / old_res.shape[1]
    flow[:,1] = flow[:,1] * new_res.shape[2] / old_res.shape[2]
    flow[:,2] = flow[:,2] * new_res.shape[3] / old_res.shape[3]
    
    return flow

def save_images_for(image_look, deform_look):
    """
    This function saves images to disk.
    """
    imageio.mimsave('image.gif', [np.abs(image_look[i, :, :])*1e15 for i in range(50)], fps=5)
    imageio.mimsave('deform.gif', [np.abs(deform_look[i, :, :])*1e15 for i in range(50)], fps=5)
    
    

def save_images_inv(image_look, deform_look, image_still):
    """
    This function saves images to disk.
    """
    imageio.mimsave('image.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(50)], fps=4)
    imageio.mimsave('deform.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(50)], fps=4)
    imageio.mimsave('image_still.gif', [np.abs(image_still[i,:,:])*1e15 for i in range(50)], fps=4)

def rescale_flow(deforma, new_res, old_res):
    """
    This function rescales the flow according to the new resolution.
    """
    flow = torch.nn.functional.interpolate(deforma, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
    flow[:,0] = flow[:,0] * new_res.shape[1] / old_res.shape[1]
    flow[:,1] = flow[:,1] * new_res.shape[2] / old_res.shape[2]
    flow[:,2] = flow[:,2] * new_res.shape[3] / old_res.shape[3]
    
    return flow

def save_images_for(image_look, deform_look):
    """
    This function saves images to disk.
    """
    imageio.mimsave('image.gif', [np.abs(image_look[i, :, :])*1e15 for i in range(50)], fps=5)
    imageio.mimsave('deform.gif', [np.abs(deform_look[i, :, :])*1e15 for i in range(50)], fps=5)
    
def gen_MSLR_for(T,rank,block_size_adj,block_size_for,scale,mps):

    import cupy
    import numpy as np
    
    import math
    import torch
    from math import ceil

    block_torch0=[]
    block_torch1=[]
    blockH_torch=[]
    #deformL_adj=[]
    #deformR_adj=[]
    deformL_for=[]
    deformR_for=[]
    ishape0a=[]
    ishape1a=[]
    j=0
    deformL_param_adj=[]
    deformR_param_adj=[]
    deformL_param_for=[]
    deformR_param_for=[]
    #gen


    block_size0=block_size_adj
    block_size1=block_size_for

    #Ltorch=[]
    #Rtorch=[]
    import torch_optimizer as optim


    for jo in block_size0:
        print(jo)

        b_j = [min(i, jo) for i in [mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale]]
        print(b_j)
        s_j = [(b+1)//2  for b in b_j]
        i_j = [ceil((i - b + s) / s) * s + b - s 
        for i, b, s in zip([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], b_j, s_j)]
        import sigpy as sp
        block=sp.linop.BlocksToArray(i_j, b_j, s_j)
       # print(block.shape)
        C_j = sp.linop.Resize([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], i_j,
                                      ishift=[0] * 3, oshift=[0] * 3)
       # b_j = [min(i, j) for i in [mps.shape[1],mps.shape[2],mps.shape[3]]]
        w_j = sp.hanning(b_j, dtype=cupy.float32, device=0)**0.5
        W_j = sp.linop.Multiply(block.ishape, w_j)
        block_final=C_j*block*W_j
        ishape1a.append(block_final.ishape)
        block_torch1.append(sp.to_pytorch_function(block_final,input_iscomplex=False,output_iscomplex=False))
       
        temp0=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),int(block_final.ishape[3]*block_final.ishape[4]*block_final.ishape[5]),rank],device='cuda')*1
  
        temp0=1e2*temp0/torch.sum(torch.square(torch.abs(temp0)))**0.5
        print(temp0.max())
        deformL_param_adj.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
       # tempa=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],dtype=torch.float16,device='cuda')
        deformR_param_adj.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
        deformL_param_for.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
        deformR_param_for.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
    return deformL_param_for,deformR_param_for,block_torch1,ishape1a

def gen_MSLR_adj(T,rank,block_size_adj,block_size_for,scale,mps):

    import cupy
    import numpy as np
    
    import math
    import torch
    from math import ceil

    block_torch0=[]
    block_torch1=[]
    blockH_torch=[]
    #deformL_adj=[]
    #deformR_adj=[]
    deformL_for=[]
    deformR_for=[]
    ishape0a=[]
    ishape1a=[]
    j=0
    deformL_param_adj=[]
    deformR_param_adj=[]
    deformL_param_for=[]
    deformR_param_for=[]
    #gen


    block_size0=block_size_adj
    block_size1=block_size_for

    #Ltorch=[]
    #Rtorch=[]
    import torch_optimizer as optim


    for jo in block_size0:
        print(jo)

        b_j = [min(i, jo) for i in [mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale]]
        print(b_j)
        s_j = [(b+1)//2  for b in b_j]
        i_j = [ceil((i - b + s) / s) * s + b - s 
        for i, b, s in zip([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], b_j, s_j)]
        import sigpy as sp
        block=sp.linop.BlocksToArray(i_j, b_j, s_j)
       # print(block.shape)
        C_j = sp.linop.Resize([mps.shape[1]//scale,mps.shape[2]//scale,mps.shape[3]//scale], i_j,
                                      ishift=[0] * 3, oshift=[0] * 3)
       # b_j = [min(i, j) for i in [mps.shape[1],mps.shape[2],mps.shape[3]]]
        w_j = sp.hanning(b_j, dtype=cupy.float32, device=0)**0.5
        W_j = sp.linop.Multiply(block.ishape, w_j)
        block_final=C_j*block*W_j
        ishape1a.append(block_final.ishape)
        block_torch1.append(sp.to_pytorch_function(block_final,input_iscomplex=False,output_iscomplex=False))
       
        temp0=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),int(block_final.ishape[3]*block_final.ishape[4]*block_final.ishape[5]),rank],device='cuda')*1
  
        temp0=1e2*temp0/torch.sum(torch.square(torch.abs(temp0)))**0.5
        print(temp0.max())
        deformL_param_adj.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
       # tempa=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],dtype=torch.float16,device='cuda')
        deformR_param_adj.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
        deformL_param_for.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
        deformR_param_for.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
    return deformL_param_adj,deformR_param_adj,block_torch1,ishape1a

def for_field_solver(slice0,deformL_param_for, deformR_param_for, deformL_param_adj, deformR_param_adj, template, ksp, coord, dcf, mps, iterations, RO, block_torch, ishape, T1, T2, spokes_per_bin, weight_dc, weight_smoother,old_res):
    """
    This function solves the forward field given the forward and adjoint deformation parameters, template, k-space data, coordinates,
    density compensation factors (dcf), mps, iteration count, read-out dimensions (RO), block_torch, ishape, T1, T2, spokes per bin,
    and weights for data consistency (dc) and smoother.
    """

    # Move template to GPU and normalize it
    template = torch.from_numpy(template).cuda()
    template_normed = template/ torch.abs(template).max()

    # Initialize deformation look and image look arrays
    deform_look = np.zeros([T2-T1, mps.shape[1], mps.shape[3]])
    image_look = np.zeros([T2-T1, mps.shape[1], mps.shape[3]])
    
    # Move mps to CPU
    mps = torch.from_numpy(mps).cpu()
    new_res=mps
    # Initialize the deform array with forward deformation parameters
    deform = [deformL_param_for[0], deformR_param_for[0], deformL_param_for[1], deformR_param_for[1], deformL_param_for[2], deformR_param_for[2]]

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([deform[i] for i in range(6)], lr=.005)

    # Start of main loop for iterations
    for iteration in range(iterations):
        # Sample random time steps
        K = random.sample(range(T1, T2), T2-T1)

        for j in K:
            # Compute forward deformation and warp template
            deforma = flows(deformL_param_for, deformR_param_for, j-T1, block_torch, ishape)
            deforma=rescale_flow(deforma, new_res, old_res)
            im_out = warp1(template_normed, deforma, mps, complex=True)
            for i in range(2):
                im_out = warp1(im_out, deforma, mps, complex=True)
           # for i in range(5):
           #     im_out = warp1(im_out, -deforma, mps, complex=True)
            
                
            

            # Compute start and end indices for trajectory
            tr_per_frame = spokes_per_bin
            tr_start = tr_per_frame * j
            tr_end = tr_per_frame * (j + 1)

            # Obtain trajectory and density compensation factors for current time frame
            ksp_ta = torch.from_numpy(ksp[:, tr_start:tr_end, :RO]).cuda() / torch.abs(template).max()
            coord_t = torch.from_numpy(coord[tr_start:tr_end, :RO]).cuda()
            dcf_t = torch.from_numpy(dcf[tr_start:tr_end, :RO]).cuda()

            # If within the time window, save deformation and image
            if j >= T1 and j < T1 + 50:
                deform_look[j-T1] = deforma[:, 0, :, slice0, :].detach().cpu().numpy()
                image_look[j-T1] = np.abs(im_out[:, slice0, :].detach().cpu().numpy())
        
            # Compute losses for smoothness, data consistency, and deformation
            loss_grad0 = f.loss(deforma)
            loss_for = _updateb(im_out.unsqueeze(0), ksp_ta, dcf_t, coord_t, mps)
            ksp_ta = torch.from_numpy(ksp[:, tr_start:tr_end, :RO]).cuda() / torch.abs(template).max()
            coord_t = torch.from_numpy(coord[tr_start:tr_end, :RO]).cuda()
            dcf_t = torch.from_numpy(dcf[tr_start:tr_end, :RO]).cuda()


            # Compute losses for left and right deformation parameters
            loss_L0, loss_R0 = 0, 0
            for i in range(3):
                loss_L0 += torch.norm(deformL_param_for[i], 'fro')**2
                loss_R0 += torch.norm(deformR_param_for[i][:, :, :, 1:] - deformR_param_for[i][:, :, :, :-1], 'fro')**2

            # Combine losses and perform backpropagation
            loss = loss_for * weight_dc + loss_grad0 * weight_smoother+ loss_L0 * 1e-7 + loss_R0 * 1e-7
            loss.backward()

            # Update optimizer and reset gradients
            optimizer.step()
            optimizer.zero_grad()
            

        # Save images after each iteration
        save_images_for(image_look, deform_look)

    # Return updated deformation parameters
    return deformL_param_for, deformR_param_for


def inv_field_solver(slice0,deformL_param_for, deformR_param_for, deformL_param_adj, deformR_param_adj, template, ksp, coord, dcf, mps, iterations, RO, block_torch, ishape, T1, T2, spokes_per_bin, weight_dc, weight_smoother,old_res):
    """
    This function solves the forward field given the forward and adjoint deformation parameters, template, k-space data, coordinates,
    density compensation factors (dcf), mps, iteration count, read-out dimensions (RO), block_torch, ishape, T1, T2, spokes per bin,
    and weights for data consistency (dc) and smoother.
    """
   # loss_tot=0
    # Move template to GPU and normalize it
    template = torch.from_numpy(template).cuda()
    template_normed = template/ torch.abs(template).max()

    # Initialize deformation look and image look arrays
    deform_look = np.zeros([T2-T1, mps.shape[1], mps.shape[3]])
    image_look = np.zeros([T2-T1, mps.shape[1], mps.shape[3]])
    image_still=np.zeros_like(image_look)

    # Move mps to CPU
    mps = torch.from_numpy(mps).cpu()
    new_res=mps
    # Initialize the deform array with forward deformation parameters
    deform = [deformL_param_adj[0], deformR_param_adj[0], deformL_param_adj[1], deformR_param_adj[1], deformL_param_adj[2], deformR_param_adj[2]]

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([deform[i] for i in range(6)], lr=.005)

    # Start of main loop for iterations
    for iteration in range(iterations):
        # Sample random time steps
        K = random.sample(range(T1, T2), T2-T1)
        loss_tot=0
        a=0
        for j in K:
            a=a+1
            print(a)
            # Compute forward deformation and warp template
            deforma = flows(deformL_param_for, deformR_param_for, j-T1, block_torch, ishape)
            deforma=rescale_flow(deforma, new_res, old_res)
            im_out = warp1(template_normed, deforma, mps, complex=True)
            for i in range(2):
                im_out = warp1(im_out, deforma, mps, complex=True)
            deforma_rev = flows(deformL_param_adj, deformR_param_adj, j-T1, block_torch, ishape)
            deforma_rev=rescale_flow(deforma_rev, new_res, old_res)
            
            im_out_rev = warp1(im_out, deforma_rev, mps, complex=True)
            for i in range(1):
                im_out_rev = warp1(im_out_rev, deforma_rev, mps, complex=True)
            # Compute start and end indices for trajectory
            '''
            tr_per_frame = spokes_per_bin
            tr_start = tr_per_frame * 0
            tr_end = tr_per_frame * (0 + 1)
            ksp_ta = torch.from_numpy(ksp[:, tr_start:tr_end, :RO]).cuda() / torch.abs(template).max()
            coord_t = torch.from_numpy(coord[tr_start:tr_end, :RO]).cuda()
            dcf_t = torch.from_numpy(dcf[tr_start:tr_end, :RO]).cuda()
           # loss_for = _updateb(im_out_rev.unsqueeze(0), ksp_ta, dcf_t, coord_t, mps)
            '''
          


            # If within the time window, save deformation and image
            if j >= T1 and j < T1 + 50:
                deform_look[j-T1] = deforma_rev[:, 0, :, slice0, :].detach().cpu().numpy()
                image_look[j-T1] = np.abs(im_out[:, slice0, :].detach().cpu().numpy())
                image_still[j-T1]=np.abs(im_out_rev[:,slice0,:].detach().cpu().numpy())

            # Compute losses for smoothness, data consistency, and deformation
            loss_grad0 = f.loss(deforma_rev)
            loss_adj=torch.nn.L1Loss()

            # Compute losses for left and right deformation parameters
            loss_L0, loss_R0 = 0, 0
           # for i in range(3):
            #    loss_L0 += torch.norm(deformL_param_for[i], 'fro')**2
             #   loss_R0 += torch.norm(deformR_param_for[i][:, :, :, 1:] - deformR_param_for[i][:, :, :, :-1], 'fro')**2

            # Combine losses and perform backpropagation
            loss =loss_adj(torch.squeeze(torch.abs(im_out_rev)),torch.squeeze(torch.abs(template_normed))) * weight_dc+ loss_grad0*weight_smoother
             #+ loss_L0 * 1e-7 + loss_R0 * 1e-7
          
            loss.backward()
            loss_tot=loss_tot+loss

            # Update optimizer and reset gradients
            optimizer.step()
            optimizer.zero_grad()

        # Save images after each iteration
        save_images_inv(image_look, deform_look,image_still)
        print(loss_tot)
    # Return updated deformation parameters
    return deformL_param_adj, deformR_param_adj


def inv_field_solver_refine(slice0,deformL_param_for, deformR_param_for, deformL_param_adj0, deformR_param_adj0, deformL_param_adj1, deformR_param_adj1, template, ksp, coord, dcf, mps, iterations, RO, block_torch, ishape, T1, T2, spokes_per_bin, weight_dc, weight_smoother,old_res):
    """
    This function solves the forward field given the forward and adjoint deformation parameters, template, k-space data, coordinates,
    density compensation factors (dcf), mps, iteration count, read-out dimensions (RO), block_torch, ishape, T1, T2, spokes per bin,
    and weights for data consistency (dc) and smoother.
    """

    # Move template to GPU and normalize it
    template = torch.from_numpy(template).cuda()
    template_normed = template #/ torch.abs(template).max()

    # Initialize deformation look and image look arrays
    deform_look = np.zeros([T2-T1, mps.shape[1], mps.shape[3]])
    image_look = np.zeros([T2-T1, mps.shape[1], mps.shape[3]])
    image_still=np.zeros_like(image_look)
    loss_tot=0
    # Move mps to CPU
    mps = torch.from_numpy(mps).cpu()
    new_res=mps
    # Initialize the deform array with forward deformation parameters
    deform = [deformL_param_adj1[0], deformR_param_adj1[0], deformL_param_adj1[1], deformR_param_adj1[1], deformL_param_adj1[2], deformR_param_adj1[2]]

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([deform[i] for i in range(6)], lr=.005)

    # Start of main loop for iterations
    for iteration in range(iterations):
        # Sample random time steps
        K = random.sample(range(T1, T2), T2-T1)

        for j in K:
            # Compute forward deformation and warp template
            
            deforma = flows(deformL_param_for, deformR_param_for, j-T1, block_torch, ishape)
            deforma=rescale_flow(deforma, new_res, old_res)
            im_out = warp1(template_normed, deforma, mps, complex=True)
            #im_out = warp1(template_normed, deforma, mps, complex=True)
            for i in range(2):
                im_out = warp1(im_out, deforma, mps, complex=True)
            deforma_rev = flows(deformL_param_adj0, deformR_param_adj0, j-T1, block_torch, ishape) #-deforma
            deforma_rev=rescale_flow(deforma_rev, new_res, old_res)
             
            im_out_rev = warp1(im_out, deforma_rev, mps, complex=True)
            for i in range(1):
                im_out_rev = warp1(im_out_rev, deforma_rev, mps, complex=True)
           
            deforma_rev1 = flows(deformL_param_adj1, deformR_param_adj1, j-T1, block_torch, ishape)
            deforma_rev1=rescale_flow(deforma_rev1, new_res, old_res)
            im_out_rev1 = warp1(im_out_rev, deforma_rev1, mps, complex=True)


            # If within the time window, save deformation and image
            if j >= T1 and j < T1 + 50:
                deform_look[j-T1] = deforma_rev1[:, 0, :, slice0, :].detach().cpu().numpy()
                image_look[j-T1] = np.abs(im_out[:, slice0, :].detach().cpu().numpy())
                image_still[j-T1]=np.abs(im_out_rev1[:,slice0,:].detach().cpu().numpy())

            # Compute losses for smoothness, data consistency, and deformation
            loss_grad0 = f.loss(deforma_rev1)
            loss_adj=torch.nn.L1Loss()

            # Compute losses for left and right deformation parameters
            loss_L0, loss_R0 = 0, 0
           # for i in range(3):
            #    loss_L0 += torch.norm(deformL_param_for[i], 'fro')**2
             #   loss_R0 += torch.norm(deformR_param_for[i][:, :, :, 1:] - deformR_param_for[i][:, :, :, :-1], 'fro')**2

            # Combine losses and perform backpropagation
            #loss = loss_adj(torch.squeeze(torch.abs(im_out_rev1)),torch.squeeze(torch.abs(template_normed))) * weight_dc + loss_grad0 * weight_smoother  #+ loss_L0 * 1e-7 + loss_R0 * 1e-7
            #loss.backward()
           # loss_tot=loss_tot+loss
            # Update optimizer and reset gradients
           # optimizer.step()
           # optimizer.zero_grad()
      #  print(loss_tot)

        # Save images after each iteration
        save_images_inv(image_look, deform_look,image_still)

    # Return updated deformation parameters
    return deformL_param_adj1, deformR_param_adj1


def normalize(mps,coord,dcf,ksp,tr_per_frame):
    mps=mps
   
   # import cupy
    import sigpy as sp
    device=0
    # Estimate maximum eigenvalue.
    coord_t = sp.to_device(coord[:tr_per_frame], device)
    dcf_t = sp.to_device(dcf[:tr_per_frame], device)
    F = sp.linop.NUFFT([mps.shape[1],mps.shape[2],mps.shape[3]], coord_t)
    W = sp.linop.Multiply(F.oshape, dcf_t)

    max_eig = sp.app.MaxEig(F.H * W * F, max_iter=500, device=0,
                            dtype=ksp.dtype,show_pbar=True).run()
    dcf1=dcf/max_eig
    return dcf1

def kspace_scaling(mps,dcf,coord,ksp):
    # Estimate scaling.
    img_adj = 0
    device=0
    dcf = sp.to_device(dcf, device)
    coord = sp.to_device(coord, device)
   
    for c in range(mps.shape[0]):
        print(c)
        mps_c = sp.to_device(mps[c], device)
        ksp_c = sp.to_device(ksp[c], device)
        img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, [mps.shape[1],mps.shape[2],mps.shape[3]])
        img_adj_c *= cupy.conj(mps_c)
        img_adj += img_adj_c


    img_adj_norm = cupy.linalg.norm(img_adj).item()
    print(img_adj_norm)
    ksp1=ksp/img_adj_norm
    return ksp1



def gen_template(ksp,coord,dcf,RO,spokes_per_bin):
    import sigpy as sp
    shape=sp.estimate_shape(coord)
    matrix_dim=np.ones([1,shape[0],shape[1],shape[2]])

    kspa=ksp[:,:,:RO]
    coorda=coord[:,:RO]
    dcfa=dcf[:,:RO]

    #generate sense maps
    import sigpy.mri as mr
    import sigpy as sp
    device=0
    mps = mr.app.JsenseRecon(kspa[:,:], coord=coorda[:], weights=dcfa[:], device=0).run()
   # mps = mr.app.JsenseRecon(kspa, coord=coorda, weights=dcfa, device=0).run()

    print(mps.shape)

    #normalize data

    device=0

    dcfa=normalize(mps,coorda,dcfa,kspa,spokes_per_bin)
    import cupy
    kspa=kspace_scaling(mps,dcfa,coorda,kspa)
    import sigpy
    #P=sigpy.mri.kspace_precond(cupy.array(mps), weights=cupy.array(dcf), coord=cupy.array(coord), lamda=0, device=0, oversamp=1.25)


    T =1
    device=0
    lamda = 1e-8
    blk_widths = [8,16,32]  # For low resolution.
    al=ksp.shape[1]//2
   # L_blocks,R_blocks,B = MultiScaleLowRankRecona(kspa[:,:,:], coorda[:,:], dcfa[:,:], mps, T, #lamda, device=device, blk_widths=blk_widths).run()

    mpsa=mps

    im_test=np.zeros([1,mpsa.shape[1],mpsa.shape[2],mpsa.shape[3]],dtype=np.complex64)
    temp=0
    '''
    for i in range(1):
        for j in range(3):
            temp=temp+B[j](L_blocks[j]*R_blocks[j][i])
        im_test[i]=temp.get()
    '''
    #im_testa=im_test
    return mps,kspa,coorda,dcfa

class MultiScaleLowRankRecona:
    r"""Multi-scale low rank reconstruction.

    Considers the objective function,

    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)

    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.

    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        mps (array): sensitivity maps of shape (C, N_D, ..., N_1).
            where (N_D, ..., N_1) represents the image shape.
        T (int): number of frames.
        lamda (float): regularization parameter.
        blk_widths (tuple of ints): block widths for multi-scale low rank.
        beta (float): step-size decay factor.
        sgw (None or array): soft-gating weights.
            Shape should be compatible with dcf.
        device (sp.Device): computing device.
        comm (None or sp.Communicator): distributed communicator.
        seed (int): random seed.
        max_epoch (int): maximum number of epochs.
        decay_epoch (int): number of epochs to decay step-size.
        max_power_iter (int): maximum number of power iteration.
        show_pbar (bool): show progress bar.

    """
    def __init__(self, ksp, coord, dcf, mps, T, lamda,
                 blk_widths=[32, 64, 128], alpha=1, beta=0.5, sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=60, decay_epoch=20, max_power_iter=5,
                 show_pbar=True):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.T = T
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)

        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        self.img_shape = self.mps.shape[1:]
        self.D = len(self.img_shape)
        self.J = len(self.blk_widths)
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        self.B = [self._get_B(j) for j in range(self.J)]
        self.G = [self._get_G(j) for j in range(self.J)]

        #self._normalize()

    def _get_B(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.D, oshift=[0] * self.D)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        with self.device:
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def _get_G(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]
        n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

        M_j = sp.prod(b_j)
        P_j = sp.prod(n_j)
        return M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5

   
    def _init_vars(self):
        self.L = []
        self.R = []
        for j in range(self.J):
            L_j_shape = self.B[j].ishape
            L_j = sp.randn(L_j_shape, dtype=self.dtype, device=self.device)
            L_j_norm = self.xp.sum(self.xp.abs(L_j)**2,
                                   axis=range(-self.D, 0), keepdims=True)**0.5
            L_j /= L_j_norm

            R_j_shape = (self.T, ) + L_j_norm.shape
            R_j = self.xp.zeros(R_j_shape, dtype=self.dtype)
            self.L.append(L_j)
            self.R.append(R_j)

    def _power_method(self):
        for it in range(self.max_power_iter):
            # R = A^H(y)^H L
            with tqdm(desc='PowerIter R {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for t in range(self.T):
                    self._AHyH_L(t)
                    pbar.update()

            # Normalize R
            for j in range(self.J):
                R_j_norm = self.xp.sum(self.xp.abs(self.R[j])**2,
                                       axis=0, keepdims=True)**0.5
                self.R[j] /= R_j_norm

            # L = A^H(y) R
            with tqdm(desc='PowerIter L {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for j in range(self.J):
                    self.L[j].fill(0)

                for t in range(self.T):
                    self._AHy_R(t)
                    pbar.update()

            # Normalize L.
            self.sigma = []
            for j in range(self.J):
                L_j_norm = self.xp.sum(self.xp.abs(self.L[j])**2,
                                       axis=range(-self.D, 0), keepdims=True)**0.5
                self.L[j] /= L_j_norm
                self.sigma.append(L_j_norm)

        for j in range(self.J):
            self.L[j] *= self.sigma[j]**0.5
            self.R[j] *= self.sigma[j]**0.5

    def _AHyH_L(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj= self.B[j].H(AHy_t)
            self.R[j][t] = self.xp.sum(AHy_tj * self.xp.conj(self.L[j]),
                                       axis=range(-self.D, 0), keepdims=True)

    def _AHy_R(self, t):
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj = self.B[j].H(AHy_t)
            self.L[j] += AHy_tj * self.xp.conj(self.R[j][t])

    def run(self):
        with self.device:
            self._init_vars()
            self._power_method()
            self.L_init = []
            self.R_init = []
            for j in range(self.J):
                self.L_init.append(sp.to_device(self.L[j]))
                self.R_init.append(sp.to_device(self.R[j]))

            done = False
            while not done:
                try:
                    self.L = []
                    self.R = []
                    for j in range(self.J):
                        self.L.append(sp.to_device(self.L_init[j], self.device))
                        self.R.append(sp.to_device(self.R_init[j], self.device))

                    self._sgd()
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        tqdm.write('\nReconstruction diverged. '
                                   'Scaling step-size by {}.'.format(self.beta))

            if self.comm is None or self.comm.rank == 0:
                return self.L,self.R,self.B
           

    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.T
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                loss = 0
                for i, t in enumerate(np.random.permutation(self.T)):
                    loss += self._update(t)
                    pbar.set_postfix(loss=loss * self.T / (i + 1))
                    pbar.update()

    def _update(self, t):
        # Form image.
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # Data consistency.
        e_t = 0
        loss_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # L gradient.
            g_L_j = self.B[j].H(e_t)
            g_L_j *= self.xp.conj(self.R[j][t])
            g_L_j += lamda_j / self.T * self.L[j]
            g_L_j *= self.T

            # R gradient.
            g_R_jt = self.B[j].H(e_t)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            g_R_jt += lamda_j * self.R[j][t]

            # Precondition.
            g_L_j /= self.J * self.sigma[j] + lamda_j
            g_R_jt /= self.J * self.sigma[j] + lamda_j

            # Add.
            self.L[j] -= self.alpha * self.beta**(self.epoch // self.decay_epoch) * g_L_j
            self.R[j][t] -= self.alpha * g_R_jt

        loss_t /= 2
        return loss_t


def gen(block_torcha,deformL_param,deformR_param,ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo):
    jb=int(jo[0])
    deform_patch_adj=torch.matmul(deformL_param,deformR_param[:,:,:,jb:jb+1])
    deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(ishape0[0]),int(ishape1[0]),int(ishape2[0]),int(ishape3[0]),int(ishape4[0]),int(ishape5[0])])
    deformx_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[0])).unsqueeze(0)
    deformy_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[1])).unsqueeze(0)
    deformz_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[2])).unsqueeze(0)
  
    return deformx_adj,deformy_adj,deformz_adj

def flows(deformL_param_adj,deformR_param_adj,j,block_torch1,ishape1a):
        jo=torch.ones([1])*j
        deform_adj=[]
        deform_for=[]
        #count=int(counta[0])
        for count in range(3):
           # print(count)
            ishape0=ishape1a[count][0]*torch.ones([1])
            ishape1=ishape1a[count][1]*torch.ones([1])
            ishape2=ishape1a[count][2]*torch.ones([1])
            ishape3=ishape1a[count][3]*torch.ones([1])
            ishape4=ishape1a[count][4]*torch.ones([1])
            ishape5=ishape1a[count][5]*torch.ones([1])
       # deformx0,deformy1,deformz0=torch.utils.checkpoint.checkpoint(gen,block_torch0,deformL_param_adj0,deformR_param_adj0,ishape00,ishape10,ishape20,ishape30,ishape40,ishape50,jo)
            deformx,deformy,deformz=gen(block_torch1[count],deformL_param_adj[count],deformR_param_adj[count],ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo)
           # deform_for.append(torch.cat([deformx,deformy,deformz],axis=0))
           # deformx,deformy,deformz=torch.utils.checkpoint.checkpoint(gen,block_torch[count],deformL_param_for[count],deformR_param_for[count],ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo,preserve_rng_state=False)
            deform_adj.append(torch.cat([deformx,deformy,deformz],axis=0))
        flow=deform_adj[0]+deform_adj[1]+deform_adj[2] #+deform_adj[2] #+deform_adj[3] #+deform_adj[4] #+deform_adj[3]+deform_adj[4]+deform_adj[5] #+deform_adj[6]+deform_adj[7]
        flow=flow.unsqueeze(0)
        return flow
    
class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) #*w0
       
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) #*w1
      
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) #*w2
       # dt = torch.abs(y_pred[1:, :, :, :, :] - y_pred[:-1, :, :, :, :])

      
        dy = dy
        dx = dx
        dz = dz
            #dt=dt*dt

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

     
        return grad
    
f=Grad()

def calculate_sense0(M_t,ksp,mps_c,coord_t,dcf):
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(),1.25,2)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        e_tc=F_torch.apply(M_t.cuda())
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1]) 
        
        res=(ksp-e_tca)*(dcf)**0.5 
        res=torch.reshape(res,[1,-1])
        lossb=(torch.linalg.norm(res,2))**2 
        return lossb

def _updateb(img_t,ksp_t,dcf_t,coord_t,mpsa): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  
# Data consistency.
  loss_t=0
  for c in range(mpsa.shape[0]):
       
    
    loss_t=loss_t+torch.utils.checkpoint.checkpoint(calculate_sense0,torch.cat([torch.reshape(torch.real(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1]),torch.reshape(torch.imag(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1])],axis=3),torch.reshape(ksp_t[c],[-1]),mpsa[c],torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
   
  
  return loss_t
    
class MoCo_MSLR:
    r"""Multi-scale low rank reconstruction.
    Considers the objective function,
    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)
    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.
    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        mps (array): sensitivity maps of shape (C, N_D, ..., N_1).
            where (N_D, ..., N_1) represents the image shape.
        T (int): number of frames.
        lamda (float): regularization parameter.
        blk_widths (tuple of ints): block widths for multi-scale low rank.
        beta (float): step-size decay factor.
        sgw (None or array): soft-gating weights.
            Shape should be compatible with dcf.
        device (sp.Device): computing device.
        comm (None or sp.Communicator): distributed communicator.
        seed (int): random seed.
        max_epoch (int): maximum number of epochs.
        decay_epoch (int): number of epochs to decay step-size.
        max_power_iter (int): maximum number of power iteration.
        show_pbar (bool): show progress bar.
    """
    def __init__(self, ksp, coord, dcf, mps, low_res,T, lamda, ishape,block_torch,deformL_param_for , deformR_param_for,deformL_param_adj0,deformR_param_adj0,deformL_param_adj1,deformR_param_adj1,
                 blk_widths=[32, 64, 128], alpha=.1, beta=.5,sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=60, decay_epoch=20, max_power_iter=1,
                 show_pbar=True):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.T = T
        #self.L=L
        #self.R=R
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)
        self.scale=1
        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        self.img_shape = self.mps.shape[1:]
        self.D = len(self.img_shape)
        self.J = len(self.blk_widths)
       
       
        self.deform_look=np.zeros([50,mps.shape[1],mps.shape[3]])
        self.max_epoch = max_epoch
        self.block_torch=block_torch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)
        self.ishape=ishape
        self.deformL_param_for=deformL_param_for
        self.deformR_param_for=deformR_param_for
        self.deformL_param_adj0=deformL_param_adj0
        self.deformR_param_adj0=deformR_param_adj0
        self.deformL_param_adj1=deformL_param_adj1
        self.deformR_param_adj1=deformR_param_adj1
        self.old_res=low_res
        self.new_res=mps
     

        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp.dtype
        self.C, self.num_tr, self.num_ro = self.ksp.shape
        self.tr_per_frame = self.num_tr // self.T
        
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        self.B = [self._get_B(j) for j in range(self.J)]
        self.G = [self._get_G(j) for j in range(self.J)]

       # self._normalize()
    
    def get_for_field(self,t):
         flow=0
         mps=self.mps
         deform_for=[]
         for count in range(3):
            #self.scale=3
            deform_patch_for=torch.matmul(self.deformL_param_for[count].cuda(),self.deformR_param_for[count][:,:,:,t:t+1].cuda())
            deform_patch_for=torch.reshape(deform_patch_for,[3,int(self.ishape[count][0]),int(self.ishape[count][1]),int(self.ishape[count][2]),int(self.ishape[count][3]),int(self.ishape[count][4]),int(self.ishape[count][5])])
            deformx_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[0])).unsqueeze(0)
            deformy_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[1])).unsqueeze(0)
            deformz_for=torch.squeeze(self.block_torch[count].apply(deform_patch_for[2])).unsqueeze(0)
            deform_for.append(torch.cat([deformx_for,deformy_for,deformz_for],axis=0))
            flow=flow+deform_for[count] 
         
        
         flow=torch.nn.functional.interpolate(flow.unsqueeze(0), size=[self.new_res.shape[1],self.new_res.shape[2],self.new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
         flow_for=torch.reshape(flow,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
         flow_for[:,0]=flow_for[:,0]*self.new_res.shape[1]/self.old_res.shape[1]
         flow_for[:,1]=flow_for[:,1]*self.new_res.shape[2]/self.old_res.shape[2]
         flow_for[:,2]=flow_for[:,2]*self.new_res.shape[3]/self.old_res.shape[3]
         
          
         return flow_for
    def get_adj_field(self,t):
        flow=0
        mps=self.mps
        deform_adj=[]
        for count in range(3):
            deform_patch_adj=torch.matmul(self.deformL_param_adj0[count].cuda(),self.deformR_param_adj0[count][:,:,:,t:t+1].cuda())
            deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(self.ishape[count][0]),int(self.ishape[count][1]),int(self.ishape[count][2]),int(self.ishape[count][3]),int(self.ishape[count][4]),int(self.ishape[count][5])])
            deformx_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[0])).unsqueeze(0)
            deformy_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[1])).unsqueeze(0)
            deformz_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[2])).unsqueeze(0)
            deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
            flow=flow+deform_adj[count] 
     
        flow=torch.reshape(flow,[1,3,self.old_res.shape[1],self.old_res.shape[2],self.old_res.shape[3]])
        flow_adj=torch.nn.functional.interpolate(flow, size=[self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
        flow_adj0=torch.reshape(flow_adj,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        flow_adj0[:,0]=flow_adj0[:,0]*self.new_res.shape[1]/self.old_res.shape[1]
        flow_adj0[:,1]=flow_adj0[:,1]*self.new_res.shape[2]/self.old_res.shape[2]
        flow_adj0[:,2]=flow_adj0[:,2]*self.new_res.shape[3]/self.old_res.shape[3]
        
        return flow_adj0
    
    def get_adj_field_refine(self,t):
        flow=0
        mps=self.mps
        deform_adj=[]
        for count in range(3):
            deform_patch_adj=torch.matmul(self.deformL_param_adj1[count].cuda(),self.deformR_param_adj1[count][:,:,:,t:t+1].cuda())
            deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(self.ishape[count][0]),int(self.ishape[count][1]),int(self.ishape[count][2]),int(self.ishape[count][3]),int(self.ishape[count][4]),int(self.ishape[count][5])])
            deformx_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[0])).unsqueeze(0)
            deformy_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[1])).unsqueeze(0)
            deformz_adj=torch.squeeze(self.block_torch[count].apply(deform_patch_adj[2])).unsqueeze(0)
            deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
            flow=flow+deform_adj[count] 
         
     
        flow=torch.reshape(flow,[1,3,self.old_res.shape[1],self.old_res.shape[2],self.old_res.shape[3]])
        flow_adj=torch.nn.functional.interpolate(flow, size=[self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
        flow_adj1=torch.reshape(flow_adj,[1,3,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        flow_adj1[:,0]=flow_adj1[:,0]*self.new_res.shape[1]/self.old_res.shape[1]
        flow_adj1[:,1]=flow_adj1[:,1]*self.new_res.shape[2]/self.old_res.shape[2]
        flow_adj1[:,2]=flow_adj1[:,2]*self.new_res.shape[3]/self.old_res.shape[3]
        
        return flow_adj1
    
    def warp(self,flow,img):
        img=img.cuda()
        mps=self.mps
        img=torch.reshape(img,[1,1,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]])
        shape=(self.mps.shape[1],self.mps.shape[2],self.mps.shape[3])
         
        spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
        shape=(mps.shape[1],mps.shape[2],mps.shape[3])
        size=shape
        vectors=[]
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        new_locs=grid.cuda()+flow
        shape=(mps.shape[1],mps.shape[2],mps.shape[3])
      #  for i in range(len(shape)):
      #      new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1) 
       # new_locs = new_locs[..., [2,1,0]]
        new_locsa = new_locs[..., [0,1,2]]

        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=1,bound='zero',extrapolate=False,prefilter=True)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=1,bound='zero',extrapolate=False,prefilter=True)
         #ima_real=torch.nn.functional.grid_sample(torch.real(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
         #ima_imag=torch.nn.functional.grid_sample(torch.imag(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
        im_out=torch.complex(ima_real,ima_imag)
        im_out=torch.squeeze(im_out)
    


        
        return im_out
        

    def _get_B(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.D, oshift=[0] * self.D)
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)
        with self.device:
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def _get_G(self, j):
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]
        s_j = [(b + 1) // 2 for b in b_j]

        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]
        n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

        M_j = sp.prod(b_j)
        P_j = sp.prod(n_j)
        return M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5

    def _normalize(self):
        with self.device:
            # Estimate maximum eigenvalue.
            coord_t = sp.to_device(self.coord[:self.tr_per_frame], self.device)
            dcf_t = sp.to_device(self.dcf[:self.tr_per_frame], self.device)
            F = sp.linop.NUFFT(self.img_shape, coord_t)
            W = sp.linop.Multiply(F.oshape, dcf_t)

            max_eig = sp.app.MaxEig(F.H * W * F, max_iter=500,
                                    dtype=self.dtype, device=self.device,
                                    show_pbar=self.show_pbar).run()
            self.dcf /= max_eig

            # Estimate scaling.
            img_adj = 0
            dcf = sp.to_device(self.dcf, self.device)
            coord = sp.to_device(self.coord, self.device)
            for c in range(self.C):
                mps_c = sp.to_device(self.mps[c], self.device)
                ksp_c = sp.to_device(self.ksp[c], self.device)
                img_adj_c = sp.nufft_adjoint(ksp_c * dcf, coord, self.img_shape)
                img_adj_c *= self.xp.conj(mps_c)
                img_adj += img_adj_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            img_adj_norm = self.xp.linalg.norm(img_adj).item()
            self.ksp /= img_adj_norm

    def _init_vars(self):
        self.L = []
        self.R = []
        for j in range(self.J):
            L_j_shape = self.B[j].ishape
            L_j = sp.randn(L_j_shape, dtype=self.dtype, device=self.device)
            L_j_norm = self.xp.sum(self.xp.abs(L_j)**2,
                                   axis=(-3,-2,-1), keepdims=True)**0.5
            L_j /= L_j_norm

            R_j_shape = (self.T, ) + L_j_norm.shape
            R_j = self.xp.zeros(R_j_shape, dtype=self.dtype)
            self.L.append(L_j)
            self.R.append(R_j)

    def _power_method(self):
        for it in range(self.max_power_iter):
            # R = A^H(y)^H L
            with tqdm(desc='PowerIter R {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for t in range(self.T):
                    self._AHyH_L(t)
                    pbar.update()

            # Normalize R
            for j in range(self.J):
                R_j_norm = self.xp.sum(self.xp.abs(self.R[j])**2,
                                       axis=0, keepdims=True)**0.5
                self.R[j] /= R_j_norm

            # L = A^H(y) R
            with tqdm(desc='PowerIter L {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for j in range(self.J):
                    self.L[j].fill(0)

                for t in range(self.T):
                    self._AHy_R(t)
                    pbar.update()

            # Normalize L.
            self.sigma = []
            for j in range(self.J):
                L_j_norm = self.xp.sum(self.xp.abs(self.L[j])**2,
                                       axis=range(-self.D, 0), keepdims=True)**0.5
                self.L[j] /= L_j_norm
                self.sigma.append(L_j_norm)

        for j in range(self.J):
            self.L[j] *= self.sigma[j]**0.5
            self.R[j] *= self.sigma[j]**0.5
            
    def _AHyH_L(self, t):
        #t=0
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc
        flow_adj0=self.get_adj_field(t)
        AHy_t=cupy.array(self.warp(flow_adj0,torch.as_tensor(AHy_t)).detach().cpu().numpy())
        AHy_t=cupy.array(self.warp(flow_adj0,torch.as_tensor(AHy_t)).detach().cpu().numpy())
        flow_adj1=self.get_adj_field_refine(t)
        AHy_t=cupy.array(self.warp(flow_adj1,torch.as_tensor(AHy_t)).detach().cpu().numpy())
        

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj= self.B[j].H(AHy_t)
            self.R[j][t] = self.xp.sum(AHy_tj * self.xp.conj(self.L[j]),
                                       axis=range(-self.D, 0), keepdims=True)

            
            
    def _AHy_R(self, t):
       # ta=0
        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc
        flow_adj0=self.get_adj_field(t)
        AHy_t=cupy.array(self.warp(flow_adj0,torch.as_tensor(AHy_t)).detach().cpu().numpy())
        AHy_t=cupy.array(self.warp(flow_adj0,torch.as_tensor(AHy_t)).detach().cpu().numpy())
        flow_adj1=self.get_adj_field_refine(t)
        AHy_t=cupy.array(self.warp(flow_adj1,torch.as_tensor(AHy_t)).detach().cpu().numpy())
      
        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.J):
            AHy_tj = self.B[j].H(AHy_t)
            self.L[j] += AHy_tj * self.xp.conj(self.R[j][t])


    def run(self):
        with self.device:
            self._init_vars()
            self._power_method()
            self.L_init = []
            self.R_init = []
            for j in range(self.J):
                self.L_init.append(sp.to_device(self.L[j]))
                self.R_init.append(sp.to_device(self.R[j]))

            done = False
            while not done:
                try:
                    self.L = []
                    self.R = []
                   
                    for j in range(self.J):
                        self.L.append(sp.to_device(self.L_init[j], self.device))
                        self.R.append(sp.to_device(self.R_init[j], self.device))

                    self._sgd()
                   
                   
                 
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        tqdm.write('\nReconstruction diverged. '
                                   'Scaling step-size by {}.'.format(self.beta))

            if self.comm is None or self.comm.rank == 0:
                return self.L,self.R,self.B
   

    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.T
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                loss = 0
                for i, t in enumerate(np.random.permutation(self.T)):
                    loss += self._update(t)
                    pbar.set_postfix(loss=loss * self.T / (i + 1))
                    pbar.update()
                     
               
               
                
               
           

    def _update(self, t):
        # Form image.
        mps=self.mps
        img_t = 0
        for j in range(self.J):
            img_t += self.B[j](self.L[j] * self.R[j][t])
        flow_for=self.get_for_field(t)
        img_t=torch.as_tensor(img_t,device='cuda')
        img_t=self.warp(flow_for,img_t).cuda()
        for i in range(2):
            img_t=self.warp(flow_for,img_t).cuda()
        img_t = xp.asarray(img_t.detach().cpu().numpy())
       

# Convert it into a CuPy array.
       # img_t = cupy.from_dlpack(img_t)
        #img_t=cupy.asarray(img_t)

        # Download k-space arrays.
        tr_start = t * self.tr_per_frame
        tr_end = (t + 1) * self.tr_per_frame
        coord_t = sp.to_device(self.coord[tr_start:tr_end], self.device)
        dcf_t = sp.to_device(self.dcf[tr_start:tr_end], self.device)
        ksp_t = sp.to_device(self.ksp[:, tr_start:tr_end], self.device)

        # Data consistency.
        e_t = 0
        loss_t = 0
        for c in range(self.C):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc
       
        e_t=torch.as_tensor(e_t,device='cuda') 
        flow_adj0=self.get_adj_field(t)
        flow_adj1=self.get_adj_field_refine(t)
        e_t=self.warp(flow_adj0,e_t)
        e_t=self.warp(flow_adj0,e_t)
        e_t=self.warp(flow_adj1,e_t)
        
        e_t = cupy.asarray(e_t.detach().cpu().numpy())
     

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient.
        for j in range(self.J):
            lamda_j = self.lamda * self.G[j]

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # L gradient.
            #Minv*A^H*
            g_L_j = self.B[j].H(e_t)
            g_L_j *= self.xp.conj(self.R[j][t])
            g_L_j += lamda_j / self.T * self.L[j]
            g_L_j *= self.T

            # R gradient.
            g_R_jt = self.B[j].H(e_t)
            g_R_jt *= self.xp.conj(self.L[j])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.D, 0), keepdims=True)
            g_R_jt += lamda_j * self.R[j][t]

      
            # Add.
            self.L[j] -= self.alpha * self.beta**(self.epoch // self.decay_epoch) * g_L_j
            self.R[j][t] -= self.alpha * g_R_jt

        loss_t /= 2
        return loss_t