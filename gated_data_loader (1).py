#load_MRI_Raw,convert_UTE,gate
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



def gating(name,normalization,RO):
    
    kdata, coords, dcfs, resps, tr=convert_ute(name,compress_coils=False)
    coords=autofova(coords, dcfs,kdata,device=None,
               thresh=0.1, scale=1, oversample=2)

    #RO=kdata.shape[2]
    ksp=kdata[:,:,:RO]
    coords=coords[:,:RO]
    dcfs=dcfs[:,:RO]

    mr_raw=load_MRI_raw(h5_filename='MRI_Raw.h5', compress_coils=False)
    import sigpy.mri as mr
    mps=mr.app.JsenseRecon(ksp, coord=coords, weights=dcfs, device=0).run()
    if normalization==1:
        dcfs=normalize(mps,coords,dcfs,ksp,kdata.shape[1])
        ksp=kspace_scaling(mps,dcfs,coords,ksp)
    else:
        ksp=ksp
        dcfs=dcfs

    mr_raw1=mr_raw
    mr_raw1.kdata[0]=ksp
    mr_raw1.coords[0]=coords
    mr_raw1.dcf[0]=dcfs
    mr_raw1.resp[0]=resps
    mr_raw.kdata[0]=mr_raw1.kdata[0] #np.reshape(mr_raw1.kdata[0],[13,-1])
    mr_raw.coords[0]=mr_raw1.coords[0] #np.reshape(mr_raw1.coords[0],[-1,3])
    mr_raw.dcf[0]=mr_raw1.dcf[0] #np.reshape(mr_raw1.dcf[0],[-1])
    mr_raw.resp[0]=resps
    mr_rawg=gate_kspace(mri_raw=mr_raw, num_frames=6, gate_type='resp', discrete_gates=False, ecg_delay=300e-3)
    return mr_rawg,mps
class MotionResolvedRecon(object):
    def __init__(self, kspg, coordg, dcfg,index_all, mps,  B,
                 lamda=1e-6, alpha=1, beta=0.5,
                 max_power_iter=10, max_iter=120,
                 device=0, margin=10,
                 coil_batch_size=None, comm=None, show_pbar=True, **kwargs):
        self.B = B
        self.index_all=index_all
        self.C = mps.shape[0]
        self.mps = mps
        #print(self.mps.shape)
        self.device = sp.Device(device)
        self.xp = xp
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.max_iter = max_iter
        self.max_power_iter = max_power_iter
        self.comm = None
        self.show_pbar=True
       # if comm is not None:
        #    self.show_pbar = show_pbar and comm.rank == 0

        self.img_shape = mps.shape[1:]
       # print(self.img_shape)

       # bins = np.percentile(resp, np.linspace(0 + margin, 100 - margin, B + 1))
        self.bksp=[]
        self.bcoord=[]
        self.bdcf=[]
        print('bin')
        for i in range(self.B):
            self.bksp.append(cupy.array(kspg[i]))
            self.bcoord.append(cupy.array(coordg[i]))
            self.bdcf.append(cupy.array(dcfg[i]))
       
      #  self._normalize()

    def _normalize(self):
        # Normalize using first phase.
        #with device:
            mrimg_adj = 0
            for c in range(self.C):
                device=0
                mrimg_c = sp.nufft_adjoint(
                    self.bksp[0][c] * self.bdcf[0], self.bcoord[0],
                    self.img_shape)
                #print(mrimg_c.shape)
               # T=self.xp.conj(sp.to_device(self.mps[c], device))
               # print(T.shape)
                mrimg_c *= self.xp.conj(sp.to_device(self.mps[c], device))
                mrimg_adj += mrimg_c

           # if comm is not None:
           #     comm.allreduce(mrimg_adj)

            # Get maximum eigenvalue.
            F = sp.linop.NUFFT(self.img_shape, self.bcoord[0])
            W = sp.linop.Multiply(F.oshape, self.bdcf[0])
            max_eig = sp.app.MaxEig(F.H * W * F,
                                    max_iter=self.max_power_iter,
                                    dtype=cupy.complex64, device=device,
                                    show_pbar=self.show_pbar).run()

            # Normalize
           # self._normalize
            self.alpha /= max_eig
            self.lamda *= max_eig * self.xp.abs(mrimg_adj).max().item()
            print(self.alpha)
            print(self.lamda)
            
            
        

    def gradf(self, mrimg):
        out = self.xp.zeros_like(mrimg)
        for b in range(self.B):
           # print(b)
            for c in range(13):
              
                mps_c = sp.to_device(self.mps[c], self.device)
                out[b] += sp.nufft_adjoint(
                    self.bdcf[b] * (sp.nufft(mrimg[b] * mps_c, self.bcoord[b])
                                    - self.bksp[b][c]),
                    self.bcoord[b],
                    oshape=mrimg.shape[1:]) * self.xp.conj(mps_c)

        if self.comm is not None:
            self.comm.allreduce(out)

        eps = 1e-31
        for b in range(self.B):
            if b > 0:
                diff = mrimg[b] - mrimg[b - 1]
                sp.axpy(out[b], self.lamda, diff / (self.xp.abs(diff) + eps))

            if b < self.B - 1:
                diff = mrimg[b] - mrimg[b + 1]
                sp.axpy(out[b], self.lamda, diff / (self.xp.abs(diff) + eps))
        import imageio
       
        return out

    def run(self):
        done = False
        while not done:
            try:
                with tqdm(total=self.max_iter, desc='MotionResolvedRecon',
                          disable=not self.show_pbar) as pbar:
                    with self.device:
                        mrimg = self.xp.zeros([self.B,self.mps.shape[1],self.mps.shape[2],self.mps.shape[3]],dtype=self.mps.dtype)
                      
                        for it in range(self.max_iter):
                            g = self.gradf(mrimg)
                            sp.axpy(mrimg, -self.alpha, g)
                            import imageio
                            imageio.mimsave('./powers3a.gif', [np.abs(np.array(mrimg[i][:,35,:].get()))*1e15 for i in range(6)], fps=4)

                            gnorm = self.xp.linalg.norm(g.ravel()).item()
                            print(gnorm)
                            if np.isnan(gnorm) or np.isinf(gnorm):
                                raise OverflowError('LowRankRecon diverges.')

                            pbar.set_postfix(gnorm=gnorm)
                            pbar.update()

                        done = True
            except OverflowError:
                self.alpha *= self.beta

        return mrimg,self.bksp,self.bcoord,self.bdcf



def for_field_solver(deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,im_testa,mps,iter,block_torch_low,block_torch_high,ishape_low,ishape_high,T1,T2,interp,new_res,old_res,weight_MSE=1e-1):
     #from utils_reg1 import flows,_updateb,f
     from torch.utils import checkpoint
     import torch
     import numpy as np
     import random
     deform=[deformL_param_for[0],deformL_param_for[1],deformL_param_for[2],deformR_param_for[0],deformR_param_for[1],deformR_param_for[2]]
     optimizer2=torch.optim.Adam([deform[i] for i in range(6)],lr=.01) #, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 

   #  optimizer3=torch.optim.LBFGS([deformR_param_for[i] for i in range(3)],lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) 
     deform_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
     image_still=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
     image_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
     P=torch.ones([40,1])
     mps=torch.from_numpy(mps).cpu()
     def closure_for():
       # from utils_reg1 import flows,_updateb,f
        from torch.utils import checkpoint
        import torch
        import numpy as np
        import random
        with torch.no_grad():
            loss_for=0
            loss_grad0=0
            loss_det=0
            loss_rev1=0

            for j in K:
       
                count=0
                lossz=np.zeros([20])

             
                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
                
              

                flowa=deforma.cuda()
                new_res=mps
                flowa=torch.nn.functional.interpolate(flowa.unsqueeze(0), size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
                flowa_inv=inverse_field(flowa,mps)
               # deforma_inv=flows(deformL_param_for,deformR_param_for,j-T1,block_torch,ishape)
                flowa_inv=deforma_inv.cuda()
                flowa_inv=torch.nn.functional.interpolate(flowa_inv.unsqueeze(0), size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
              
                
                im_test=(torch.from_numpy(im_testa).cuda().unsqueeze(0)) #LR(Ltorch,Rtorch,j).cuda().unsqueeze(0)
                all0=torch.abs(im_test).max()
                im_test=im_test
               
              #  spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                vectors=[]
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors = [ torch.arange(0, s) for s in size ] 
                #vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               # for i in range(len(shape)):
               #     new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                #new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locsa = new_locs[..., [0,1,2]]
               # ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
               # ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                ima_real=grid_pull(torch.squeeze(torch.real(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                ima_imag=grid_pull(torch.squeeze(torch.imag(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)

               # ima_real=_compute_warped_image_multiNC_3d(torch.real((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
               ## ima_imag=_compute_warped_image_multiNC_3d(torch.imag((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
                im_out=torch.complex(ima_real,ima_imag)
              
               
              #  spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs1=grid.cuda()+flowa_inv.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               # for i in range(len(shape)):
               #     new_locs1[:,i,...] = 2*(new_locs1[:,i,...]/(shape[i]-1) - 0.5)
                new_locs1 = new_locs1.permute(0, 2, 3, 4, 1) 
                new_locs1 = new_locs1[..., [0,1,2]]
                im_inv_real=grid_pull(torch.squeeze(torch.real(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                im_inv_imag=grid_pull(torch.squeeze(torch.imag(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
               # im_inv_real=torch.nn.functional.grid_sample(torch.real(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
               # im_inv_imag=torch.nn.functional.grid_sample(torch.imag(im_out), new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                im_inv=torch.complex(im_inv_real,im_inv_imag)
           
                diff=im_inv-im_test
                loss_rev1=loss_rev1+torch.norm(diff,2)**2
              
                loss_grad0=loss_grad0+torch.utils.checkpoint.checkpoint(f.loss,flowa_inv.cuda())
            
                loss_L=0
                loss_R=0
                loss_L0=0
                loss_R0=0
               # for i in range(5):
                 #   loss_L=loss_L+torch.norm(deformL_param_adj[i],'fro')**2
                 #   loss_R=loss_R+torch.norm(deformR_param_adj[i][:,:,:,1:]-deformR_param_adj[i][:,:,:,:-1],'fro')**2
               # loss_L0=loss_L0+torch.norm(Ltorch[i],'fro')**2
               # loss_R0=loss_R0+torch.norm(Rtorch[i],'fro')**2
      
            loss=loss_rev1*weight_MSE+loss_grad0 #+loss_grad0*30 #+loss_R*1e-6 #loss_L0*1e-8+loss_R0*1e-8 
            return loss

     for io in range(iter):

            
            optimizer2.zero_grad()
          #  optimizer3.zero_grad()
       
            loss_grad0=0
            loss_tot=0
            loss_for=0
            loss_rev=0
            loss_for=0
            loss_grad0=0
            flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
            count=0
            lossz=np.zeros([20])
          
            K=random.sample(range(T1,T2), T2-T1)
            for j in K:
                loss_rev1=0

                print(j)
               # optimizer0.zero_grad()


                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch_low,ishape_low)
                print(deforma.shape)
                flowa=deforma.cuda()
                new_res=mps
                flowa=torch.nn.functional.interpolate(flowa, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
                flowa[:,0]=flowa[:,0]*mps1.shape[1]/mps0.shape[1]*3 #/94 #new_res.shape[1]/old_res.shape[1]
                flowa[:,1]=flowa[:,1]*mps1.shape[2]/mps0.shape[2]*3 #3*240/110 #new_res.shape[2]/old_res.shape[2]
                flowa[:,2]=flowa[:,2]*mps1.shape[3]/mps0.shape[3]*3 #3*480/220 #new_res.shape[3]/old_res.shape[3]
                #flowa_inv=inverse_field(flowa,mps)
                deforma_inv=flows(deformL_param_for,deformR_param_for,j-T1,block_torch_high,ishape_high)
                flowa_inv=deforma_inv.cuda()
                flowa_inv=torch.nn.functional.interpolate(flowa_inv, size=[new_res.shape[1],new_res.shape[2],new_res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)
               
                loss_for=0


                im_test=(torch.from_numpy(im_testa).cuda().unsqueeze(0))
                all0=torch.abs(im_test).max()

                im_test=im_test

                spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs=grid.cuda()+flowa.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               # for i in range(len(shape)):
               #     new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
                new_locs = new_locs.permute(0, 2, 3, 4, 1) 
                new_locsa = new_locs[..., [0,1,2]]
                ima_real=grid_pull(torch.squeeze(torch.real(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                ima_imag=grid_pull(torch.squeeze(torch.imag(im_test)).cuda(),torch.squeeze(new_locsa).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
               # ima_real=torch.nn.functional.grid_sample(torch.real(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
               # ima_imag=torch.nn.functional.grid_sample(torch.imag(im_test), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
                

               # ima_real=_compute_warped_image_multiNC_3d(torch.real((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
               ## ima_imag=_compute_warped_image_multiNC_3d(torch.imag((im_test)), flowa, spacing, spline_order=1,zero_boundary=False,use_01_input=True)
                im_out=torch.complex(ima_real,ima_imag)
               
                #ima=torch.nn.functional.grid_sample(im_test, new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
              

                spacing=(1/mps.shape[1],1/mps.shape[2],1/mps.shape[3])
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
                size=shape
                vectors=[]
                vectors = [ torch.arange(0, s) for s in size ] 
                grids = torch.meshgrid(vectors) 
                grid  = torch.stack(grids) # y, x, z
                grid  = torch.unsqueeze(grid, 0)  #add batch
                grid = grid.type(torch.FloatTensor)
                new_locs1=grid.cuda()+flowa_inv.cuda()
                shape=(mps.shape[1],mps.shape[2],mps.shape[3])
               ## for i in range(len(shape)):
                    #new_locs1[:,i,...] = 2*(new_locs1[:,i,...]/(shape[i]-1) - 0.5)
                new_locs1 = new_locs1.permute(0, 2, 3, 4, 1) 
                new_locs1 = new_locs1[..., [0,1,2]]
                im_inv_real=grid_pull(torch.squeeze(torch.real(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                im_inv_imag=grid_pull(torch.squeeze(torch.imag(im_out)).cuda(),torch.squeeze(new_locs1).cuda(),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
                im_inv=torch.complex(im_inv_real,im_inv_imag)
                #im_inv=torch.nn.functional.grid_sample(im_out, new_locs1, mode='bilinear', padding_mode='reflection', align_corners=True)
                diff=im_inv-im_test
                loss_rev1=loss_rev1+torch.norm(diff,2)**2
               # loss_self1=torch.nn.MSELoss()
                #loss_rev1=loss_self1(torch.squeeze(im_inv),torch.squeeze(torch.abs(im_test)))*10


                if j>=T1 and j<T2+500:
                    deform_look[j-T1]=np.abs(flowa_inv[:,0,:,100,:].detach().cpu().numpy())
                       # image_rev[j]=np.abs(im_rev.detach8.cpu().numpy())
                    image_still[j-T1]=np.abs(im_inv[:,100,:].detach().cpu().numpy())
                    image_look[j-T1]=np.abs(im_out[:,100,:].detach().cpu().numpy())
                   # image_still[j-50]=np.abs(im_test.detach().cpu().numpy())

                loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa_inv.cuda())

                lo=0
                print(loss_rev1*.5)
                print(loss_grad0*.5)
                
                loss=loss_rev1*1+loss_grad0*1
                print(loss)

                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()



          #  optimizer3.step(closure_for)

            import imageio
            imageio.mimsave('image.gif', [np.abs(image_look[i,:180,:])*1e15 for i in range(110)], fps=4)
            imageio.mimsave('deform.gif', [np.abs(deform_look[i,:180,:])*1e15 for i in range(110)], fps=4)
            imageio.mimsave('image_still.gif', [np.abs(image_still[i,:180,:])*1e15 for i in range(110)], fps=4)
     return deformL_param_for,deformR_param_for

def gen_MSLR(T,rank,block_size_adj,block_size_for,scale,mps):

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
  
        temp0=1e3*temp0/torch.sum(torch.square(torch.abs(temp0)))**0.5
        print(temp0.max())
        deformL_param_adj.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
       # tempa=torch.rand([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],dtype=torch.float16,device='cuda')
        deformR_param_adj.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
        deformL_param_for.append(torch.nn.parameter.Parameter(temp0,requires_grad=True))
        deformR_param_for.append(torch.nn.parameter.Parameter(torch.zeros([3,int(block_final.ishape[0]*block_final.ishape[1]*block_final.ishape[2]),rank,T],device='cuda'),requires_grad=True))
    return deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,block_torch1,ishape1a

def gen(block_torcha,deformL_param,deformR_param,ishape0,ishape1,ishape2,ishape3,ishape4,ishape5,jo):
    jb=int(jo[0])
   # print(jb)
    deform_patch_adj=torch.matmul(deformL_param,deformR_param[:,:,:,jb:jb+1])
    deform_patch_adj=torch.reshape(deform_patch_adj,[3,int(ishape0[0]),int(ishape1[0]),int(ishape2[0]),int(ishape3[0]),int(ishape4[0]),int(ishape5[0])])
    deformx_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[0])).unsqueeze(0)
    deformy_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[1])).unsqueeze(0)
    deformz_adj=torch.squeeze(block_torcha.apply(deform_patch_adj[2])).unsqueeze(0)
   # deform_adj.append(torch.cat([deformx_adj,deformy_adj,deformz_adj],axis=0))
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
        torch.cuda.empty_cache()
        loss=0
      
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(),1.25,2)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
       
        torch.cuda.empty_cache()
       
        torch.cuda.empty_cache()
        e_tc=F_torch.apply(M_t.cuda())
       # print(e_tc.shape)
        #Ptc=torch.reshape(Ptc,[-1])
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1]) #*torch.reshape(Ptc,[-1])
       # e_tca=(e_tca/torch.abs(e_tca).max())*torch.abs(ksp).max()
       # e_tc_update=e_tca
      #  print(torch.abs(e_tca).max())
      #  print(torch.abs(ksp).max())
        #loss_self1=torch.nn.MSELoss()
        torch.cuda.empty_cache()
        #ksp_real=torch.real(ksp)*dcf
        #ksp_imag=torch.imag(ksp)*dcf
        #e_tca_real=torch.real(e_tca)*dcf
        #e_tca_imag=torch.imag(e_tca)*dcf
       # ksp=torch.complex(ksp_real,ksp_imag)
       # e_tca=torch.complex(e_tca_real,e_tca_imag)
        res=(ksp-e_tca)*(dcf)**0.5 #**0.5 #**0.5
        res=torch.reshape(res,[1,-1])
        lossb=(torch.linalg.norm(res,2))**2 #**2 #**2 #torch.abs(torch.sum(((ksp-e_tca))))
        
        #loss=(torch.norm(resk)) #/torch.norm(ksp,2)+torch.norm(resk,1)/torch.norm(ksp,1) #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
       
        
       # torch.cuda.empty_cache()
       # loss=torch.norm(resk,1) #*index_all/index_max  #*index_all/ksp.shape[0]
        #print(loss)
        return lossb

def _updateb(img_t,ksp_t,dcf_t,coord_t,mpsa): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  
# Data consistency.
  loss_t=0
  #print(mpsa.shape)
  #print(img_t.shape)
  #n=int(cooo[0])
  for c in range(mpsa.shape[0]):
       
    
    loss_t=loss_t+torch.utils.checkpoint.checkpoint(calculate_sense0,torch.cat([torch.reshape(torch.real(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1]),torch.reshape(torch.imag(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1])],axis=3),torch.reshape(ksp_t[c],[-1]),mpsa[c],torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
   # m=mpsa[c].detach().cpu().numpy()
   # torch.cuda.empty_cache()
  #del img_t
  #torch.cuda.empty_cache()
  
  return loss_t
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
    L_blocks,R_blocks,B = MultiScaleLowRankRecon(kspa[:,:,:], coorda[:,:], dcfa[:,:], mps, T, lamda, device=device, blk_widths=blk_widths).run()

    mpsa=mps

    im_test=np.zeros([1,mpsa.shape[1],mpsa.shape[2],mpsa.shape[3]],dtype=np.complex64)
    temp=0
    for i in range(1):
        for j in range(3):
            temp=temp+B[j](L_blocks[j]*R_blocks[j][i])
        im_test[i]=temp.get()

    im_testa=im_test
    return im_testa,mps,kspa,coorda,dcfa


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
        ima_real=grid_pull(torch.squeeze(torch.real(img)),torch.squeeze(new_locsa),interpolation=1,bound='zero',extrapolate=False,prefilter=False)
        ima_imag=grid_pull(torch.squeeze(torch.imag(img)),torch.squeeze(new_locsa),interpolation=1,bound='zero',extrapolate=False,prefilter=False)
        im_out=torch.complex(ima_real,ima_imag)
    else:
        im_out=grid_pull(torch.squeeze((img)),torch.squeeze(new_locsa),interpolation=3,bound='zero',extrapolate=False,prefilter=True)
     #ima_real=torch.nn.functional.grid_sample(torch.real(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
     #ima_imag=torch.nn.functional.grid_sample(torch.imag(img), new_locs, mode='bilinear', padding_mode='reflection', align_corners=True)
    #im_out=torch.complex(ima_real,ima_imag)
    return im_out


def adj_field_solver(deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,im_template,ksp,coord,dcf,mps,iter_adj,RO,block_torch,ishape,T1,T2,interp,res,spokes_per_bin,weight_dc,weight_smoother): #,conL,conR,block_torchcon,ishapecon):
   # from utils_reg1 import flows,_updateb,f
    from torch.utils import checkpoint
    import torch
    import numpy as np
    import random
    scaler = torch.cuda.amp.GradScaler()
    #readout images and deformation fields during training
    deform_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_still=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    image_look=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    im_tester=np.zeros([T2-T1,mps.shape[1],mps.shape[3]])
    P=torch.ones([40,1])
    mps=torch.from_numpy(mps).cpu()
   # deform=[]
    deform=[deformL_param_adj[0],deformR_param_adj[0],deformL_param_adj[1],deformR_param_adj[1],deformL_param_adj[2],deformR_param_adj[2]] #,conL[0],conR[0],conL[1],conR[1]] #,deformR_param_adj[1],deformR_param_adj[2]]
    import torch_optimizer as optim
   # optimizer0=torch.optim.Adam([deform[i] for i in range(6)],lr=.01) # , max_iter=1, max_eval=None, tolerance_grad=1e-200, tolerance_change=1e-400, history_size=100, line_search_fn='backtracking') #[deformL_param_adj[i] for i in range(3)],lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None)
   # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer0, gamma=0.9)
   # deform[0].requires_grad=True
   # deform[1].requires_grad=True
   # deform[2].requires_grad=True
   # deform[3].requires_grad=True
   # deform[4].requires_grad=True
   # deform[5].requires_grad=True
   # optimizer0=torch.optim.Adam([deform[i] for i in range(6)], lr=.01) #, max_eval=1, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn='strong_wolfe') 
   # optimizer0=torch.optim.Adam([deform[i] for i in range(4)],.01) #, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn=None) #[deformL_param_adj[i] for i in range(3)],lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=10, line_search_fn=None)
    optimizer0=torch.optim.Adam([deform[i] for i in range(6)],.001)
 #  optimizer1=torch.optim.Adam([deform[i] for i in range(12,16)],.01)
  #  optimizer0=torch.optim.Adam([deform[i] for i in range(6,10)],lr=.001) #, max_iter=1, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=1, line_search_fn='strong_wolfe') 
   
    count=0
    count0=0
    L=np.zeros([1])
    for io in range(iter_adj):
          #  optimizer0.zero_grad()

           # loss_grad0=0
           # loss_tot=0
           # loss_for=0
           # loss_rev=0
           # loss_for=0
           # loss_grad0=0
            flowa=0 #torch.zeros([10,3,mps.shape[1],mps.shape[2],mps.shape[3]])
            count=0
            lossz=np.zeros([20])

         
            K=random.sample(range(T1,T2), T2-T1)
            print(K)
            for j in K:
                jo=j
                #j=91
                print(count)
               # print(j)
               
                count0=count%1
                L[count0]=np.int(j)
               # print(count)
               # print(j)
                               
              #  print(j)
               # optimizer0.zero_grad()


                deforma=flows(deformL_param_adj,deformR_param_adj,j-T1,block_torch,ishape)
               # con=cons(conL,conR,j-T1,block_torchcon,ishapecon)
               # print(con.shape)
               # all0=torch.abs(con).max()
              #  con_real=torch.squeeze(torch.nn.functional.interpolate(torch.real(con.unsqueeze(0)), size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None))
              #  con_imag=torch.squeeze(torch.nn.functional.interpolate(torch.imag(con.unsqueeze(0)), size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None))
               # con=torch.complex(con_real,con_imag)
              #  all1=torch.abs(con).max()
              
                flowa=deforma
                flowa=torch.nn.functional.interpolate(flowa, size=[res.shape[1],res.shape[2],res.shape[3]], scale_factor=None, mode='trilinear', align_corners=True, recompute_scale_factor=None)*interp
                print(flowa.shape)
                #flowa[:,0]=flowa[:,0]*inter
                #flowa[:,1]=flowa[:,1]*3
                #flowa[:,2]=flowa[:,2]*3
               # flowa_rev=flows(deformL_param_for,deformR_param_for,j-T1,block_torch,ishape)

               # loss_for=0


                im_test=torch.from_numpy(im_template).cuda().unsqueeze(0)
                all0=torch.abs(im_test).max()
                con0=im_test #con
                im_out=warp1(con0/all0,flowa,mps,complex=True)
                print('im_out')
                print(im_out.shape)
               # flowa_inv=inverse_field(flowa)
               # im_rev=warp0(im_out,flowa_inv,complex=True)
                
                tr_per_frame=spokes_per_bin
                tr_start=tr_per_frame*(j)
                tr_end=tr_per_frame*(j+1)
                ksp_ta=torch.from_numpy(ksp[:,tr_start:tr_end,:RO]).cuda()/all0
                coord_t=torch.from_numpy(coord[tr_start:tr_end,:RO]).cuda()
                dcf_t=torch.from_numpy(dcf[tr_start:tr_end,:RO]).cuda()
                Pt=(P[:,tr_start:tr_end]).cuda()
               # from cupyx.scipy import ndimage
               # testing=cupyx.scipy.ndimage.map_coordinates(cupy.abs(cupy.asarray(im_test[0].detach().cpu().numpy())), cupy.asarray(new_locs.detach().cpu().numpy()), output=None, order=3, mode='reflect', cval=0.0, prefilter=True)

               # flowa=torch.squeeze(flowa)
                print(flowa.shape)
                if j>=T1 and j<T1+50:
                   # im_out1=torch.nn.functional.interpolate(torch.abs(im_out), size=[210,123,219], scale_factor=None, mode='trilinear', align_corners=False, recompute_scale_factor=None) #+torch.nn.functional.interpolate(deform_fulla[j:j+1], size=[mps.shape[1],mps.shape[2],mps.shape[3]], scale_factor=None, mode='trilinear', align_corners=None, recompute_scale_factor=None)
                    deform_look[j-T1]=(flowa[:,0,:,30,:].detach().cpu().numpy())
                       # image_rev[j]=np.abs(im_rev.detach().cpu().numpy())
                   # image_still[j-50]=np.abs(im_inv[0,0,:,:,35].detach().cpu().numpy())
                    image_look[j-T1]=np.abs(im_out[:,30,:].detach().cpu().numpy())
                 #   im_tester[j-T1]=np.squeeze(im_no_con[:,30,:].detach().cpu().numpy())
                   # con=torch.squeeze(con)
                    im_test=torch.squeeze(im_test)
                   # image_still[j-T1]=np.abs((con[:,:,120].detach().cpu().numpy())+(im_test[:,:,120].detach().cpu().numpy()))

                loss_grad0=torch.utils.checkpoint.checkpoint(f.loss,flowa)
                lo=0
                cooo=torch.ones([1])*lo
                print('im_out')
                print(im_out.shape)
                print('mps')
                print(mps.shape)
                loss_for=torch.utils.checkpoint.checkpoint(_updateb,im_out.unsqueeze(0),ksp_ta,dcf_t,coord_t,mps) #+loss_grad+(torch.sum(deformL2a**2)+torch.sum(deformR2a**2))*1e-9+(torch.sum(deformL4a**2)+torch.sum(deformR4a**2))*1e-9+(torch.sum(deformL8a**2)+torch.sum(deformR8a**2))*1e-9
               #Q print(loss_for)
                loss_L=0
                loss_R=0
                loss_L0=0
                loss_R0=0
                for i in range(3):
                   # loss_L=loss_L+torch.norm(conL[i],'fro')**2
                   # loss_R=loss_R+torch.norm(conR[i][:,:,:,:])**2 #-conR[i][:,:,:,:-1],'fro')**2
                   
                    loss_L0=loss_L0+torch.norm(deformL_param_adj[i],'fro')**2
                    loss_R0=loss_R0+torch.norm(deformR_param_adj[i][:,:,:,1:]-deformR_param_adj[i][:,:,:,:-1],'fro')**2
               # np.save('data/lung/deformL_param_adj0.npy',deformL_param_adj[0].detach().cpu().numpy())
               # np.save('data/lung/deformL_param_adj1.npy',deformL_param_adj[1].detach().cpu().numpy())
               # np.save('p1/deformL_param_adj2.npy',deformL_param_adj[2].detach().cpu().numpy())

              #  np.save('data/lung/deformR_param_adj0.npy',deformR_param_adj[0].detach().cpu().numpy())
              ##  np.save('data/lung/deformR_param_adj1.npy',deformR_param_adj[1].detach().cpu().numpy())
                #np.save('data/lung/deformR_param_adj2.npy',deformR_param_adj[2].detach().cpu().numpy())
                count=count+1
               # torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0,
               # torch.nn.utils.clip_grad_value_([deform[i] for i in range(3)], 1e-1)
                print('loss')
                print(loss_for*weight_dc)
                print(loss_grad0*weight_smoother)
                print(loss_L*1e-5)
               # print(loss_rev)
                loss=loss_for*weight_dc/1+loss_grad0*weight_smoother+loss_L0*1e-7+loss_R0*1e-7 #+loss_L0*1e-7+loss_R0*1e-5 #+loss_L*1e-5+loss_R*1e-5 #+loss_L0*1e-7+loss_R0*1e-7 #+loss_L0*1e-9+loss_R0*1e-7 #+loss_L*1e-8+loss_R*1e-8 #+loss_L*1e-6 #loss_grad0*weight_smoother #+loss_R*1e-4 #+loss_L*1e-8 #+loss_R*1e-6 #+loss_L*1e-9+loss_R*1e-9 #+loss_R*1e-6
                (loss).backward()
                #loss.backward()
                (optimizer0).step()
               # optimizer1.step()
                optimizer0.zero_grad()
              #  optimizer1.zero_grad()
               # scalar.update()
               
               
               # loss_for=0
               # loss_grad0=0
                   
           # loss_for=0
           #         loss_grad0=0
          #  scheduler.step()      
           # optimizer1.step(closure_adj)
            import imageio
            imageio.mimsave('image.gif', [np.abs(image_look[i,:,:])*1e15 for i in range(50)], fps=5)
            imageio.mimsave('deform.gif', [np.abs(deform_look[i,:,:])*1e15 for i in range(50)], fps=5)
            imageio.mimsave('image_tester.gif', [np.abs(im_tester[i,:,:])*1e15 for i in range(50)], fps=5)
           # imageio.mimsave('image_still.gif', [np.abs(image_still[i,:,:])*1e15 for i in range(250)], fps=10)
    return deformL_param_adj,deformR_param_adj,image_look

def calculate_sense0b(ksp,mps_c,coord_t,dcf):
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], cupy.array(coord_t), oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
       # print(ksp.shape)
       # print(dcf.shape)
        
       
       
        resk=(ksp)*dcf
        resk_real=torch.real(torch.squeeze(resk))
        resk_imag=torch.imag(torch.squeeze(resk))
       # print(resk_imag.shape)
        resk_real=resk_real.unsqueeze(axis=1)
        resk_imag=resk_imag.unsqueeze(axis=1)
        resk_com=torch.cat([resk_real,resk_imag],axis=1)
        g=FH_torch.apply(resk_com)
       # print(g.shape)
        g=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c)
        ksp=ksp.detach()
        
       
        #loss=torch.norm(resk)**2 #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
        return g

def init_im(ksp_t,dcf_t,coord_t,mps): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  #img_t = 0
 # for j in range(self.J):
  #      img_t += self.B[j](self.L[j] * self.R[j][t])
  #img_t= torch.as_tensor(im , device='cuda') 
  
 


  



# Data consistency.
  loss_t=0
  #diff=torch.zeros([mps.shape[1],mps.shape[2],mps.shape[3],2]).
#  div0=torch.zeros([mpsa.shape[0]])
  loss_tot=0
  g=0
  for c in range(mps.shape[0]):
   # print(c)
 
   
   

   # torch.unsqueeze(diff0, dim)

    g=g+torch.utils.checkpoint.checkpoint(calculate_sense0b,torch.reshape(ksp_t[c],[-1]),mps[c].cuda(),torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
 
 
  
 # diff=diff.detach()
  #dcf_t=dcf_t.detach()
   




  return g

class MRI_Raw:
    Num_Encodings = 0
    Num_Coils = 0
    trajectory_type = None
    dft_needed = None
    Num_Frames = None
    coords = None
    time = None
    ecg = None
    prep = None
    resp = None
    dcf = None
    kdata = None
    target_image_size = [256, 256, 64]
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
def load_MRI_raw(h5_filename=None, max_coils=None, compress_coils=False):
    with h5py.File(h5_filename, 'r') as hf:

      
        Num_Encodings = np.squeeze(hf['Kdata'].attrs['Num_Encodings'])
        Num_Coils = np.squeeze(hf['Kdata'].attrs['Num_Coils'])
        Num_Frames = np.squeeze(hf['Kdata'].attrs['Num_Frames'])

        trajectory_type = [np.squeeze(hf['Kdata'].attrs['trajectory_typeX']),
                           np.squeeze(hf['Kdata'].attrs['trajectory_typeY']),
                           np.squeeze(hf['Kdata'].attrs['trajectory_typeZ'])]

        dft_needed = [np.squeeze(hf['Kdata'].attrs['dft_neededX']), np.squeeze(hf['Kdata'].attrs['dft_neededY']),
                      np.squeeze(hf['Kdata'].attrs['dft_neededZ'])]

        

       # except Exception:
       #     logging.info('Missing header data')
       #     pass

       # if max_coils is None:
       #     Num_Coils = 32 #min( max_coils, Num_Coils)

        # Get the MRI Raw structure setup
        mri_raw = MRI_Raw()
        mri_raw.Num_Coils = int(Num_Coils)
        mri_raw.Num_Encodings = int(Num_Encodings)
        print(Num_Encodings)
        mri_raw.dft_needed = tuple(dft_needed)
        mri_raw.trajectory_type = tuple(trajectory_type)

        # List array
        mri_raw.coords = []
        mri_raw.dcf = []
        mri_raw.kdata = []
        mri_raw.time = []
        mri_raw.prep = []
        mri_raw.ecg = []
        mri_raw.resp = []
        print('2')
        for encode in range(Num_Encodings):
            print(encode)
           

            # Get the coordinates
            coord = []
            for i in ['Z', 'Y', 'X']:
                print(i)
                coord.append(np.array(hf['Kdata'][f'K{i}_E{encode}'])) #.flatten()
            coord = np.stack(coord, axis=-1)

            dcf = np.array(hf['Kdata'][f'KW_E{encode}'])

            # Load time data
            try:
                time_readout = np.array(hf['Gating']['time'])
            except Exception:
                time_readout = np.array(hf['Gating'][f'TIME_E{encode}'])

            try:
                ecg_readout = np.array(hf['Gating']['ecg'])
            except Exception:
                ecg_readout = np.array(hf['Gating'][f'ECG_E{encode}'])

          

            try:
                prep_readout = np.array(hf['Gating']['prep'])
            except Exception:
                prep_readout = np.array(hf['Gating'][f'PREP_E{encode}'])

            try:
                resp_readout = np.array(hf['Gating']['resp'])
            except Exception:
                resp_readout = np.array(hf['Gating'][f'RESP_E{encode}'])


            # This assigns the same time to each point in the readout
            time_readout = np.expand_dims(time_readout, -1)
            ecg_readout = np.expand_dims(ecg_readout, -1)
            resp_readout = np.expand_dims(resp_readout, -1)
            prep_readout = np.expand_dims(prep_readout, -1)

            time = np.tile(time_readout,(1, 1, dcf.shape[2]))
            resp = np.tile(resp_readout,(1, 1, dcf.shape[2]))
            ecg = np.tile(ecg_readout, (1, 1, dcf.shape[2]))
            prep = np.tile(prep_readout, (1, 1, dcf.shape[2]))

            prep = prep.flatten()
            resp = resp.flatten()
            ecg = ecg.flatten()
            dcf = dcf
            time = time.flatten()
            print('a')
            # Get k-space
            ksp = []
            for c in range(Num_Coils):
                print(c)
               

                k = hf['Kdata'][f'KData_E{encode}_C{c}']
                print('all')
                ksp.append(np.array(k['real'] + 1j * k['imag'])) #flatten()
                print('better')
            #ksp = np.stack(ksp, axis=0)
            

            # Append to list
            print('b')
            mri_raw.coords.append(np.squeeze(coord))
            mri_raw.dcf.append(np.squeeze(dcf))
            mri_raw.kdata.append(np.squeeze(ksp))
            mri_raw.time.append(time)
            mri_raw.prep.append(prep)
            mri_raw.ecg.append(ecg)
            mri_raw.resp.append(resp)

            # Log the data
       
        '''
        try:
            noise = hf['Kdata']['Noise']['real'] + 1j * hf['Kdata']['Noise']['imag']
            logging.info('Whitening ksp.')
            cov = mr.util.get_cov(noise)
            ksp = mr.util.whiten(ksp, cov)
        except Exception:
            ksp /= np.abs(ksp).max()
            logging.info('No noise data.')
            pass
        '''
        print('3')
        # Scale k-space to max 1
        kdata_max = [ np.abs(ksp).max() for ksp in mri_raw.kdata]
        print(f'Max kdata {kdata_max}')
        kdata_max = np.max(np.array(kdata_max))
        for ksp in mri_raw.kdata:
            ksp /= kdata_max

        kdata_max = [np.abs(ksp).max() for ksp in mri_raw.kdata]
        print(f'Max kdata {kdata_max}')

        if compress_coils:
            # Compress Coils
            if 18 < Num_Coils <= 32:
                mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=20)
                mri_raw.Num_Coils = 20

            if Num_Coils > 32:
                mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=20)
                mri_raw.Num_Coils = 20

        return mri_raw
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
    
def pca_coil_compression(kdata=None, axis=0, target_channels=None):

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
'''
def autofov(kdata,coords,dcf, device=None,
            thresh=0.1, scale=1, oversample=1.5):
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
        dcf = dcf

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

        for c in range(20):
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

    for e in range(len(coords)):
        coords[e] *= img_scale

    new_img_shape = sp.estimate_shape(coords)
    print(sp.estimate_shape(coords))
    #print(sp.estimate_shape(mri_raw.coords[1]))
    #print(sp.estimate_shape(mri_raw.coords[2]))
    #print(sp.estimate_shape(mri_raw.coords[3]))
    #print(sp.estimate_shape(mri_raw.coords[4]))

   # logger.info('Image shape: {}'.format(new_img_shape))
    return coords
'''
def convert_ute(h5_file, max_coils=20, dsfSpokes=1.0, compress_coils=False):
    with h5py.File(h5_file, "r") as hf:
        try:
            num_encodes = 1 #np.squeeze(hf["Kdata"].attrs["Num_Encodings"])
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
                ksp = pca_cc(kdata=ksp, axis=0, target_channels=max_coils)
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
    return kdata, coords, dcfs, resps / resps.max(), tr,ecgs/ecgs.max()

def gate_kspace(mri_raw=None, num_frames=10, gate_type='resp', discrete_gates=False, ecg_delay=300e-3):
    logger = logging.getLogger('Gate k-space')

    # Assume the input is a list

    # Get the MRI Raw structure setup
    mri_rawG = MRI_Raw()
    mri_rawG.Num_Coils = mri_raw.Num_Coils
    mri_rawG.Num_Encodings = mri_raw.Num_Encodings
    mri_rawG.dft_needed = mri_raw.dft_needed
    mri_rawG.trajectory_type = mri_raw.trajectory_type

    # List array
    mri_rawG.coords = []
    mri_rawG.dcf = []
    mri_rawG.kdata = []
    mri_rawG.time = []
    mri_rawG.ecg = []
    mri_rawG.prep = []
    mri_rawG.resp = []

    gate_signals = {
        'ecg': mri_raw.ecg,
        'time': mri_raw.time,
        'prep': mri_raw.prep,
        'resp': mri_raw.resp
    }
    gate_signal = gate_signals.get(gate_type, f'Cannot interpret gate signal {gate_type}')

    # For ECG, delay the waveform
    if gate_type == 'ecg':
       time = mri_raw.time

       for e in range(mri_raw.Num_Encodings):
           time_encode = time[e].flatten()
           ecg_encode = gate_signal[e].flatten()

            #Sort the data by time
           idx = np.argsort(time_encode)
           idx_inverse = idx.argsort()

           # Estimate the delay
           if e == 0:
               print(f'Time max {time_encode.max()}')
               print(f'Time size {time_encode.size}')
               print(f'Time ecg delay {ecg_delay}')
               
               ecg_shift = int(ecg_delay / time_encode.max() * time_encode.size)
               print(f'Shifting by {ecg_shift}')

           #Using circular shift for now. This should be fixed
           ecg_sorted = ecg_encode[idx]
           ecg_shifted = np.roll( ecg_sorted, -ecg_shift)
           gate_signal[e] = np.reshape(ecg_shifted[idx_inverse], time[e].shape)

    print(f'Gating off of {gate_type}')

    t_min, t_max, delta_time = get_gate_bins(gate_signal, gate_type, num_frames, discrete_gates)

    points_per_bin = []
    count = 0
    for t in range(num_frames):
        for e in range(mri_raw.Num_Encodings):
            t_start = t_min + delta_time * t
            t_stop = t_start + delta_time

            # # Find index where value is held
            # idx = np.argwhere(np.logical_and.reduce([
            #     np.abs(gate_signal[e]) >= t_start,
            #     np.abs(gate_signal[e]) < t_stop]))
            
            idx = np.logical_and.reduce([
                np.abs(gate_signal[e]) >= t_start,
                np.abs(gate_signal[e]) < t_stop])
            current_points = np.sum(idx)

            post_gate = gate_signal[e][idx]
            #print(f'Post gate min = {np.min(post_gate)}')
            #print(f'Post gate max = {np.max(post_gate)}')
            #print(f'Size of gate = {gate_signal[e].shape}')

           # ecg = mri_raw.ecg[e][idx]
            #print(f'Post ecg min = {np.min(ecg)}')
            #print(f'Post ecg max = {np.max(ecg)}')
            #print(f'Size of ecg = {mri_raw.ecg[e].shape}')


            # Gate the data
            points_per_bin.append(current_points)

            #print('(t_start,t_stop) = (', t_start, ',', t_stop, ')')
            logger.info(f'Frame {t} [{t_start} to {t_stop} ] | {e}, Points = {current_points}')

            #print(gate_signal[e].shape)
            #print(mri_raw.coords[e].shape)
            #print(mri_raw.coords[e].shape)
            #print(idx.shape)

            # mri_rawG.coords.append(mri_raw.coords[e][idx[:, 0], :])
            # mri_rawG.dcf.append(mri_raw.dcf[e][idx[:, 0]])
            # mri_rawG.kdata.append(mri_raw.kdata[e][:, idx[:, 0]])
            # mri_rawG.time.append(mri_raw.time[e][idx[:, 0]])
            # mri_rawG.resp.append(mri_raw.resp[e][idx[:, 0]])
            # mri_rawG.prep.append(mri_raw.prep[e][idx[:, 0]])
            # mri_rawG.ecg.append(mri_raw.ecg[e][idx[:, 0]])

            new_kdata = []
            for coil in range(mri_raw.kdata[e].shape[0]):
                old_kdata = mri_raw.kdata[e][coil]
                new_kdata.append(old_kdata[idx])
            mri_rawG.kdata.append(np.stack(new_kdata, axis=0))

            new_coords = []
            for dim in range(mri_raw.coords[e].shape[-1]):
                old_coords = mri_raw.coords[e][...,dim]
                new_coords.append(old_coords[idx])
            mri_rawG.coords.append(np.stack(new_coords, axis=-1))

            mri_rawG.dcf.append(mri_raw.dcf[e][idx])
           # mri_rawG.time.append(mri_raw.time[e][idx])
            mri_rawG.resp.append(mri_raw.resp[e][idx])
           # mri_rawG.prep.append(mri_raw.prep[e][idx])
           # mri_rawG.ecg.append(mri_raw.ecg[e][idx])

            #print(f'ECG Time before = {np.min(mri_raw.ecg[e])} {np.max(mri_raw.ecg[e])}')
            #print(f'ECG Time after = {np.min(mri_rawG.ecg[-1])} {np.max(mri_rawG.ecg[-1])}')

            #ecg = mri_raw.ecg[e][idx]
            #print(f'Post ecg min = {np.min(ecg)}')
            #print(f'Post ecg max = {np.max(ecg)}')
            #print(f'Size of ecg = {mri_raw.ecg[e].shape}')


            #mri_rawG.kdata.append(mri_raw.kdata[e][idx_kdata])
            #mri_rawG.coords.append(mri_raw.coords[e][idx_coord])
            
            count += 1

    max_points_per_bin = np.max(np.array(points_per_bin))
    logger.info(f'Max points = {max_points_per_bin}')
    logger.info(f'Points per bin = {points_per_bin}')
    logger.info(
        f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
    logger.info(f'Standard deviation = {np.std(points_per_bin)}')

    mri_rawG.Num_Frames = num_frames

    return (mri_rawG)
def get_gate_bins( gate_signal, gate_type, num_frames, discrete_gates=False, prep_disdaqs=0):
    logger = logging.getLogger('Get Gate bins')

    #print(gate_signal)
    #print(gate_signal[0].dtype)

    # Loop over all encodes
    t_min = np.min([np.min(gate) for gate in gate_signal])
    t_max = np.max([np.max(gate) for gate in gate_signal])

    if gate_type == 'ecg':
        logger.info('Using median ECG value for tmax')
        median_rr = np.mean([np.median(gate) for gate in gate_signal])
        median_rr = 2.0 * (median_rr - t_min) + t_min
        t_max = median_rr
        logger.info(f'Median RR = {median_rr}')

        # Check the range
        sum_within = np.sum([np.sum(gate < t_max) for gate in gate_signal])
        sum_total = np.sum([gate.size for gate in gate_signal])
        within_rr = 100.0 * sum_within / sum_total
        logger.info(f'ECG, {within_rr} percent within RR')
    elif gate_type == 'resp':
        # Outlier rejection
        q05 = np.mean([np.quantile(gate, 0.05) for gate in gate_signal])
        q95 = np.mean([np.quantile(gate, 0.95) for gate in gate_signal])

        # Linear fit
        t_max = q95 + (q95 - q05) / 0.9 * 0.05
        t_min = q05 + (q95 - q05) / 0.9 * -0.05
    elif gate_type == 'prep':
        # Skip a number of projections
        t_min = np.min([np.min(gate) for gate in gate_signal]) + prep_disdaqs


    if discrete_gates:
        t_min -= 0.5
        t_max += 0.5
    else:
        # Pad so bins are inclusive
        t_min -= 1e-6
        t_max += 1e-6

    logger.info(f'Max time = {t_max}')
    logger.info(f'Min time = {t_min}')

    delta_time = (t_max - t_min) / num_frames
    logger.info(f'Delta = {delta_time}')

    return t_min, t_max, delta_time



def gate_kspace(mri_raw=None, num_frames=10, gate_type='resp', discrete_gates=False, ecg_delay=300e-3):
    logger = logging.getLogger('Gate k-space')

    # Assume the input is a list

    # Get the MRI Raw structure setup
    mri_rawG = MRI_Raw()
    mri_rawG.Num_Coils = mri_raw.Num_Coils
    mri_rawG.Num_Encodings = mri_raw.Num_Encodings
    mri_rawG.dft_needed = mri_raw.dft_needed
    mri_rawG.trajectory_type = mri_raw.trajectory_type

    # List array
    mri_rawG.coords = []
    mri_rawG.dcf = []
    mri_rawG.kdata = []
    mri_rawG.time = []
    mri_rawG.ecg = []
    mri_rawG.prep = []
    mri_rawG.resp = []

    gate_signals = {
        'ecg': mri_raw.ecg,
        'time': mri_raw.time,
        'prep': mri_raw.prep,
        'resp': mri_raw.resp
    }
    gate_signal = gate_signals.get(gate_type, f'Cannot interpret gate signal {gate_type}')

    # For ECG, delay the waveform
    if gate_type == 'ecg':
       time = mri_raw.time

       for e in range(mri_raw.Num_Encodings):
           time_encode = time[e].flatten()
           ecg_encode = gate_signal[e].flatten()

            #Sort the data by time
           idx = np.argsort(time_encode)
           idx_inverse = idx.argsort()

           # Estimate the delay
           if e == 0:
               print(f'Time max {time_encode.max()}')
               print(f'Time size {time_encode.size}')
               print(f'Time ecg delay {ecg_delay}')
               
               ecg_shift = int(ecg_delay / time_encode.max() * time_encode.size)
               print(f'Shifting by {ecg_shift}')

           #Using circular shift for now. This should be fixed
           ecg_sorted = ecg_encode[idx]
           ecg_shifted = np.roll( ecg_sorted, -ecg_shift)
           gate_signal[e] = np.reshape(ecg_shifted[idx_inverse], time[e].shape)

    print(f'Gating off of {gate_type}')

    t_min, t_max, delta_time = get_gate_bins(gate_signal, gate_type, num_frames, discrete_gates)

    points_per_bin = []
    count = 0
    for t in range(num_frames):
        for e in range(mri_raw.Num_Encodings):
            t_start = t_min + delta_time * t
            t_stop = t_start + delta_time

            # # Find index where value is held
            # idx = np.argwhere(np.logical_and.reduce([
            #     np.abs(gate_signal[e]) >= t_start,
            #     np.abs(gate_signal[e]) < t_stop]))
            
            idx = np.logical_and.reduce([
                np.abs(gate_signal[e]) >= t_start,
                np.abs(gate_signal[e]) < t_stop])
            current_points = np.sum(idx)

            post_gate = gate_signal[e][idx]
            #print(f'Post gate min = {np.min(post_gate)}')
            #print(f'Post gate max = {np.max(post_gate)}')
            #print(f'Size of gate = {gate_signal[e].shape}')

           # ecg = mri_raw.ecg[e][idx]
            #print(f'Post ecg min = {np.min(ecg)}')
            #print(f'Post ecg max = {np.max(ecg)}')
            #print(f'Size of ecg = {mri_raw.ecg[e].shape}')


            # Gate the data
            points_per_bin.append(current_points)

            #print('(t_start,t_stop) = (', t_start, ',', t_stop, ')')
            logger.info(f'Frame {t} [{t_start} to {t_stop} ] | {e}, Points = {current_points}')

            #print(gate_signal[e].shape)
            #print(mri_raw.coords[e].shape)
            #print(mri_raw.coords[e].shape)
            #print(idx.shape)

            # mri_rawG.coords.append(mri_raw.coords[e][idx[:, 0], :])
            # mri_rawG.dcf.append(mri_raw.dcf[e][idx[:, 0]])
            # mri_rawG.kdata.append(mri_raw.kdata[e][:, idx[:, 0]])
            # mri_rawG.time.append(mri_raw.time[e][idx[:, 0]])
            # mri_rawG.resp.append(mri_raw.resp[e][idx[:, 0]])
            # mri_rawG.prep.append(mri_raw.prep[e][idx[:, 0]])
            # mri_rawG.ecg.append(mri_raw.ecg[e][idx[:, 0]])

            new_kdata = []
            for coil in range(mri_raw.kdata[e].shape[0]):
                old_kdata = mri_raw.kdata[e][coil]
                new_kdata.append(old_kdata[idx])
            mri_rawG.kdata.append(np.stack(new_kdata, axis=0))

            new_coords = []
            for dim in range(mri_raw.coords[e].shape[-1]):
                old_coords = mri_raw.coords[e][...,dim]
                new_coords.append(old_coords[idx])
            mri_rawG.coords.append(np.stack(new_coords, axis=-1))

            mri_rawG.dcf.append(mri_raw.dcf[e][idx])
           # mri_rawG.time.append(mri_raw.time[e][idx])
            mri_rawG.resp.append(mri_raw.resp[e][idx])
           # mri_rawG.prep.append(mri_raw.prep[e][idx])
           # mri_rawG.ecg.append(mri_raw.ecg[e][idx])

            #print(f'ECG Time before = {np.min(mri_raw.ecg[e])} {np.max(mri_raw.ecg[e])}')
            #print(f'ECG Time after = {np.min(mri_rawG.ecg[-1])} {np.max(mri_rawG.ecg[-1])}')

            #ecg = mri_raw.ecg[e][idx]
            #print(f'Post ecg min = {np.min(ecg)}')
            #print(f'Post ecg max = {np.max(ecg)}')
            #print(f'Size of ecg = {mri_raw.ecg[e].shape}')


            #mri_rawG.kdata.append(mri_raw.kdata[e][idx_kdata])
            #mri_rawG.coords.append(mri_raw.coords[e][idx_coord])
            
            count += 1

    max_points_per_bin = np.max(np.array(points_per_bin))
    logger.info(f'Max points = {max_points_per_bin}')
    logger.info(f'Points per bin = {points_per_bin}')
    logger.info(
        f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
    logger.info(f'Standard deviation = {np.std(points_per_bin)}')

    mri_rawG.Num_Frames = num_frames

    return (mri_rawG)
def get_gate_bins( gate_signal, gate_type, num_frames, discrete_gates=False, prep_disdaqs=0):
    logger = logging.getLogger('Get Gate bins')

    #print(gate_signal)
    #print(gate_signal[0].dtype)

    # Loop over all encodes
    t_min = np.min([np.min(gate) for gate in gate_signal])
    t_max = np.max([np.max(gate) for gate in gate_signal])

    if gate_type == 'ecg':
        logger.info('Using median ECG value for tmax')
        median_rr = np.mean([np.median(gate) for gate in gate_signal])
        median_rr = 2.0 * (median_rr - t_min) + t_min
        t_max = median_rr
        logger.info(f'Median RR = {median_rr}')

        # Check the range
        sum_within = np.sum([np.sum(gate < t_max) for gate in gate_signal])
        sum_total = np.sum([gate.size for gate in gate_signal])
        within_rr = 100.0 * sum_within / sum_total
        logger.info(f'ECG, {within_rr} percent within RR')
    elif gate_type == 'resp':
        # Outlier rejection
        q05 = np.mean([np.quantile(gate, 0.05) for gate in gate_signal])
        q95 = np.mean([np.quantile(gate, 0.95) for gate in gate_signal])

        # Linear fit
        t_max = q95 + (q95 - q05) / 0.9 * 0.05
        t_min = q05 + (q95 - q05) / 0.9 * -0.05
    elif gate_type == 'prep':
        # Skip a number of projections
        t_min = np.min([np.min(gate) for gate in gate_signal]) + prep_disdaqs


    if discrete_gates:
        t_min -= 0.5
        t_max += 0.5
    else:
        # Pad so bins are inclusive
        t_min -= 1e-6
        t_max += 1e-6

    logger.info(f'Max time = {t_max}')
    logger.info(f'Min time = {t_min}')

    delta_time = (t_max - t_min) / num_frames
    logger.info(f'Delta = {delta_time}')

    return t_min, t_max, delta_time


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

        self._normalize()

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

#RO=150



#ksp=kspa[:,:,:RO]
##coord=coorda[:,:RO]
#d#cf=dcfa[:,:RO]
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
    L_blocks,R_blocks,B = MultiScaleLowRankRecona(kspa[:,:,:], coorda[:,:], dcfa[:,:], mps, T, lamda, device=device, blk_widths=blk_widths).run()

    mpsa=mps

    im_test=np.zeros([1,mpsa.shape[1],mpsa.shape[2],mpsa.shape[3]],dtype=np.complex64)
    temp=0
    for i in range(1):
        for j in range(3):
            temp=temp+B[j](L_blocks[j]*R_blocks[j][i])
        im_test[i]=temp.get()

    im_testa=im_test
    return im_testa,mps,kspa,coorda,dcfa

#use for unrolls
def calculate_sense0(M_t,ksp,mps_c,coord_t,dcf,index_max,index_frame):
        torch.cuda.empty_cache()
        r = torch.cuda.memory_reserved(0) /1e9
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], torch.reshape(coord_t,[-1,3]), oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
       
        torch.cuda.empty_cache()
       
        torch.cuda.empty_cache()
        e_tc=F_torch.apply(M_t)
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1])
       # e_tca=(e_tca/torch.abs(e_tca).max())*torch.abs(ksp).max()
       # e_tc_update=e_tca
     #   print(torch.abs(e_tca).max())
        print(torch.abs(ksp).max())
        #loss_self1=torch.nn.MSELoss()
        torch.cuda.empty_cache()
        resk=(((ksp-e_tca)*dcf**0.5))
        torch.cuda.empty_cache()
        #loss=(torch.norm(resk)) #/torch.norm(ksp,2)+torch.norm(resk,1)/torch.norm(ksp,1) #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
     
        loss=torch.norm(resk,2)**2  #*index_all/ksp.shape[0]
        r = torch.cuda.memory_reserved(0) /1e9
      #  print(r)
        return loss

def _updateb(img_t,ksp_t,dcf_t,coord_t,mpsa): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  
# Data consistency.
  loss_t=0
  #print(mpsa.shape)
  #print(img_t.shape)
  
  for c in range(mpsa.shape[0]):
    torch.cuda.empty_cache()
    loss_t=loss_t+torch.utils.checkpoint.checkpoint(calculate_sense0,torch.cat([torch.reshape(torch.real(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1]),torch.reshape(torch.imag(img_t*mpsa[c].cuda()),[mpsa.shape[1],mpsa.shape[2],mpsa.shape[3],1])],axis=3),torch.reshape(ksp_t[c],[-1]),mpsa[c].cuda(),torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]),index_max,index_frame)
  
  print(loss_t)
  return loss_t

#use for unrolls
def initialize0(M_ta,ksp,mps_c,coord_t,dcf):
   # print(mps_c.shape)
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
   # print('a')
    #print(cupy.abs(F(M_ta*mps_c)).max())
    #print(cupy.abs(ksp).max())
    #e_tca=operator(F_torch,M_ta,mps[c],dcf)
    #e_tca=F.H(F(M_ta*mps_c)*dcf)*cupy.conj(mps_c)
    k=(ksp)*dcf
   # print(k.shape)
    k_real=torch.real(torch.squeeze(k))
    k_imag=torch.imag(torch.squeeze(k))
    k_real=k_real.unsqueeze(axis=2)
    k_imag=k_imag.unsqueeze(axis=2)
    k_com=torch.cat([k_real,k_imag],axis=2)
    
    g=FH_torch.apply(k_com)
    y_guess=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c).cuda()
   
    Ap0=torch.utils.checkpoint.checkpoint(operator0,M_ta,mps_c,dcf,coord_t)
   # print(y_guess.shape)
    resk=y_guess-Ap0
    p=resk
    rzold=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    rzold=rzold
    p=p
    resk=resk
    torch.cuda.empty_cache()
    return rzold,p,resk
    
    
def update_CG0(M_ta,ksp,mps,coord_t,dcf,rzold,p,resk):
    F = sp.linop.NUFFT([mps.shape[1], mps.shape[2], mps.shape[3]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
    
   # print('a')
    #print(cupy.abs(F(M_ta*mps_c)).max())
    #print(cupy.abs(ksp).max())
    Ap=0

    for c in range(mps.shape[0]):
        #p.requires_grad=True
        Ap=Ap+torch.utils.checkpoint.checkpoint(operator,p,mps[c],dcf,coord_t)
   # print('c')
    
    pAp=torch.real(torch.vdot(p.flatten(),Ap.flatten()))
   
    alpha=rzold/pAp
   # alpha=np.float32(alpha)*torch.ones([1])
    alpha=alpha.cuda()
   
    M_ta=M_ta+alpha*p
    resk=resk-alpha*Ap
    
    rznew=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    beta=rznew/rzold
    
   # beta=(beta)*torch.ones([1])
    beta=beta.cuda()
    
    p=resk+beta*p
   
    torch.cuda.empty_cache()
    return rznew,p,M_ta,resk
  
def operator0(x,mps_c,dcf,coord_t):
   # print(mps_c.shape)
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
    mps_c=mps_c.cuda()
    diff=torch.cat([torch.reshape(torch.real(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1]),torch.reshape(torch.imag(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1])],axis=3)
    
    e_tc=F_torch.apply(diff)
   # print(torch.abs(e_tc).max())
    e_tca=torch.complex(e_tc[:,:,0],e_tc[:,:,1])*dcf
    e_tca_real=torch.real(torch.squeeze(e_tca))
    e_tca_imag=torch.imag(torch.squeeze(e_tca))
    e_tca_real=e_tca_real.unsqueeze(axis=2)
    e_tca_imag=e_tca_imag.unsqueeze(axis=2)
    e_tca_com=torch.cat([e_tca_real,e_tca_imag],axis=2)
    g=FH_torch.apply(e_tca_com)
   # print(g.shape)
    gout=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c).cuda() #+.0001*x
    #resk=e_tca*dcf
   
    return gout 
def initialize(M_ta,ksp,mps_c,coord_t,dcf,alpha):
   # print(mps_c.shape)
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=2)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
   # print('a')
    #print(cupy.abs(F(M_ta*mps_c)).max())
    #print(cupy.abs(ksp).max())
    #e_tca=operator(F_torch,M_ta,mps[c],dcf)
    #e_tca=F.H(F(M_ta*mps_c)*dcf)*cupy.conj(mps_c)
    k=(ksp)*dcf
   
    k_real=torch.real(torch.squeeze(k))
    k_imag=torch.imag(torch.squeeze(k))
    k_real=k_real.unsqueeze(axis=2)
    k_imag=k_imag.unsqueeze(axis=2)
    k_com=torch.cat([k_real,k_imag],axis=2)
    
    g=FH_torch.apply(k_com)
    y_guess=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c).cuda()
   
    Ap0=torch.utils.checkpoint.checkpoint(operator0,M_ta,mps_c,dcf,coord_t)
   # print(y_guess.shape)
    resk=y_guess-Ap0
    p=resk
    rzold=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    rzold=rzold
    p=p
    resk=resk
    torch.cuda.empty_cache()
    return rzold,p,resk
    
    
def update_CG(M_ta,ksp,mps,coord_t,dcf,rzold,p,resk,alpha1):
    F = sp.linop.NUFFT([mps.shape[1], mps.shape[2], mps.shape[3]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
      
    
   # print('a')
    #print(cupy.abs(F(M_ta*mps_c)).max())
    #print(cupy.abs(ksp).max())
    Ap=0

    for c in range(mps.shape[0]):
        #p.requires_grad=True
        Ap=Ap+torch.utils.checkpoint.checkpoint(operator,p,mps[c],dcf,coord_t,alpha1)
   # print('c')
    Ap=Ap.cuda()
    pAp=torch.real(torch.vdot(p.flatten(),Ap.flatten()))
   
    alpha=rzold/pAp
   # alpha=np.float32(alpha)*torch.ones([1])
    alpha=alpha.cuda()
   
    M_ta=M_ta+alpha*p
    resk=resk-alpha*Ap
    
    rznew=torch.real(torch.vdot(resk.flatten(),resk.flatten()))
    beta=rznew/rzold
    
   # beta=(beta)*torch.ones([1])
    beta=beta.cuda()
    
    p=resk+beta*p
   
    torch.cuda.empty_cache()
    return rznew,p,M_ta,resk
  
def operator(x,mps_c,dcf,coord_t,alpha):
    F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(), oversamp=1.25, width=4)
    F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
    FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
    mps_c=mps_c.cuda()
    diff=torch.cat([torch.reshape(torch.real(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1]),torch.reshape(torch.imag(x*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1])],axis=3)
    
    e_tc=F_torch.apply(diff)
   # print(torch.abs(e_tc).max())
    e_tca=torch.complex(e_tc[:,:,0],e_tc[:,:,1])*dcf
    e_tca_real=torch.real(torch.squeeze(e_tca))
    e_tca_imag=torch.imag(torch.squeeze(e_tca))
    e_tca_real=e_tca_real.unsqueeze(axis=2)
    e_tca_imag=e_tca_imag.unsqueeze(axis=2)
    e_tca_com=torch.cat([e_tca_real,e_tca_imag],axis=2)
    g=FH_torch.apply(e_tca_com)
   # print(g.shape)
    gout=torch.complex(g[:,:,:,0],g[:,:,:,1]).cuda()*torch.conj(mps_c).cuda()+alpha.cuda()*x.cuda()

    #resk=e_tca*dcf
   
    return gout
def run_unroll1(out0,ksp,coord,dcf,mps,alpha,iters):
    for n in range(1):
           
           # M_ta=torch.zeros([mps[lo].shape[1],mps[lo].shape[2],mps[lo].shape[3]],dtype=torch.cfloat).cuda()

        
        
           # M_ta1=0
            resk1=0
            p1=0
            rzold1=0
           
           
          #  loo=loo.detach().cpu().numpy()
            print(ksp.shape)
            for c in range(mps.shape[0]):
             
               # if c>0:
               #     ksp=torch.from_numpy(ksp)
               #     coord=torch.from_numpy(coord)
               #     dcf=torch.from_numpy(dcf)
               # ksp.requires_grad=False
               # coord.requires_grad=False
               # mps.requires_grad=False
               # dcf.requires_grad=False
                ksp=ksp.detach()
                coord=coord.detach()
                mps=mps.detach()
                dcf=dcf.detach()
                torch.cuda.empty_cache()
           
               
              #  d=d.type(torch.complex64)
                n=0
                if n>=0:
                    k=ksp[c,:,:]
                    co=coord[:,:]
                    d=dcf[:,:]
                
              
               # out0.requires_grad=True
                k=k.cuda()
                co=co.cuda()
                d=d.cuda()
                
                rzold,p,resk=torch.utils.checkpoint.checkpoint(initialize0,out0,k,mps[c],co,d)
                rzold1=rzold1+rzold
                p1=p1+p
                resk1=resk1+resk
              #  k=k.detach().cpu().numpy()
              #  mo=mo.detach().cpu().numpy()
               # d=d.detach().cpu().numpy()
                #ksp=ksp.detach().cpu().numpy()
                #dcf=dcf.detach().cpu().numpy()
               # mps=mps.detach().cpu().numpy()
                #coord=coord.detach().cpu().numpy()
                #co=co.detach().cpu().numpy()
                torch.cuda.empty_cache()
               


            for j in range(iters):
               # print(j)
                r = torch.cuda.memory_reserved(0) /1e9
              
             
                torch.cuda.empty_cache()
               
               # ksp=torch.from_numpy(ksp)
              
           
               # dcf=torch.from_numpy(dcf)
               # coord=torch.from_numpy(coord)
               # k=k.type(torch.complex64)
               # k.requires_grad=False
               # co.requires_grad=False
               # mps.requires_grad=False
               # d.requires_grad=False
                torch.cuda.empty_cache()

             
                #M_ta=torch.from_numpy(M_ta).cuda()
                if n>=0:
                    k=ksp[:,:,:]
                    co=coord[:,:]
                    d=dcf[:,:]
             
             
              
                k=k.cuda()
                co=co.cuda()
                d=d.cuda()
               # mps=mps.cuda()
               # p1=torch.from_numpy(p1).cuda()
               # p1=p1.type(torch.complex64)
                #rzold1=torch.from_numpy(rzold1).cuda()
               # p1=torch.from_numpy(p1).cuda()
                resk1=(resk1).cuda()
               # resk1=resk1.type(torch.complex64)
               # out0.requires_grad=True
               # alpha=torch.ones([1])
                rzold1,p1,out0,resk1=torch.utils.checkpoint.checkpoint(update_CG,out0,k,mps,co,d,rzold1,p1,resk1,alpha)
                loss=0
              

                #loss+=torch.utils.checkpoint.checkpoint(_updateb,M_ta.cuda(),k.cuda(),d.cuda(),c.cuda(),m.cuda(),beta.cuda(),gamma.cuda())
               # k=k.detach().cpu().numpy()
               ## mps=mps.detach().cpu().numpy()
               # d=d.detach().cpu().numpy()
               ## co=co.detach().cpu().numpy()
                ##ksp=ksp.detach().cpu().numpy()
                #dcf=dcf.detach().cpu().numpy()
                #coord=coord.detach().cpu().numpy()
                torch.cuda.empty_cache()

              
               # M_ta=M_ta.detach().cpu().numpy()
    return out0
def calculate_sense0(img_t,ksp,mps_c,coord_t,dcf):
        torch.cuda.empty_cache()
       # r = torch.cuda.memory_reserved(0) /1e9
        ksp=torch.reshape(ksp,[-1])
        coord_t=torch.reshape(coord_t,[-1,3])
        dcf=torch.reshape(dcf,[-1])
        mps_c=mps_c.cuda()
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], torch.reshape(coord_t,[-1,3]), oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        M_t=torch.cat([torch.reshape(torch.real(img_t*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1]),torch.reshape(torch.imag(img_t*mps_c),[mps_c.shape[0],mps_c.shape[1],mps_c.shape[2],1])],axis=3)
       
        
       
        torch.cuda.empty_cache()
       
        torch.cuda.empty_cache()
        e_tc=F_torch.apply(M_t)
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1])
       # e_tca=(e_tca/torch.abs(e_tca).max())*torch.abs(ksp).max()
       # e_tc_update=e_tca
      #  print(ksp.shape)
      #  print(e_tca.shape)
       # print(torch.abs(e_tca).max())
       # print(torch.abs(ksp).max())
        #loss_self1=torch.nn.MSELoss()
        torch.cuda.empty_cache()
        resk=((ksp-e_tca)*(dcf)**0.5)
        torch.cuda.empty_cache()
        #loss=(torch.norm(resk)) #/torch.norm(ksp,2)+torch.norm(resk,1)/torch.norm(ksp,1) #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
       
        loss=torch.norm(resk,2)**2 #*index_all/ksp.shape[0]
       # r = torch.cuda.memory_reserved(0) /1e9
       # print(r)
      #  print(r)
        ksp=ksp.detach().cpu().numpy()
        coord_t=coord_t.detach().cpu().numpy()
        dcf=dcf.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return (loss)

def _updateb(img_t,ksp_t,dcf_t,coord_t,mpsa): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  
# Data consistency.
  loss_t=0
  #print(mpsa.shape)
  #print(img_t.shape)
  
  for c in range(mpsa.shape[0]):
   # torch.cuda.empty_cache()
    loss_t=loss_t+torch.utils.checkpoint.checkpoint(calculate_sense0,img_t,ksp_t[c],mpsa[c],coord_t,dcf_t)
  
  print(loss_t)
  return loss_t

def calculate_sense0a(M_t,ksp,mps_c,coord_t,dcf):
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t, oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        
        e_tc=F_torch.apply(M_t)
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1])
       # e_tc_update=e_tca
       # print(torch.abs(e_tca).max())
       # print(torch.abs(ksp).max())
       # loss_self1=torch.nn.MSELoss()
       
        resk=(e_tca-ksp)*dcf
        resk_real=torch.real(torch.squeeze(resk))
        resk_imag=torch.imag(torch.squeeze(resk))
       # print(resk_imag.shape)
        resk_real=resk_real.unsqueeze(axis=1)
        resk_imag=resk_imag.unsqueeze(axis=1)
        resk_com=torch.cat([resk_real,resk_imag],axis=1)
        g=FH_torch.apply(resk_com)
       # print(g.shape)
        g=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c)
       # ksp=ksp.detach()
        
       
        #loss=torch.norm(resk)**2 #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
        return g.cuda()

def grad_step(img_t,ksp_t,dcf_t,coord_t,mps,alpha): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  #img_t = 0
 # for j in range(self.J):
  #      img_t += self.B[j](self.L[j] * self.R[j][t])
  #img_t= torch.as_tensor(im , device='cuda') 
  
 


  



# Data consistency.
  loss_t=0
  #diff=torch.zeros([mps.shape[1],mps.shape[2],mps.shape[3],2]).
#  div0=torch.zeros([mpsa.shape[0]])
  loss_tot=0
  g=0
  for c in range(mps.shape[0]):
 
   ## diff0=torch.reshape(torch.real(img_t*mps[c]),[mps.shape[1],mps.shape[2],mps.shape[3],1])
   # diff1=torch.reshape(torch.imag(img_t*mps[c]),[mps.shape[1],mps.shape[2],mps.shape[3],1])
   # diff=torch.cat([torch.reshape(torch.real(img_t*mps[c]),[mps.shape[1],mps.shape[2],mps.shape[3],1]),torch.reshape(torch.imag(img_t*mps[c]),[mps.shape[1],mps.shape[2],mps.shape[3],1])],axis=3)
   

   # torch.unsqueeze(diff0, dim)

    g=g+torch.utils.checkpoint.checkpoint(calculate_sense0a,torch.cat([torch.reshape(torch.real(img_t*mps[c].cuda()),[mps.shape[1],mps.shape[2],mps.shape[3],1]),torch.reshape(torch.imag(img_t*mps[c].cuda()),[mps.shape[1],mps.shape[2],mps.shape[3],1])],axis=3),torch.reshape(ksp_t[c],[-1]),mps[c].cuda(),torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
  img_t=img_t-alpha*g
 
 # diff=diff.detach()
  #dcf_t=dcf_t.detach()
   




  return img_t

def calculate_sense0b(ksp,mps_c,coord_t,dcf):
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t, oversamp=1.25, width=4)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        
       
       
        resk=(ksp)*dcf
        resk_real=torch.real(torch.squeeze(resk))
        resk_imag=torch.imag(torch.squeeze(resk))
       # print(resk_imag.shape)
        resk_real=resk_real.unsqueeze(axis=1)
        resk_imag=resk_imag.unsqueeze(axis=1)
        resk_com=torch.cat([resk_real,resk_imag],axis=1)
        g=FH_torch.apply(resk_com)
       # print(g.shape)
        g=torch.complex(g[:,:,:,0],g[:,:,:,1])*torch.conj(mps_c)
        ksp=ksp.detach()
        
       
        #loss=torch.norm(resk)**2 #torch.abs(torch.sum((e_tca.cuda()*dcf.cuda()**0.5-ksp_t.cuda()*dcf.cuda()**0.5)**2))
        return g

def init_im(ksp_t,dcf_t,coord_t,mps): #ima,deform_adjoint1,ksp,coord,dcf,mps,t,device,tr_per_frame):
  #img_t = 0
 # for j in range(self.J):
  #      img_t += self.B[j](self.L[j] * self.R[j][t])
  #img_t= torch.as_tensor(im , device='cuda') 
  
 


  



# Data consistency.
  loss_t=0
  #diff=torch.zeros([mps.shape[1],mps.shape[2],mps.shape[3],2]).
#  div0=torch.zeros([mpsa.shape[0]])
  loss_tot=0
  g=0
  for c in range(mps.shape[0]):
 
   
   

   # torch.unsqueeze(diff0, dim)

    g=g+torch.utils.checkpoint.checkpoint(calculate_sense0b,torch.reshape(ksp_t[c],[-1]),mps[c].cuda(),torch.reshape(coord_t,[-1,3]),torch.reshape(dcf_t,[-1]))
 
 
  
 # diff=diff.detach()
  #dcf_t=dcf_t.detach()
   




  return g

