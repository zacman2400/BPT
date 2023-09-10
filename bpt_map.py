import numpy as np
import sigpy as sp
import cupy
import torch
from gated_data_loader import autofova, init_im,flows,warp1
from moco_scripts14 import gen_template
def compute_bpt_over_frames(frames, channels,bpt,ksp):
   
    spokes_per_bin=ksp.shape[1]//frames
    bpta=np.zeros([frames,channels],np.complex64)
    bpt_norm=np.zeros_like(bpta)
    for j in range(frames):
        tr_start=spokes_per_bin*j
        tr_end=spokes_per_bin*(j+1)
        bpta[j]=np.mean(bpt[tr_start:tr_end],axis=0)
    for c in range(channels):
        interm=(bpta[:,c]-np.mean(bpta[:,c]))
        bpt_norm[:,c]=interm/np.abs(interm).max()
    return bpt_norm

def norm_data(ksp,coord,dcf,RO,spokes_per_bin):
   # coord=autofova(coord, dcf,ksp,device=None,
    #        thresh=0.1, scale=1, oversample=2.0) #adjust size if needed
    # normalize
    mps,kspa,coorda,dcfa=gen_template(ksp,coord,dcf,RO,spokes_per_bin) #note RO is your readout length
    #gen template with conjugate gradient descent steps, goal is to ensure template is 
    #data-consistent with acquitred k-space data
    alpha=torch.zeros([1]).cuda()
    iters=10
    
    return mps,kspa,coorda,dcfa

#loss function for network
def calculate_sense0(M_t,ksp,mps_c,coord_t,dcf):
        F = sp.linop.NUFFT([mps_c.shape[0], mps_c.shape[1], mps_c.shape[2]], coord_t.detach().cpu().numpy(),1.25,2)
        F_torch = sp.to_pytorch_function(F, input_iscomplex=True, output_iscomplex=True)
        FH_torch = sp.to_pytorch_function(F.H, input_iscomplex=True, output_iscomplex=True)
        e_tc=F_torch.apply(M_t.cuda())
        e_tca=torch.complex(e_tc[:,0],e_tc[:,1]) 
        print(torch.abs(ksp).max())
        print(torch.abs(e_tca).max())
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


def adj_field_solver(deformL_param_adj,deformR_param_adj,deformL_param_for,deformR_param_for,im_template,ksp,coord,dcf,mps,iter_adj,RO,block_torch,ishape,T1,T2,interp,res,spokes_per_bin,weight_dc,weight_smoother): #,conL,conR,block_torchcon,ishapecon):
   # from utils_reg1 import flows,_updateb,f
    from torch.utils import checkpoint
    import torch
    import numpy as np
    import random
    scaler = torch.cuda.amp.GradScaler()
    #readout images and deformation fields during training
    deform_look=np.zeros([T2-T1,mps.shape[2],mps.shape[3]])
    image_still=np.zeros([T2-T1,mps.shape[2],mps.shape[3]])
    image_look=np.zeros([T2-T1,mps.shape[2],mps.shape[3]])
    im_tester=np.zeros([T2-T1,mps.shape[2],mps.shape[3]])
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
                    deform_look[j-T1]=(flowa[:,0,20,:,:].detach().cpu().numpy())
                       # image_rev[j]=np.abs(im_rev.detach().cpu().numpy())
                   # image_still[j-50]=np.abs(im_inv[0,0,:,:,35].detach().cpu().numpy())
                    image_look[j-T1]=np.abs(im_out[20,:,:].detach().cpu().numpy())
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