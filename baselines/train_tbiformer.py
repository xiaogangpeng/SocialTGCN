import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from tqdm import tqdm
import numpy as np
import itertools
import torch
import torch.optim as optim
import torch_dct as dct  # https://github.com/zh217/torch-dct
import time
from torch.utils.data import DataLoader
from TBIFormer.models import TBIFormer
from TBIFormer.utils.soft_dtw_cuda import SoftDTW
from TBIFormer.utils.iRPE import piecewise_index
from TBIFormer.utils.opt import Options
from TBIFormer.utils.TRPE import bulding_TRPE_matrix
from util.dataloader import Data
from util.metrics import RFDE, GJPE, AJPE, print_errors
import sys
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_model(is_train, model, dataloader, optimizer, opt):
    loss_list = []
    batch_num=0
    frame_idx = [2, 4, 8, 10, 14, 18, 25]
    gjpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
    ajpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
    root_err_total = np.arange(len(frame_idx), dtype=np.float_)

    if is_train == 0:
        model.train()
    else:
        model.eval()
        

    for batch_id, batch_data in tqdm(enumerate(dataloader)): 
        batch_num+=1  
        input_seq, output_seq = batch_data
        long_output_seq = output_seq[:,:,opt.output_time:,:,:]
        output_seq = output_seq[:,:,:opt.output_time,:, :]
        bs, n, t, jn, d = input_seq.shape
        input_ = input_seq.reshape(bs*n, opt.input_time, -1)
        output_ = torch.cat((input_[:,-1:,:], output_seq.reshape(bs*n, opt.output_time, -1)), 1)
        
        trj_dist = bulding_TRPE_matrix(input_seq, opt)

        
        # output_ = output_seq.reshape(bs*n, opt.output_time, -1)
        motion_offset = input_[:, 1:opt.input_time, :] - input_[:, :opt.input_time-1, :]
        input_src = dct.dct(motion_offset)

        rec_ = model.forward(input_src, n, trj_dist)
        rec = dct.idct(rec_)
        results = output_[:, :1, :]
        for i in range(1, opt.output_time + 1):
            results = torch.cat(
                [results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)],
                dim=1)
        results = results[:, 1:, :]  # 3 15 45
     
        pred_poses = results.reshape(bs, n, opt.output_time, jn, -1)
        gt_poses = output_seq

        loss = torch.mean((rec[:, :opt.output_time, :] - (output_[:, 1:opt.output_time+1, :] - output_[:, :opt.output_time, :])) ** 2)   # mpjpe loss
        loss_all = loss
        loss_list.append(loss.item())
        if is_train==0:
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
        elif is_train==1:
            gjpe_err = GJPE(gt_poses, pred_poses, frame_idx)
            ajpe_err = AJPE(gt_poses, pred_poses, frame_idx)
            root_err = RFDE(gt_poses, pred_poses, frame_idx)
            gjpe_err_total += gjpe_err
            ajpe_err_total += ajpe_err
            root_err_total += root_err
       
                   
    ret = {}
    if is_train == 0:
        ret["loss_3d"] = np.mean(loss_list)
    elif is_train==1:
        ret["loss_3d"] = np.mean(loss_list)
        ret["gjpe_err"] = gjpe_err_total / batch_num
        ret["ajpe_err"] = ajpe_err_total / batch_num
        ret["root_err"] = root_err_total / batch_num
    return ret




def processor(opt):
    device = opt.device
    scenes = ['Park']
    setup_seed(opt.seed)
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))


    model = TBIFormer(input_dim=opt.d_model, d_model=opt.d_model,
                        d_inner=opt.d_inner, n_layers=opt.num_stage,
                        n_head=opt.n_head , d_k=opt.d_k, d_v=opt.d_v, dropout=opt.dropout, device=device, kernel_size=opt.kernel_size, opt=opt).to(device)

    dataset = Data(dataset='MI-Motion', train_mode=True, scene=None, device=device,  input_time=opt.input_time, output_time=opt.output_time)
    
    test_dataset_list=[]
    for scene in scenes:
        test_dataset_list.append(Data(dataset='MI-Motion',  train_mode=False, scene=scene, device=device,  input_time=opt.input_time, output_time=opt.output_time))
   
    print(stamp)
    dataloader = DataLoader(dataset,
                            batch_size=opt.train_batch,
                            shuffle=True, drop_last=True)

    print(">>> training params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    Evaluate = True
    save_model = True
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           lr=opt.lr)

    loss_min = 100
    for epoch_i in range(1, opt.epochs+1):
        """
        ==================================
           Training Processing
        ==================================
        """
        ret = run_model(is_train=0, model=model, dataloader=dataloader, optimizer=optimizer, opt=opt)


      
        train_loss = ret['loss_3d']
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_i
        }
        print('epoch:', epoch_i, 'loss:', train_loss, "lr: {:.10f} ".format(optimizer.param_groups[0]['lr']))
        
        if save_model: 
            save_path = os.path.join('baselines/TBIFormer/checkpoints', f'epoch_{epoch_i}.model')
            torch.save(checkpoint, save_path)

        if Evaluate:
            with torch.no_grad():
                """
                  ==================================
                     Validating Processing
                  ==================================
                  """
                print("\033[0:35mEvaluating.....\033[m")
                for i, scene in enumerate(scenes):
                    test_dataloader = DataLoader(test_dataset_list[i],
                                      batch_size=1,
                                      shuffle=False, drop_last=True)
                
                    ret = run_model(is_train=1, model=model, dataloader=test_dataloader, optimizer=None, opt=opt)
                    print_errors(scene, ret['gjpe_err'], ret['ajpe_err'], ret['root_err'])
                
                test_loss_cur = ret['loss_3d']

                if test_loss_cur < loss_min:
                    save_path = os.path.join('baselines/TBIFormer/checkpoints', f'best_epoch.model')
                    torch.save(checkpoint, save_path)
                    loss_min = test_loss_cur
                    print(f"Best epoch_{checkpoint['epoch']} model is saved!")


if __name__ == '__main__':
    option = Options().parse()
    processor(option)






