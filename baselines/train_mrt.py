import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torch_dct as dct  # https://github.com/zh217/torch-dct
from torch.utils.data import DataLoader
from MRT.MRT_utils import disc_l2_loss, adv_disc_l2_loss
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
from MRT.Models import Transformer, Discriminator 
from MRT.opt import Options
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


def run_model(is_train, model, discriminator,  dataloader, real_motion_all, optimizer, optimizer_d, opt):
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
        
    for batch_i, batch_data in tqdm(enumerate(dataloader)): 
        batch_num+=1  
        input_seq, output_seq = batch_data
        long_output_seq = output_seq[:,:,opt.output_time:,:,:]
        output_seq = output_seq[:,:,:opt.output_time,:, :]       
        bs, n, t, jn, d = input_seq.shape
        input_ = input_seq.reshape(bs*n, opt.input_time, -1)
        output_ = torch.cat((input_[:,-1:,:], output_seq.reshape(bs*n, opt.output_time, -1)), 1)

                
        # output_ = output_seq.reshape(bs*n, opt.output_time, -1)
        motion_offset = input_[:, 1:opt.input_time, :] - input_[:, :opt.input_time-1, :]
        input_src = dct.dct(motion_offset)

        rec_ = model.forward(input_src, dct.idct(input_[:, -1:, :]), input_seq.reshape(bs,n,t,-1), None)
        rec = dct.idct(rec_)
        results = output_[:, :1, :]
        for i in range(1, opt.output_time + 1):
            results = torch.cat(
                [results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)],
                dim=1)
        results = results[:, 1:, :]  # 96 25 60
    
        pred_poses = results.reshape(bs, n, opt.output_time, jn, -1)
        gt_poses = output_seq

        loss = torch.mean((rec[:, :opt.output_time, :] - (output_[:, 1:opt.output_time+1, :] - output_[:, :opt.output_time, :])) ** 2)   # mpjpe loss

        if (batch_i + 1) % 2 == 0 and is_train==0:
            fake_motion = results
            disc_loss = disc_l2_loss(discriminator(fake_motion))
            loss = loss + 0.0005 * disc_loss

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
       

        if (batch_i + 1) % 2 == 0 and is_train==0:
            fake_motion = results
            fake_motion = fake_motion.detach()
            real_motion = real_motion_all[int(batch_i / 2)][1][1]
            real_motion = real_motion.view(-1, opt.output_time, opt.joints_num * 3)[:, :opt.output_time, :].float().to(opt.device)  #  60 is joints *3

            fake_disc_value = discriminator(fake_motion)
            real_disc_value = discriminator(real_motion)

            d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = adv_disc_l2_loss(real_disc_value,
                                                                                          fake_disc_value)

            optimizer_d.zero_grad()
            d_motion_disc_loss.backward()
            optimizer_d.step()
                   
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
    # scenes = ['Park', 'Street', 'Indoor', 'Special_Locations', 'Complex Crowd']
    scenes = ['Park']
    setup_seed(opt.seed)
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))


    model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
                    n_layers=3, n_head=8, d_k=64, d_v=64,device=device, opt=opt).to(device)

    discriminator = Discriminator(d_word_vec=opt.joints_num*3, d_model=opt.joints_num*3, d_inner=256,
                                n_layers=3, n_head=8, d_k=32, d_v=32,device=device).to(device)

    dataset = Data(dataset='MI-Motion', train_mode=True, scene=None, device=device,  input_time=opt.input_time, output_time=opt.output_time)
    
    test_dataset_list=[]
    for scene in scenes:
        test_dataset_list.append(Data(dataset='MI-Motion',  train_mode=False, scene=scene, device=device,  input_time=opt.input_time, output_time=opt.output_time))
   
    print(stamp)
    dataloader = DataLoader(dataset,
                            batch_size=opt.train_batch,
                            shuffle=True, drop_last=True)

    
    real_ = Data(dataset='Discrimation', train_mode=True, scene=None, device=device,  input_time=opt.input_time, output_time=opt.output_time)
    real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=opt.train_batch, shuffle=True, drop_last=True)
    real_motion_all = list(enumerate(real_motion_dataloader))

    print(">>> training params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    Evaluate = True
    save_model = True
    

    params = [
        {"params": model.parameters(), "lr": 0.0003}
    ]
    optimizer = optim.Adam(params)
    params_d = [
        {"params": discriminator.parameters(), "lr": 0.0005}
    ]
    optimizer_d = optim.Adam(params_d)

    loss_min = 100
    for epoch_i in range(1, opt.epochs+1):
        """
        ==================================
           Training Processing
        ==================================
        """
        ret = run_model(is_train=0, model=model, discriminator=discriminator, dataloader=dataloader, real_motion_all=real_motion_all, optimizer=optimizer, optimizer_d=optimizer_d, opt=opt)


      
        train_loss = ret['loss_3d']
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_i
        }
        print('epoch:', epoch_i, 'loss:', train_loss, "lr: {:.10f} ".format(optimizer.param_groups[0]['lr']))
        
        if save_model: 
            save_path = os.path.join('baselines/MRT/checkpoints', f'epoch_{epoch_i}.model')
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
                
                    ret = run_model(is_train=1, model=model,discriminator=None, dataloader=test_dataloader, real_motion_all=None, optimizer=None, optimizer_d=None, opt=opt)
                    print_errors(scene, ret['gjpe_err'], ret['ajpe_err'], ret['root_err'])
                
                test_loss_cur = ret['loss_3d']

                if test_loss_cur < loss_min:
                    save_path = os.path.join('baselines/MRT/checkpoints', f'best_epoch.model')
                    torch.save(checkpoint, save_path)
                    loss_min = test_loss_cur
                    print(f"Best epoch_{checkpoint['epoch']} model is saved!")


if __name__ == '__main__':
    option = Options().parse()
    processor(option)






