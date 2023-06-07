import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
from HRI.AttnModel import AttModel
from HRI.opt import Options
from util.dataloader import Data
from util.metrics import *
from util.plot_pose import save_imgs
from fvcore.nn import FlopCountAnalysis
import sys
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def run_model(is_train, model,  dataloader, opt, scene, sample_idx):
    all_gt_poses = torch.zeros(0)
    all_pred_poses = torch.zeros(0)
    all_test_poses = torch.zeros(0)
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
       
        input_src = input_seq.reshape(bs*n, t, -1).clone()  # 192 15 45
        out_all = model(input_src, input_n=opt.input_time, output_n=opt.output_n, itera=10)  # 32 20 1 64
        
        # test_tensor = (input_src, opt.input_time, opt.output_n, 10)
        # flops = FlopCountAnalysis(model, test_tensor)
        # print(">>> FLOPs: {:.3f}G".format(flops.total() / 1000000000.0))

        out = out_all[:, opt.kernel_size:, :].reshape(bs * n, -1, jn*d)  # 192 45 45
        gt_poses = output_seq.view(bs, n, -1, jn, d)
        pred_poses = out.view(bs, n, -1, jn, d)


        if is_train == 1:
            gjpe_err = GJPE(gt_poses, pred_poses, frame_idx)
            ajpe_err = AJPE(gt_poses, pred_poses, frame_idx)
            root_err = RFDE(gt_poses, pred_poses, frame_idx)
            gjpe_err_total += gjpe_err
            ajpe_err_total += ajpe_err
            root_err_total += root_err

                
        # For ultra-long-term prediction
        if opt.ultra_long_term_prediction: 
            long_gt_poses = long_output_seq.reshape(n, opt.output_time, jn, -1)
            long_pred_poses = out[:,opt.output_time:,:].reshape(n, opt.output_time, jn, -1)         
            
            all_pred_poses = torch.cat((all_pred_poses, long_pred_poses.detach().cpu()), dim=0)
            all_gt_poses = torch.cat((all_gt_poses, long_gt_poses.detach().cpu()), dim=0)
            temp = torch.cat((input_seq.reshape(n, opt.output_time, jn, -1).detach().cpu(), all_gt_poses), dim=0)
            all_test_poses = torch.cat((all_test_poses, temp), dim=0)


        if opt.rendering_imgs:
            if batch_i == sample_idx:
                print(f"Saving images for batch: {batch_i}")
                if opt.ultra_long_term_prediction:
                    pred_poses_ = pred_poses
                    gt_poses_ = torch.cat((gt_poses, long_gt_poses.unsqueeze(0)), dim=2)
                    save_imgs(batch_i, 'HRI', scene, pred_poses_, gt_poses_, input_seq, opt)
                else:
                    save_imgs(batch_i, 'HRI', scene, pred_poses[:,:,:opt.output_time], gt_poses, input_seq, opt)
    ret = {}

    if is_train == 1:
        ret["gjpe_err"] = gjpe_err_total / batch_num
        ret["ajpe_err"] = ajpe_err_total / batch_num
        ret["root_err"] = root_err_total / batch_num
        if opt.ultra_long_term_prediction:
            long_term_metric(all_test_poses.permute(0,2,1,3).numpy(), all_pred_poses.permute(0,2,1,3).numpy(), all_gt_poses.permute(0,2,1,3).numpy())

    return ret



if __name__ == '__main__':
    opt = Options().parse()
    device = opt.device
    scenes = ['Park', 'Street', 'Indoor', 'Special_Locations', 'Complex_Crowd']
    # scenes = ['Special_Locations', 'Complex_Crowd']
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    
    model = AttModel(in_features=opt.in_features, kernel_size=opt.kernel_size, d_model=opt.d_model,
             num_stage=opt.num_stage, dct_n=opt.dct_n)
    model.to(opt.device)

    test_dataset_list = []
    for scene in scenes:
        test_dataset_list.append(
            Data(dataset='MI-Motion', train_mode=False, scene=scene, device=device, input_time=opt.input_time,
                 output_time=opt.output_time))

    print(">>> training params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))


    loss_min = 100    
    checkpoint = torch.load('baselines/HRI/checkpoints/pretrained_hri.model', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model loaded.')
    print(model)
    print(f"best_epoch: {checkpoint['epoch']}")


    """
                     ==================================
                        Validating Processing
                     ==================================
    """
    print("\033[0:35mEvaluating.....\033[m")
    with torch.no_grad():
        sample_idx = [15,25,20,1,30]  #  for visualization
        # sample_idx = [1,30]  #  for visualization
        for i, scene in enumerate(scenes):
            test_dataloader = DataLoader(test_dataset_list[i],
                                        batch_size=1,
                                        shuffle=False, drop_last=False)

            ret = run_model(is_train=1, model=model, dataloader=test_dataloader, opt=opt, scene=scene, sample_idx=sample_idx[i])
            print_errors(scene, ret['gjpe_err'], ret['ajpe_err'], ret['root_err'])




        
