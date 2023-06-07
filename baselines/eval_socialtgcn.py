import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch_dct as dct  # https://github.com/zh217/torch-dct
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from SocialTGCN.net import SocialTGCN
from SocialTGCN.utils.opt import Options
from util.dataloader import Data
from util.metrics import *
from util.plot_pose import plot_pose,save_imgs
from fvcore.nn import FlopCountAnalysis
import sys
sys.path.append('..')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_model(is_train, model, dataloader, opt, scene, sample_idx):
    all_gt_poses = torch.zeros(0)
    all_pred_poses = torch.zeros(0)
    all_test_poses = torch.zeros(0)
    batch_num = 0
    frame_idx = [2, 4, 8, 10, 14, 18, 25]
    gjpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
    ajpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
    root_err_total = np.arange(len(frame_idx), dtype=np.float_)

    if is_train == 0:
        model.train()
    else:
        model.eval()

    for batch_i, batch_data in tqdm(enumerate(dataloader)):
        batch_num += 1
        input_seq, output_seq = batch_data
        long_output_seq = output_seq[:,:,opt.output_time:,:,:]
        output_seq = output_seq[:,:,:opt.output_time,:, :]
        bs, n, t, jn, d = input_seq.shape
        input_ = input_seq.reshape(bs * n, opt.input_time, -1)
        output_ = torch.cat((input_[:, -1:, :], output_seq.reshape(bs * n, opt.output_time, -1)), 1)

        motion_offset = input_[:, 1:opt.input_time, :] - input_[:, :opt.input_time - 1, :]
        input_src = dct.dct(motion_offset).view(bs, n, -1, jn * d)

        root_pos = input_[:, :-1, :3]


        rec_ = model.forward(input_src, root_pos)
        rec = dct.idct(rec_)
        results = output_[:, :1, :]
        for i in range(1, opt.output_time + 1):
            results = torch.cat(
                [results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)],
                dim=1)
        results = results[:, 1:, :]  # 3 15 45

        pred_poses = results.reshape(bs, n, opt.output_time, jn, -1)
        gt_poses = output_seq

        if is_train == 1:
            gjpe_err = GJPE(gt_poses, pred_poses, frame_idx)
            ajpe_err = AJPE(gt_poses, pred_poses, frame_idx)
            root_err = RFDE(gt_poses, pred_poses, frame_idx)
            gjpe_err_total += gjpe_err
            ajpe_err_total += ajpe_err
            root_err_total += root_err
       

        
        # For ultra-long-term prediction
        if opt.ultra_long_term_prediction:
            # new_input_seq = torch.cat([input_seq, pred_poses], dim=-3)
            new_input_seq = pred_poses
            new_input_ = new_input_seq.reshape(bs * n, opt.output_time, -1)

            new_root_pos = new_input_[:, :-1, :3]   
            new_motion_offset = new_input_[:, 1:, :] - new_input_[:, :-1, :]  
            new_input_src = dct.dct(new_motion_offset).view(bs, n, -1, jn * d)
            new_rec_ = model.forward(new_input_src, new_root_pos)
            new_rec = dct.idct(new_rec_)

            new_results = new_input_seq.reshape(bs*n, opt.output_time, -1)[:, -1:, :]  
            for i in range(1, opt.output_time + 1):
                new_results = torch.cat(
                    [new_results, new_input_seq.reshape(bs*n, opt.input_time, -1)[:, -1:, :] + torch.sum(
                        new_rec[:, :i, :], dim=1, keepdim=True)], dim=1)
            new_results = new_results[:, 1:, :]     

            long_gt_poses = long_output_seq.reshape(n, opt.output_time, jn, -1)
            long_pred_poses = new_results.reshape(n, opt.output_time, jn, -1)

            long_gt_poses = long_output_seq.reshape(n, opt.output_time, jn, -1)  
            long_pred_poses = results.reshape(bs, n, opt.output_time, jn, -1).squeeze()

            all_pred_poses = torch.cat((all_pred_poses, long_pred_poses.detach().cpu()), dim=0)
            all_gt_poses = torch.cat((all_gt_poses, long_gt_poses.detach().cpu()), dim=0)
            temp = torch.cat((input_seq.reshape(n, opt.output_time, jn, -1).detach().cpu(), all_gt_poses), dim=0)
            all_test_poses = torch.cat((all_test_poses, temp), dim=0)


        if opt.rendering_imgs:
            if batch_i == sample_idx:
                print(f"Saving images for batch: {batch_i}")
                if opt.ultra_long_term_prediction:
                    pred_poses_ = torch.cat((pred_poses, long_pred_poses.unsqueeze(0)), dim=2)  
                    gt_poses_ = torch.cat((gt_poses, long_gt_poses.unsqueeze(0)), dim=2)
                    save_imgs(batch_i, 'SocialTGCN', scene, pred_poses_, gt_poses_, input_seq, opt)
                else:
                    save_imgs(batch_i, 'SocialTGCN', scene, pred_poses, gt_poses, input_seq, opt)

            
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
    # scenes = ['Special_Locations',  'Complex_Crowd']
    setup_seed(opt.seed)
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

    model = SocialTGCN(args=opt, output_time_frame=opt.output_time,
                         joints_to_consider=opt.joints_num).to(device)

    test_dataset_list = []
    for scene in scenes:
        test_dataset_list.append(
            Data(dataset='MI-Motion', train_mode=False, scene=scene, device=device, input_time=opt.input_time,
                 output_time=opt.output_time))

    print(stamp)
    print(">>> training params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))


    checkpoint = torch.load('baselines/SocialTGCN/checkpoints/pretrained_socialtgcn.model', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model loaded.')
    print(model)
    print(f"best_epoch: {checkpoint['epoch']}")

    loss_min = 100
    with torch.no_grad():
        """
          ==================================
             Validating Processing
          ==================================
          """
        print("\033[0:35mEvaluating.....\033[m")
        sample_idx = [15,25,20,1,30]  #  for visualization
        for i, scene in enumerate(scenes):
            test_dataloader = DataLoader(test_dataset_list[i],
                                         batch_size=1,
                                         shuffle=False, drop_last=True)

            ret = run_model(is_train=1, model=model, dataloader=test_dataloader, opt=opt, scene=scene, sample_idx=sample_idx[i])
            print_errors(scene, ret['gjpe_err'], ret['ajpe_err'], ret['root_err'])





