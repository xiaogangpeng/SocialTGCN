import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
from HRI.AttnModel import AttModel
from HRI.opt import Options
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


def run_model(is_train, model,dataloader, optimizer, opt):
    train_loss_list = []
    test_loss_list = []
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

        input_seq = input_seq.reshape(bs*n, -1, jn*d)
        output_seq = output_seq.reshape(bs*n, -1, jn*d)
        
        input_src = input_seq.clone()  # 192 15 45
        out_all = model(input_src, input_n=opt.input_time, output_n=opt.output_n, itera=5)  # 32 20 1 66
        out = out_all[:, opt.kernel_size:, :].reshape(bs * n, -1, jn*d)  # 192 45 45

        gt_poses = output_seq.view(bs, n, -1, jn, d)
        pred_poses = out.view(bs, n, -1, jn, d)
    

        # print(f"gt:{gt_poses.shape}  pred:{ pred_poses.shape}")
        # 3d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss = torch.mean(torch.norm(out[:, :opt.output_time:, ...] - output_seq, dim=2))
            loss_all = loss
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l = loss.cpu().data.numpy()
            train_loss_list.append(l.item())

        if is_train == 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d = torch.mean(torch.norm(out[:, :opt.output_time, ...] - output_seq, dim=2)).detach().cpu()
            m_p3d = mpjpe_p3d.data.numpy()
            test_loss_list.append(m_p3d.item())

            gjpe_err = GJPE(gt_poses, pred_poses, frame_idx)
            ajpe_err = AJPE(gt_poses, pred_poses, frame_idx)
            root_err = RFDE(gt_poses, pred_poses, frame_idx)
            gjpe_err_total += gjpe_err
            ajpe_err_total += ajpe_err
            root_err_total += root_err
       


    ret = {}
    if is_train == 0:
        ret["loss_3d"] = np.mean(train_loss_list)
    elif is_train == 1:
        ret["loss_3d"] = np.mean(test_loss_list)
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

    model = AttModel(in_features=opt.in_features, kernel_size=opt.kernel_size, d_model=opt.d_model,
             num_stage=opt.num_stage, dct_n=opt.dct_n)
    model.to(opt.device)

    dataset = Data(dataset='MI-Motion', train_mode=True, scene=None, device=device, input_time=opt.input_time,
                   output_time=opt.output_time)

    test_dataset_list = []
    for scene in scenes:
        test_dataset_list.append(
            Data(dataset='MI-Motion', train_mode=False, scene=scene, device=device, input_time=opt.input_time,
                 output_time=opt.output_time))

    print(stamp)
    dataloader = DataLoader(dataset,
                            batch_size=opt.train_batch,
                            shuffle=True, drop_last=True)

    print(">>> training params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))

    Evaluate = True
    save_model = True

    params = [
        {"params": model.parameters(), "lr": opt.lr_now}
    ]
    optimizer = optim.Adam(params)


    loss_min = 100
    for epoch_i in range(1, opt.epochs + 1):
        """
        ==================================
           Training Processing
        ==================================
        """
        ret = run_model(is_train=0, model=model,  dataloader=dataloader, optimizer=optimizer, opt=opt)

        train_loss = ret['loss_3d']
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_i
        }
        print('epoch:', epoch_i, 'loss:', train_loss, "lr: {:.10f} ".format(optimizer.param_groups[0]['lr']))

        if save_model:
            save_path = os.path.join('baselines/HRI/checkpoints', f'epoch_{epoch_i}.model')
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
                                                 shuffle=False, drop_last=False)

                    ret = run_model(is_train=1, model=model,  dataloader=test_dataloader, optimizer=None, opt=opt)
                    print_errors(scene, ret['gjpe_err'], ret['ajpe_err'], ret['root_err'])

                test_loss_cur = ret['loss_3d']

                if test_loss_cur < loss_min:
                    save_path = os.path.join('baselines/HRI/checkpoints', f'best_epoch.model')
                    torch.save(checkpoint, save_path)
                    loss_min = test_loss_cur
                    print(f"Best epoch_{checkpoint['epoch']} model is saved!")

if __name__ == '__main__':
    option = Options().parse()
    processor(option)






