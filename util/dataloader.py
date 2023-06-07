import numpy as np
import torch
import torch.utils.data as data


class Data(data.Dataset):
    def __init__(self, dataset, train_mode=True, scene=None, device='cuda', input_time=25, output_time=25):
        scenes = ['Park', 'Street', 'Indoor', 'Special_Locations', 'Complex_Crowd']
        if dataset == "MI-Motion":
            scenes = ['Park', 'Street', 'Indoor', 'Special_Locations', 'Complex_Crowd']
            if train_mode:
                self.data = np.load('./data/'+dataset +'/data_train.npy', allow_pickle=True)
            else:   
                scene_idx = scenes.index(scene)         
                self.data = np.load('./data/'+dataset + '/data_test_S' + str(scene_idx) + '.npy', allow_pickle=True)
        else:
            self.data = np.load('baselines/MRT/discriminator_3_75_mocap.npy')
         
        self.len = len(self.data)
        self.device = device
        self.dataset = dataset
        self.input_time = input_time
        self.output_time = output_time


    def __getitem__(self, index):
        data = self.data[index]
        input_seq = data[:, :self.input_time, ...]
        output_seq = data[:, self.input_time:,...]
        input_seq = torch.as_tensor(input_seq, dtype=torch.float32).to(self.device)
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32).to(self.device)
        
        # last_input = input_seq[:, -1:, :]
        # output_seq = torch.cat([last_input, output_seq], dim=1)
        # input_seq = input_seq.reshape(input_seq.shape[0], input_seq.shape[1], -1)
        # output_seq = output_seq.reshape(output_seq.shape[0], output_seq.shape[1], -1)

        return input_seq, output_seq

    def __len__(self):
        return self.len




