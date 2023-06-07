''' Define the model '''
from . import encoders
import numpy as np
import torch
import torch.nn as nn
from .encoders import SkelGCN
from scipy.spatial.distance import pdist, squareform


class SGCN_layer(nn.Module):
    def __init__(self, args,in_channels=3, out_channels=3):
        super(SGCN_layer, self).__init__()
        self.encoder = getattr(encoders, args.model)(args, in_channels, out_channels)
    def forward(self, x, A):
        x = x.permute(0, 2, 3, 1)
        out = self.encoder.encode(x, A)
        return out, A



class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        
        
        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)] 
             
        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):
        
        output= self.block(x)
        return output


class SocialTGCN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self, args,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dropout,
                 bias=True):
        
        super(SocialTGCN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)


        self.gcn = SGCN_layer(args, in_channels=in_channels, out_channels=out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )            
        
        if stride != 1 or in_channels != out_channels:

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.residual=nn.Identity()
        
          

    def forward(self, x, adj):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
     #    res=self.residual(x)
        x, adj = self.gcn(x, adj)
        x = x.permute(0, 3, 1, 2)
        x = self.tcn(x)
        # x=x+res
        return x, adj




class SocialTGCN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, args=None,  output_time_frame=25, joints_to_consider=15):
        super().__init__()
        

        self.joints_to_consider = joints_to_consider

        self.social_tgcns=nn.ModuleList()
        self.social_tgcns.append(SocialTGCN_layer(args, self.joints_to_consider*3, 512,[1,1],1,args.dropout))
        self.social_tgcns.append(SocialTGCN_layer(args, 512, 512,[1,1],1,args.dropout))
        self.social_tgcns.append(SocialTGCN_layer(args, 512, self.joints_to_consider*3,[1,1],1,args.dropout))


        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.fn = int((args.input_time-1 - self.kernel_size) / self.stride + 1)


        self.conv = nn.Conv1d(self.joints_to_consider*3, self.joints_to_consider*3, kernel_size=self.kernel_size)
        self.skel_gcn = SkelGCN(input_feature=self.fn, out_feature=self.fn, hidden_feature=256, p_dropout=args.dropout,
                           num_stage=args.gcn_enc_layers,
                           node_n=self.joints_to_consider*3)

        txc_kernel_size = [3, 3]
        txc_dropout = 0.0       
        self.txcnns=nn.ModuleList()
        self.n_txcnn_layers = args.tcn_dec_layers
        self.txcnns.append(CNN_layer(self.fn, output_time_frame,txc_kernel_size,txc_dropout))
        for i in range(1,self.n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))
        self.prelus = nn.ModuleList()
        for j in range(self.n_txcnn_layers):
            self.prelus.append(nn.PReLU())



    def forward(self, input_src, root_pos):
        '''
        src_seq:  B*N, T, J*3
        '''

        bs, n, t, _ = input_src.shape
        x = input_src.reshape(bs * n, t, -1)

        idx = np.expand_dims(np.arange(self.kernel_size), axis=0) + \
              np.expand_dims(np.arange(self.fn), axis=1) * self.stride
        x = x[:, idx, :].reshape(bs * n * self.fn, self.kernel_size, -1)

        root_pos = root_pos[:, idx, :].reshape(bs * n * self.fn, self.kernel_size, -1)

        # Constructing ajacency martix
        root_pos = root_pos.mean(1).reshape(bs * self.fn, n, -1).detach().cpu().numpy()
        dis_list = []
        for k in range(bs * self.fn):
            dist = pdist(root_pos[k], metric='euclidean').astype(np.float32)
            dist_ = squareform(dist)
            A = np.zeros_like(dist_)
            for i in range(n):
                A[i, i] = 1
                for j in range(i + 1, n):
                    # A[i, j] = 1.0 / (dist_[i, j]+2)
                    # A[j, i] = 1.0 / (dist_[i, j]+2)
                    A[i, j] = np.exp(-dist_[i, j] * 100)
                    A[j, i] = np.exp(-dist_[i, j] * 100)
            dis_list.append(np.expand_dims(A, 0))
        adj = np.concatenate(dis_list, axis=0).reshape(bs, self.fn, n, n)
        adj = torch.from_numpy(adj).to(x.device)
        # adj = adj.repeat_interleave(self.joints_to_consider, dim=3).repeat_interleave(self.joints_to_consider, dim=2)

        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(bs * n, self.fn, -1)
        x = self.skel_gcn(x.permute(0, 2, 1))  # (B*N, C, T)

        res_x = x.reshape(bs, n, self.fn, -1)

        x_ = res_x.permute(0, 3, 2, 1)
        #  ST-GCN Encoder
        for gcn in (self.social_tgcns):
            x_, adj = gcn(x_, adj)  # (B, C, T, N*J)


        out = res_x + x_.permute(0, 3, 2, 1)

        # out = res_x

        #  Decoder
        out = out.reshape(bs, n, self.joints_to_consider * 3, -1).permute(0, 3, 1, 2)  # (B, N, C, T)->(B, T, N, C)
        out = self.prelus[0](self.txcnns[0](out))
        for i in range(1, self.n_txcnn_layers):
            out = self.prelus[i](self.txcnns[i](out)) + out  # temporal conv + residual connection

        return out.permute(0, 2, 1, 3).reshape(bs * n, -1, self.joints_to_consider * 3)







