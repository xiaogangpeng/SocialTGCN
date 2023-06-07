from torch.nn import Module

from baselines.HRI.GCN import *
from baselines.HRI.util import *


class AttModel(Module):

    def __init__(self, in_features=54, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 5
        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=3,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=3,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

    def forward(self, src, input_n=25,  output_n=25,
                itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:

        :return:
        """

        # B, T, D
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]  32 50 45
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()  # 192 45 12
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # 192 45 12

        dct_m, idct_m = get_dct_matrix(self.kernel_size + output_n)   # one sub in DCT
        dct_m = torch.from_numpy(dct_m).float().to(src.device)
        idct_m = torch.from_numpy(idct_m).float().to(src.device)

        vn = input_n - self.kernel_size - output_n + 1  # 31      N - M - T + 1  number of subs
        # input_n is input seq len,   M is len of sub_seq,  number of subs is vn

        vl = self.kernel_size + output_n   # 20   M + T
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)      # idx.shape  31, 20
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])    # 1920 6 45
        # (1, 20, 20)  (32*31, 20, 66)
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # B, 6, D * dct_n

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        # -10 -> -1, [-1]*10
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)   # B, D, 10

        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)

            # q {32,256,1}  k {32, 256,31}
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15

            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))   # v 192 10 270
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])   # 32, 66, 20

            input_gcn = src_tmp[:, idx]     # 32, 20 frames, 66 !!! last 10 frames+ 10*last frames
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)  # 32 66 40  D & U
            dct_out_tmp = self.gcn(dct_in_tmp)  # 32 66 40
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))    # 32 5 1 66
            if itera > 1:
                # update key-value query1
                out_tmp = out_gcn.clone()[:, 0 - output_n:]

                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)  # 192 20 45


                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)


                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)

                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])

                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)
        outputs = torch.cat(outputs, dim=2)  # 32 20 1 66  predicted value
        return outputs



