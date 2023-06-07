import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--input_time', type=int, default=25, help='size of each model layer')
        self.parser.add_argument('--output_time', type=int, default=25, help='size of each model layer')
        self.parser.add_argument('--joints_num', type=int, default=18, help='number of skeleton joints')
        self.parser.add_argument('--rendering_imgs', type=int, default=0)
        self.parser.add_argument('--ultra_long_term_prediction', type=int, default=0)
        # ===============================================================
        #                     Model options
        # ===============================================================

        self.parser.add_argument('--kernel_size', type=int, default=10)
        self.parser.add_argument('--stride', type=int, default=1)
        
        self.parser.add_argument('--model', default='GCN', help='which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN, HyboNet]')
        self.parser.add_argument('--gcn-enc-layers', default=12, help='gcn layers of each block')
        self.parser.add_argument('--tcn-dec-layers', default=4, help='layers of tcn decoder')
        self.parser.add_argument('--act', default='leaky_relu', help='which activation function to use (or None for no activation)   relu   prelu  ')
        self.parser.add_argument('--bias', default=1, help='whether to use bias (1) or not (0)')


        # ===============================================================
        #                     Running options
        # ===============================================================

        self.parser.add_argument('--lr', type=float, default=0.0003)
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.2,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train-batch', type=int, default=32)
        self.parser.add_argument('--test-batch', type=int, default=1)
        self.parser.add_argument('--device', type=str, default='cuda:0')
        self.parser.add_argument('--seed', type=int, default=10)


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt
