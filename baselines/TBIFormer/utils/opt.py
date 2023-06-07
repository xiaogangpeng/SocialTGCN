import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):

        self.parser.add_argument('--input_time', type=int, default=25, help='size of each model layer')
        self.parser.add_argument('--output_time', type=int, default=25, help='size of each model layer')
        self.parser.add_argument('--joints_num', type=int, default=18, help='number of skeleton joints')
        self.parser.add_argument('--rendering_imgs', type=int, default=0)
        self.parser.add_argument('--ultra_long_term_prediction', type=int, default=0)


        # ===============================================================
        #                     Model options
        # ===============================================================

        self.parser.add_argument('--d_model', type=int, default=128, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=2, help='layers in linear model')
        self.parser.add_argument('--d_inner', type=int, default=1024, help='dim for inner ')
        self.parser.add_argument('--n_head', type=int, default=8, help='dim for inner ')
        self.parser.add_argument('--d_k', type=int, default=64, help='dim for inner ')
        self.parser.add_argument('--d_v', type=int, default=64, help='dim for inner ')
        self.parser.add_argument('--kernel_size', type=int, default=5)
        self.parser.add_argument('--stride', type=int, default=1)



        # ===============================================================
        #                     Running options
        # ===============================================================

        self.parser.add_argument('--lr', type=float, default=0.0003)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.2,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=32)
        self.parser.add_argument('--test_batch', type=int, default=1)
        self.parser.add_argument('--device', type=str, default='cuda:0')
        self.parser.add_argument('--seed', type=int, default=10)
        self.parser.add_argument('--theta', type=int, default=2000)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt
