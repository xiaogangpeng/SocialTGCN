#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):


        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--ultra_long_term_prediction', type=int, default=0)
        self.parser.add_argument('--rendering_imgs', type=int, default=0)

        self.parser.add_argument('--in_features', type=int, default=54, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=256, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=5, help='past frame number')
        # self.parser.add_argument('--drop_out', type=float, default=0.5, help='drop out probability')

        # ===============================================================
        #                     Running options     
        # ===============================================================
        self.parser.add_argument('--input_time', type=int, default=25, help='size of each model layer')
        self.parser.add_argument('--output_n', type=int, default=5, help='future frame number')
        self.parser.add_argument('--output_time', type=int, default=25, help='size of each model layer')
        self.parser.add_argument('--joints_num', type=int, default=18, help='number of skeleton joints')

        self.parser.add_argument('--dct_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--lr_now', type=float, default=0.0003)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--train_batch', type=int, default=32)
        self.parser.add_argument('--test_batch', type=int, default=1)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')
        self.parser.add_argument('--device', type=str, default='cuda:0')
        self.parser.add_argument('--seed', type=int, default=10)
    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        return self.opt
