import os
import logging
import warnings
import torch

warnings.filterwarnings('ignore') # sklearn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from transformers import logging
logging.set_verbosity_error()

import sys

temp_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
if '--TEMP_DIR' in sys.argv:
    ind = sys.argv.index('--TEMP_DIR')
    temp_dir = sys.argv[ind + 1]
os.environ["TMPDIR"] = temp_dir
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir

import argparse
import platform
import numpy as np
from CEPC.src.ELib import ELib
from CEPC.src.EDomainAdaptProj import EDomainAdaptProj

def main():
    parser = argparse.ArgumentParser()
    # general params
    parser.add_argument("--cmd", default=None, type=str, required=True, help='')
    parser.add_argument("--TEMP_DIR", default=temp_dir, type=str, required=False, help='')

    # domain adaptation
    parser.add_argument("--itr", default=1, type=int, required=False, help='')
    parser.add_argument("--model_path", default=None, type=str, required=True, help='')
    parser.add_argument("--data_path", default=None, type=str, required=True, help='')
    parser.add_argument("--output_dir", default=None, type=str, required=True, help='')
    parser.add_argument("--device", default=None, type=int, required=True, help='')
    parser.add_argument("--device_2", default=None, type=int, required=False, help='')
    parser.add_argument("--seed", default=None, type=int, required=True, help='')
    parser.add_argument("--tgt_d", default=None, type=str, required=False, help='')
    parser.add_argument("--src_d", default=None, type=str, required=False, help='')
    parser.add_argument("--cache_dir", default=None, type=str, required=False, help='')
    parser.add_argument("--flag", default=None, type=str, required=False, help='')

    args, unknown = parser.parse_known_args()

    device = 'cpu'
    device_name = device
    if args.device >= 0:
        device = 'cuda:' + str(args.device)
        device_name = torch.cuda.get_device_name(args.device)
    device_2 = 'cpu'
    if 'device_2' in args and (args.device_2 is not None and args.device_2 >= 0):
        device_2 = 'cuda:' + str(args.device_2)
        device_name = device_name + ', ' + torch.cuda.get_device_name(args.device)
    print('setup:',
          '| python>', platform.python_version(),
          '| numpy>', np.__version__,
          '| pytorch>', torch.__version__,
          '| device>', device_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.cmd.startswith('da_'):
        EDomainAdaptProj.run(args.cmd, args.itr, args.model_path, args.data_path, args.output_dir,
                             device, device_2, args.seed, args.tgt_d, args.src_d, args.cache_dir, args.flag)
        ELib.PASS()
    ELib.PASS()

if __name__ == "__main__":
    print("Started at", ELib.get_time())
    main()
    print("\nDone at", ELib.get_time())
    pass
