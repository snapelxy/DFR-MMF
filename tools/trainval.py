import argparse
import shutil
import time
from pathlib import Path


from vedacore.misc import Config, mkdir_or_exist, set_random_seed
from vedacore.parallel import init_dist
from vedadet.assembler import trainval
from vedadet.misc import get_root_logger
from vedacore.misc import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--cfg_file', default="cfgs\\rd4ad_reg_800my_booam_512.py", help='train config file path')
    parser.add_argument('--workdir', help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=1)  # TODO
    parser.add_argument('--mode', type=str, default="train")  # TODO

    args = parser.parse_args()
    return args


def main(args):
    args.cfg_file = Path(args.cfg_file)
    cfg = Config.fromfile(args.cfg_file)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # workdir is determined in this priority: CLI > segment in file > filename
    if args.workdir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.workdir = args.workdir
    elif cfg.get('workdir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.workdir = Path('workdir')/args.cfg_file.stem
        cfg.workdir.resolve()

    seed = cfg.get('seed', None)
    deterministic = cfg.get('deterministic', False)
    set_random_seed(seed, deterministic)

    # create work_dir
    mkdir_or_exist(cfg.workdir)
    shutil.copy(args.cfg_file, cfg.workdir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = cfg.workdir/f'{timestamp}.log'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)



    trainval(cfg, distributed, logger, args.mode)


if __name__ == '__main__':
    args = parse_args()
    main(args)
