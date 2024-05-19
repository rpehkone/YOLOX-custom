from loguru import logger
import random
import warnings

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices


@logger.catch
def main(exp: Exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()

class Args:
    def __init__(self):
        self.experiment_name=None
        self.name="yolox-s"#model name
        self.exp_file="exps/drone_s.py"#experiment description file
        self.ckpt=None
        self.logger="wandb"
        # self.logger="tensorboard"
        self.cache="ram"#Caching imgs to ram/disk for fast training.
        # self.cache=None#Caching imgs to ram/disk for fast training.
        # self.dist_backend="nccl"#distributed backend
        self.dist_backend=None
        self.dist_url=None#distributed training url
        self.batch_size=32
        self.devices=1#device for training
        self.resume=False
        self.start_epoch=None
        self.num_machines=1
        self.machine_rank=0#node rank for multi-node training
        self.fp16=False
        self.occupy=False#Allocate needed GPU memory before training.
        self.opts=[]

if __name__ == "__main__":
    configure_module()
    args = Args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
