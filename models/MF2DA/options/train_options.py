from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--data_aug', action='store_true',
                            help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam',
                            help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true',
                            help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=400,
                            help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count')
        parser.add_argument('--last_epoch', type=int, default=-1,
                            help='starting epoch count for scheduler initialization')
        parser.add_argument('--niter', type=int, default=20,
                            help='total epochs')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001,
                            help='initial learning rate for adam')
        parser.add_argument('--val_real_dirs', type=str, default='',
                    help='comma-separated validation real image directories')
        parser.add_argument('--val_fake_dirs', type=str, default='',
                            help='comma-separated validation fake image directories')
        parser.add_argument('--val_real_num', type=int, default=0,
                            help='max number of validation real images, 0 means all')
        parser.add_argument('--val_fake_num', type=int, default=0,
                            help='max number of validation fake images, 0 means all')
        parser.add_argument('--benchmark_warmup', type=int, default=10,
                            help='number of warmup steps for efficiency benchmarking')
        parser.add_argument('--benchmark_steps', type=int, default=30,
                            help='number of timed steps for efficiency benchmarking')

        self.isTrain = True
        return parser
