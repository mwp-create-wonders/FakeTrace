import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14', help='see my_models/__init__.py')
        parser.add_argument('--fix_backbone', action='store_true')

        # 数据增强
        parser.add_argument('--rz_interp', type=str, default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0.5)
        parser.add_argument('--blur_sig', type=str, default='0.0,3.0')
        parser.add_argument('--jpg_prob', type=float, default=0.5)
        parser.add_argument('--jpg_method', type=str, default='cv2,pil')
        parser.add_argument('--jpg_qual', type=str, default='30,100')

        # =========================================================
        # 显式指定多目录输入
        # =========================================================
        parser.add_argument(
            '--real_dirs_train',
            type=str,
            required=True,
            help='训练集真实样本目录，多个目录用英文逗号分隔'
        )
        parser.add_argument(
            '--fake_dirs_train',
            type=str,
            required=True,
            help='训练集生成样本目录，多个目录用英文逗号分隔'
        )
        parser.add_argument(
            '--real_dirs_val',
            type=str,
            required=True,
            help='验证集真实样本目录，多个目录用英文逗号分隔'
        )
        parser.add_argument(
            '--fake_dirs_val',
            type=str,
            required=True,
            help='验证集生成样本目录，多个目录用英文逗号分隔'
        )

        parser.add_argument(
            '--real_num_per_dir_train',
            type=str,
            default=None,
            help='训练集每个真实目录采样数，多个值用英文逗号分隔，例如 10000,10000'
        )
        parser.add_argument(
            '--fake_num_per_dir_train',
            type=str,
            default=None,
            help='训练集每个生成目录采样数，多个值用英文逗号分隔，例如 5000,5000,5000,5000'
        )
        parser.add_argument(
            '--real_num_per_dir_val',
            type=str,
            default=None,
            help='验证集每个真实目录采样数，多个值用英文逗号分隔'
        )
        parser.add_argument(
            '--fake_num_per_dir_val',
            type=str,
            default=None,
            help='验证集每个生成目录采样数，多个值用英文逗号分隔'
        )

        parser.add_argument('--data_label', type=str, default='train',
                            help='train 或 val，用于决定加载哪组目录')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='loss weight for l2 reg')

        # dataloader / image process
        parser.add_argument('--class_bal', action='store_true', help='是否启用类别平衡采样')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 / 0,1,2 / -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--num_threads', type=int, default=4, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='是否顺序采样')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop',
                            help='scaling and cropping of images at load time')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--no_resize', action='store_true',
                            help='if specified, do not resize the images')
        parser.add_argument('--no_crop', action='store_true',
                            help='if specified, do not crop the images')

        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', type=str, default='',
                            help='customized suffix: opt.name = opt.name + suffix')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, mode=0o777, exist_ok=True)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def _split_str_list(self, s):
        if s is None:
            return None
        if isinstance(s, list):
            return s
        s = s.strip()
        if s == '':
            return []
        return [item.strip() for item in s.split(',') if item.strip() != '']

    def _split_int_list(self, s):
        if s is None:
            return None
        if isinstance(s, list):
            return [int(x) for x in s]
        s = s.strip()
        if s == '':
            return None
        return [int(item.strip()) for item in s.split(',') if item.strip() != '']

    def _check_dir_list_length(self, dirs, nums, name):
        if dirs is None:
            return
        if nums is not None and len(dirs) != len(nums):
            raise ValueError(
                f'Length mismatch for {name}: number of dirs = {len(dirs)}, '
                f'but number list = {len(nums)}'
            )

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            idx = int(str_id)
            if idx >= 0:
                opt.gpu_ids.append(idx)

        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_ids[0])

        # =========================================================
        # 解析增强参数
        # =========================================================
        opt.rz_interp = self._split_str_list(opt.rz_interp)
        opt.blur_sig = [float(s) for s in self._split_str_list(opt.blur_sig)]
        opt.jpg_method = self._split_str_list(opt.jpg_method)
        opt.jpg_qual = [int(s) for s in self._split_str_list(opt.jpg_qual)]

        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        # =========================================================
        # 解析多目录数据参数
        # =========================================================
        opt.real_dirs_train = self._split_str_list(opt.real_dirs_train)
        opt.fake_dirs_train = self._split_str_list(opt.fake_dirs_train)
        opt.real_dirs_val = self._split_str_list(opt.real_dirs_val)
        opt.fake_dirs_val = self._split_str_list(opt.fake_dirs_val)

        opt.real_num_per_dir_train = self._split_int_list(opt.real_num_per_dir_train)
        opt.fake_num_per_dir_train = self._split_int_list(opt.fake_num_per_dir_train)
        opt.real_num_per_dir_val = self._split_int_list(opt.real_num_per_dir_val)
        opt.fake_num_per_dir_val = self._split_int_list(opt.fake_num_per_dir_val)

        # 长度检查
        self._check_dir_list_length(opt.real_dirs_train, opt.real_num_per_dir_train, 'real_dirs_train')
        self._check_dir_list_length(opt.fake_dirs_train, opt.fake_num_per_dir_train, 'fake_dirs_train')
        self._check_dir_list_length(opt.real_dirs_val, opt.real_num_per_dir_val, 'real_dirs_val')
        self._check_dir_list_length(opt.fake_dirs_val, opt.fake_num_per_dir_val, 'fake_dirs_val')

        if print_options:
            self.print_options(opt)

        self.opt = opt
        return self.opt