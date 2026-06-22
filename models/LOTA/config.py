import argparse
import os
import torch


class ConfigurationManager:
    def __init__(self):
        self.config_initialized = False
        self.argument_parser = None

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1', 'y'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0', 'n'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    def parse_dir_list(self, value):
        """
        将命令行中的目录参数解析为 list
        支持:
        1. 单个目录
        2. 多个目录用英文逗号分隔
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value

        value = value.strip()
        if value == "":
            return []

        return [item.strip() for item in value.split(",") if item.strip()]

    def merge_dir_lists(self, *dir_groups):
        merged_dirs = []
        seen = set()

        for dir_group in dir_groups:
            for directory in self.parse_dir_list(dir_group):
                normalized_directory = os.path.normpath(directory)
                if normalized_directory not in seen:
                    merged_dirs.append(directory)
                    seen.add(normalized_directory)

        return merged_dirs

    def resolve_child_dirs(self, parent_dir, child_names, select_num=0, use_parent_when_no_child_names=False):
        parent_dir = (parent_dir or "").strip()
        if not parent_dir:
            return []

        if not os.path.isdir(parent_dir):
            raise ValueError(f"Parent directory not found: {parent_dir}")

        child_names = self.parse_dir_list(child_names)

        if child_names:
            resolved_dirs = []
            missing_children = []

            for child_name in child_names:
                child_path = os.path.join(parent_dir, child_name)
                if os.path.isdir(child_path):
                    resolved_dirs.append(child_path)
                else:
                    missing_children.append(child_name)

            if missing_children:
                missing_str = ", ".join(missing_children)
                raise ValueError(
                    f"Cannot find the following child directories under {parent_dir}: {missing_str}"
                )
        elif use_parent_when_no_child_names:
            resolved_dirs = [parent_dir]
        else:
            resolved_dirs = sorted(
                os.path.join(parent_dir, item_name)
                for item_name in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, item_name))
            )

        try:
            select_num = int(select_num or 0)
        except (TypeError, ValueError):
            select_num = 0

        if select_num > 0:
            resolved_dirs = resolved_dirs[:select_num]

        return resolved_dirs

    def define_arguments(self, argument_parser):
        """Define all configuration parameters"""

        # =========================
        # Basic training parameters
        # =========================
        argument_parser.add_argument(
            '--batchsize', type=int, default=64,
            help='Number of samples processed simultaneously'
        )
        argument_parser.add_argument(
            '--val_batchsize', type=int, default=64,
            help='Batch size for validation/testing'
        )
        argument_parser.add_argument(
            '--epoch', type=int, default=30,
            help='Total training epochs'
        )
        argument_parser.add_argument(
            '--lr', type=float, default=1e-4,
            help='Initial learning rate'
        )
        argument_parser.add_argument(
            '--load', type=str, default=None,
            help='Path to pre-trained model weights'
        )
        argument_parser.add_argument(
            '--save_path', type=str,
            default='/home/hdd1/chengrenxi/sdv5_thresholding2/',
            help='Directory for saving model outputs'
        )

        # =========================
        # Data directory parameters
        # =========================
        # 训练集：支持多个真实目录 / 多个生成目录
        argument_parser.add_argument(
            '--train_real_dirs', type=str, default='',
            help='Training real-image directories, separated by commas'
        )
        argument_parser.add_argument(
            '--train_fake_dirs', type=str, default='',
            help='Training fake-image directories, separated by commas'
        )

        # 验证集：支持多个真实目录 / 多个生成目录
        argument_parser.add_argument(
            '--val_real_dirs', type=str, default='',
            help='Validation real-image directories, separated by commas'
        )
        argument_parser.add_argument(
            '--val_fake_dirs', type=str, default='',
            help='Validation fake-image directories, separated by commas'
        )

        # 测试集：后续 test.py 也可以直接复用
        argument_parser.add_argument(
            '--test_real_dirs', type=str, default='',
            help='Test real-image directories, separated by commas'
        )
        argument_parser.add_argument(
            '--test_fake_dirs', type=str, default='',
            help='Test fake-image directories, separated by commas'
        )
        argument_parser.add_argument(
            '--test_real_parent_dir', type=str, default='',
            help='Parent directory used to resolve test real-image child directories'
        )
        argument_parser.add_argument(
            '--test_real_child_names', type=str, default='',
            help='Comma-separated child directory names under --test_real_parent_dir'
        )
        argument_parser.add_argument(
            '--test_real_select_num', type=int, default=0,
            help='Use the first N resolved real child directories; 0 means use all'
        )
        argument_parser.add_argument(
            '--test_fake_parent_dir', type=str, default='',
            help='Parent directory used to resolve test fake-image child directories'
        )
        argument_parser.add_argument(
            '--test_fake_child_names', type=str, default='',
            help='Comma-separated child directory names under --test_fake_parent_dir'
        )
        argument_parser.add_argument(
            '--test_fake_select_num', type=int, default=0,
            help='Use the first N resolved fake child directories; 0 means use all'
        )

        argument_parser.add_argument(
            '--recursive', action='store_true',
            help='Recursively search images inside subdirectories'
        )
        argument_parser.add_argument(
            '--test_num', type=int, default=2000,
            help='Maximum number of real samples and fake samples used in each test; use all if insufficient'
        )

        # =========================
        # Image preprocessing parameters
        # =========================
        argument_parser.add_argument(
            '--isPatch', type=self.str2bool, default=True,
            help='Enable patch processing mode (true/false)'
        )
        argument_parser.add_argument(
            '--img_height', type=int, default=256,
            help='Target image height'
        )
        argument_parser.add_argument(
            '--bit_mode', type=str, default='scaling',
            choices=['scaling', 'thresholding'],
            help='Bit plane processing method'
        )
        argument_parser.add_argument(
            '--patch_size', type=int, default=32,
            help='Dimension for patch extraction'
        )
        argument_parser.add_argument(
            '--patch_mode', type=str, default='max',
            choices=['max', 'min', 'random'],
            help='Patch selection strategy'
        )

        # =========================
        # Device parameters
        # =========================
        argument_parser.add_argument(
            '--gpu_id', type=str, default='0',
            help='Identifier for GPU device'
        )

        # =========================
        # Compatibility parameters
        # =========================
        # 保留旧字段，避免你其他代码引用时报错
        argument_parser.add_argument(
            '--choices', nargs='*', type=int, default=[1, 1, 1, 1, 1, 1, 1, 1],
            help='Deprecated. Kept only for compatibility.'
        )
        argument_parser.add_argument(
            '--image_root', type=str, default='',
            help='Deprecated. Kept only for compatibility.'
        )

        return argument_parser

    def collect_arguments(self):
        """Gather and process command-line arguments"""
        if not self.config_initialized:
            argument_parser = argparse.ArgumentParser(
                description='Model Training Configuration',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            argument_parser = self.define_arguments(argument_parser)
            self.config_initialized = True
            self.argument_parser = argument_parser

        config, _ = self.argument_parser.parse_known_args()
        return self.argument_parser.parse_args()

    def postprocess_arguments(self, config):
        """
        对解析后的参数进行后处理：
        1. 将逗号分隔目录字符串转为 list
        2. 设置设备
        3. 创建保存目录
        """

        # 将目录参数统一转换成 list[str]
        config.train_real_dirs = self.parse_dir_list(config.train_real_dirs)
        config.train_fake_dirs = self.parse_dir_list(config.train_fake_dirs)
        config.val_real_dirs = self.parse_dir_list(config.val_real_dirs)
        config.val_fake_dirs = self.parse_dir_list(config.val_fake_dirs)
        config.test_real_dirs = self.parse_dir_list(config.test_real_dirs)
        config.test_fake_dirs = self.parse_dir_list(config.test_fake_dirs)
        config.test_real_child_names = self.parse_dir_list(config.test_real_child_names)
        config.test_fake_child_names = self.parse_dir_list(config.test_fake_child_names)

        resolved_test_real_dirs = self.resolve_child_dirs(
            config.test_real_parent_dir,
            config.test_real_child_names,
            config.test_real_select_num,
            use_parent_when_no_child_names=True
        )
        resolved_test_fake_dirs = self.resolve_child_dirs(
            config.test_fake_parent_dir,
            config.test_fake_child_names,
            config.test_fake_select_num
        )

        config.test_real_dirs = self.merge_dir_lists(config.test_real_dirs, resolved_test_real_dirs)
        config.test_fake_dirs = self.merge_dir_lists(config.test_fake_dirs, resolved_test_fake_dirs)

        # 设置 CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建保存目录
        if config.save_path is not None and config.save_path != "":
            os.makedirs(config.save_path, exist_ok=True)

        return config

    def display_configuration(self, config):
        """Present configuration details in a structured format"""
        configuration_details = []
        configuration_details.append("╔════════════════════════════════════════════════════════════╗")
        configuration_details.append("║                   TRAINING CONFIGURATION                  ║")
        configuration_details.append("╠════════════════════════════════════════════════════════════╣")

        for parameter, value in sorted(vars(config).items()):
            configuration_details.append(
                f"║ {parameter:>20}: {str(value):<35}║"
            )

        configuration_details.append("╚════════════════════════════════════════════════════════════╝")
        print("\n".join(configuration_details))

    def parse(self, display_settings=True):
        """Parse and return configuration settings"""
        config = self.collect_arguments()
        config = self.postprocess_arguments(config)

        config.isTrain = True
        config.isVal = False

        if display_settings:
            self.display_configuration(config)

        self.config = config
        return self.config
