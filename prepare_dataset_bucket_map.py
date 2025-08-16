from utils.utils import instantiate_from_config
from omegaconf import OmegaConf,DictConfig
import argparse
from tqdm import tqdm

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser

parser = get_parser()
opt, unknown = parser.parse_known_args()
unknown = [s.lstrip('-') for s in unknown]
configs = [OmegaConf.load(cfg) for cfg in opt.base]
cli = OmegaConf.from_dotlist(unknown)
print('###### cli input training setup:  ######\n',cli)
config = OmegaConf.merge(*configs, cli)

if config.resolution % (16 * config.cond_scale_factor) != 0:
    raise ValueError(
        f"Image resolution {config.resolution} must be divisible by {16 * config.cond_scale_factor} "
        f"(16 * cond_scale_factor) to ensure proper feature map alignment in the model. "
        f"Please adjust either the resolution or cond_scale_factor."
    )


train_dataset = instantiate_from_config(config.data.train)
