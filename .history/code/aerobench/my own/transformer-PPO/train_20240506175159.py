import torch
from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser
import sys
import numpy as np
import random
def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/F_16env_trxl.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser(options["--config"]).get_config()

    # Determine the device to be used for training and set the default tensor type
    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config,filename, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()