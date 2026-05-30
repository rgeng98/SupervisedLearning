import ProtoTorch
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[
        # 1. Write to your log file (replaces filename and filemode='w')
        logging.FileHandler(
            filename=f'{Path(__file__).parent}/logs/{Path(__file__).stem}.log', 
            mode='w'
        ),
        # 2. Print to the terminal console
        logging.StreamHandler(sys.stdout)
    ]
)


if __name__ == "__main__":
    args = ProtoTorch.ArgParser.parse_args()

    model = ProtoTorch.DeepLearningWrapper.Model(args)

    for _ in range(args.Epochs):
        model.training_epoch()
        model.validation_epoch()

    model.display_model_performance()