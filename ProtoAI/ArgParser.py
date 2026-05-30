# config.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Artificial Intelligence Model Rapid Prototyping Framework Default Argument Parser")
    
    # Define your arguments
    parser.add_argument("-m", "--model_config", dest="ModelConfig", type=str, default="config/Model/NetworkConfiguration.json")
    parser.add_argument("-t", "--training_data_dir", dest="TrainDir", type=str, default="Datasets/Training", help="Path to training dataset")
    parser.add_argument("-v", "--validation_data_dir", dest="ValidationDir", type=str, default="/mnt/c/Users/gengr/Documents/deep-learning/Datasets/Testing", help="Path to validation dataset")
    parser.add_argument("-b", "--batch_size", dest="BatchSize", type=int, default=32, help="Input batch size for training")
    parser.add_argument("-e", "--epochs", dest="Epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("-l", "--lr", dest="LearningRate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-w","--num_workers", dest="NumWorkers", type=int, default=2, help="Number of workers for DataLoaders")
    parser.add_argument("-x", "--image_dim_x", dest="ImageDimX", type=int, default=227, help="X-dimension to resize images")
    parser.add_argument("-y", "--image_dim_y", dest="ImageDimY", type=int, default=235, help="Y-dimension to resize images")
    parser.add_argument("-d", "--device", dest="Device", type=str, default="cuda:0", help="Device to train network on")
    # Parse and return the clean object
    return parser.parse_args()