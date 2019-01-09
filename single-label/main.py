import os
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import argparse
from dataset import create_dataset
from models import create_model
from train import train_model
import ipdb

parser = argparse.ArgumentParser("train model")
parser.add_argument("--csv_file", type=str, default='csv', help="input file")
parser.add_argument("--img_size", type=int, default=224, help="input image size")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument('--optimizer', type=str, default='Adam', help='Chosed optimizer')
parser.add_argument("--model", type=str, default='resnet18', help="Train model")
parser.add_argument('--epochs', type=int, default=24, help='number of epochs to train (default: 25)')
parser.add_argument('--pretrained', type=str, default='False',
                    choices=['True', 'False'], 
                    help='If True, only train last layer of model')
# parser.add_argument("--log", type=str, default='', help="Training log file")
# parser.add_argument("--saved_model_file", type=str, default='', help="Saved trainde models")
# parser.add_argument("--checkpoint", type=str, default='', help="Load chenkpoint")
args = parser.parse_args()


csv_file = args.csv_file  + '/{}.csv'
root_dir = './'
out = create_dataset(csv_file=csv_file,
                     root_dir=root_dir,
                     img_size=args.img_size,
                     batch_size=args.batch_size)
dataloaders = out['dataloaders']
dataset_sizes = out['dataset_sizes']

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# use_gpu = torch.cuda.is_available()
model_conv = create_model(model_key=args.model,
                          pretrained=eval(args.pretrained),
                          num_of_classes=75,
                          device=device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
if eval(args.pretrained):
    parameters_totrain = model_conv.fc.parameters()
else:
    parameters_totrain = model_conv.parameters()

# Choose optimizer for training
if args.optimizer == 'SGD':
    optimizer_conv = optim.SGD(parameters_totrain, lr=0.001, momentum=0.9)
elif args.optimizer == 'Adam':
    optimizer_conv = optim.Adam(parameters_totrain, lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

saved_model_file = os.path.join('saved_model_file', args.csv_file, args.model)
if not os.path.exists(saved_model_file):
    os.makedirs(saved_model_file)
saved_model = saved_model_file + '/' + args.model + '_' + args.optimizer + '.pkl'

log = os.path.join('log', args.csv_file, args.model)
if not os.path.exists(log):
    os.makedirs(log)
saved_log = log + '/' + args.model + '_' + args.optimizer 

writer = SummaryWriter(saved_log)

train_model(model_conv,
            criterion,
            optimizer_conv,
            exp_lr_scheduler,
            dataloaders,
            dataset_sizes,
            writer,
            num_epochs=args.epochs,
            saved_model=saved_model,
            device=device)
writer.close()
