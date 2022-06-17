import argparse
from architectures import resnets
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import logging
from loss_functions import KldLoss, MseDirectionLoss

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility
seed = 10
torch.manual_seed(seed)
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
# torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    np.random.seed(seed)
    random.seed(seed)
generator = torch.Generator()
generator.manual_seed(seed)

parser = argparse.ArgumentParser(description="Knowledge Distillation From a Single Image For Anomaly Detection.")
parser.add_argument("--images_dir", type=str, required=True, help="path to one-image dataset")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--test_bs", type=int, default=512, help="Test Batch size")
parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument("--eval", type=int, default=1, help="Evaluate after this number of epochs")
parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--temperature", type=float, default=8, help="temperature logits are divided by")
parser.add_argument("--normal_classes", required=True, nargs="+", type=int, help="Normal classes in cifar10")
parser.add_argument("--debug", action='store_true')
parser.add_argument("--student_layers", default=[3, 4, 6, 3], nargs=4, type=int,
                    help="Number of blocks in each layer in student wide-resnet.")
parser.add_argument("--withfc", action='store_true',
                    help="Get ouputs of teacher and student after fully-connected layer or before.")
parser.add_argument("--log", type=str, default="log.txt", help="location of log file")
parser.add_argument("--lossfn", type=str, choices=["kl", "mse"],
                    default="kl", help="Loss function - KL divergance or MSE")
parser.add_argument("--lamda", type=float, default=0.01, help="coefficient for MSE loss")
args = parser.parse_args()

logging.basicConfig(filename=args.log, filemode='w', level=logging.DEBUG if args.debug else logging.INFO)
logging.info(args)

def kd_loss_fn(teacher_outs, student_outs):
    loss = nn.KLDivLoss(reduction="batchmean")
    kd_loss = loss(F.log_softmax(student_outs / args.temperature, dim=1),
                        F.softmax(teacher_outs / args.temperature, dim=1))
    return kd_loss


def test(teacher, student, normal_dataloader, anomaly_dataloader):
    with torch.no_grad():
        student.eval()

        targets = []
        losses = []
        targets_1v1 = [[] for i in range(10)]
        losses_1v1 = [[] for i in range(10)]
        if args.lossfn == "kl":
            criterion = KldLoss(reduction="none", temperature=args.temperature)
        elif args.lossfn == "mse":
            criterion = MseDirectionLoss(args.lamda, reduction="none")
        else:
            logging.ERROR("Loss function not defined!")

        for data, _ in normal_dataloader:
            data = data.to(device)
            teacher_outs = teacher(data)
            student_outs = student(data)
            loss = criterion(student_outs, teacher_outs)
            for l in loss:
                losses.append(l.item())
                targets.append(0)
                for i in range(10):
                    losses_1v1[i].append(l.item())
                    targets_1v1[i].append(0)

        for data, cls in anomaly_dataloader:
            data = data.to(device)
            teacher_outs = teacher(data)
            student_outs = student(data)
            loss = criterion(F.log_softmax(student_outs / args.temperature, dim=1),
                             F.softmax(teacher_outs / args.temperature, dim=1))
            loss = loss.sum(dim=1)
            for i, l in enumerate(loss):
                losses.append(l.item())
                targets.append(1)
                losses_1v1[cls[i]].append(l.item())
                targets_1v1[cls[i]].append(1)

        for i in range(10):
            try:
                auc = roc_auc_score(targets_1v1[i], losses_1v1[i])
                logging.info("AUROC vs class " + str(i) + ":\t" + str(auc))
            except:
                logging.info("AUROC vs class " + str(i) + ":\t--------")
        auc = roc_auc_score(targets, losses)
        logging.info("AUROC: " + str(auc))


def train(teacher, student):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trnsfrms = [
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        args.images_dir,
        transforms.Compose(trnsfrms))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    optimizer = AdamW(student.parameters(), lr=args.lr,
                      weight_decay=args.wd)

    # test dataloaders
    cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose(trnsfrms))
    normal_mask = (torch.tensor(cifar10_trainset.targets) == args.normal_classes[0])
    for i in range(1, len(args.normal_classes)):
        normal_mask = normal_mask | (torch.tensor(cifar10_trainset.targets) == args.normal_classes[i])
    anomaly_mask = ~normal_mask
    normal_indices = normal_mask.nonzero().reshape(-1)
    normal_subset = Subset(cifar10_trainset, normal_indices)
    normal_dataloader = DataLoader(
        normal_subset,
        shuffle=False,
        batch_size=args.test_bs,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )
    anomaly_indices = anomaly_mask.nonzero().reshape(-1)
    anomaly_subset = Subset(cifar10_trainset, anomaly_indices)
    anomaly_dataloader = DataLoader(
        anomaly_subset,
        shuffle=False,
        batch_size=args.test_bs,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    teacher.eval()

    if args.lossfn == "kl":
        criterion = KldLoss(reduction="batchmean", temperature=args.temperature)
    elif args.lossfn == "mse":
        criterion = MseDirectionLoss(args.lamda)
    else:
        logging.ERROR("Loss function not defined!")

    for i in range(args.epochs):
        student.train()
        l = 0
        for data, _ in train_loader:
            data = data.to(device)
            with torch.no_grad():
                teacher_outs = teacher(data)
            student_outs = student(data)
            loss = criterion(teacher_outs, student_outs)

            l += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info("Epoch: " + str(i) + "\tLoss:" + str(l))    # Todo loss is not accurate (KL loss -> reduction(mean))

        if i % args.eval == 0:
            test(teacher, student, normal_dataloader, anomaly_dataloader)



def main(args):
    teacher = resnets.wide_resnet50_2(pretrained=True, withfc=args.withfc)
    teacher.to(device)
    student = resnets.custom_wrn(layers=args.student_layers, withfc=args.withfc)
    student.to(device)
    train(teacher, student)


if __name__ == '__main__':
    main(args)