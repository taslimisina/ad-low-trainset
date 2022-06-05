import argparse
from architectures import resnets
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
args = parser.parse_args()

def debug(*argsss):
    if args.debug:
        print(argsss)

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
        criterion = nn.KLDivLoss(reduction='batchmean')

        for data, _ in normal_dataloader:
            data = data.to(device)
            teacher_outs = teacher(data)
            student_outs = student(data)
            debug("teacher_outs.shape:", teacher_outs.shape)
            debug("student_outs.shape:", student_outs.shape)
            loss = criterion(F.log_softmax(student_outs / args.temperature, dim=1),
                        F.softmax(teacher_outs / args.temperature, dim=1))
            debug("loss.shape:", loss.shape)
            for l in loss:
                debug("l.shape:", l.shape)
                losses.append(l.item())
                targets.append(0)

        for data, _ in anomaly_dataloader:
            data = data.to(device)
            teacher_outs = teacher(data)
            student_outs = student(data)
            loss = criterion(F.log_softmax(student_outs / args.temperature, dim=1),
                             F.softmax(teacher_outs / args.temperature, dim=1))
            for l in loss:
                losses.append(l.item())      # is it okay for >1 test batch size?
                targets.append(1)

        debug("targets:", targets)
        debug("Losses:", losses)
        auc = roc_auc_score(targets, losses)
        print("AUROC:", auc)

        student.train()


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
        drop_last=True
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
    normal_dataloader = DataLoader(normal_subset, shuffle=False, batch_size=args.test_bs)
    anomaly_indices = anomaly_mask.nonzero().reshape(-1)
    anomaly_subset = Subset(cifar10_trainset, anomaly_indices)
    anomaly_dataloader = DataLoader(anomaly_subset, shuffle=False, batch_size=args.test_bs)

    teacher.eval()  # okay?

    for i in range(args.epochs):
        l = 0
        for data, _ in train_loader:
            data = data.to(device)
            with torch.no_grad():
                teacher_outs = teacher(data)
            student_outs = student(data)
            loss = kd_loss_fn(teacher_outs, student_outs)
            if i == 0:
                debug("loss:", loss.shape, loss)

            l += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch:", i, "\tLoss:", l)    # Todo loss is not accurate (KL loss -> reduction(mean))

        if i % args.eval == 0:
            test(teacher, student, normal_dataloader, anomaly_dataloader)



def main(args):
    teacher = resnets.wide_resnet50_2(pretrained=True)
    teacher.to(device)
    student = resnets.wide_resnet50_2(pretrained=False)
    student.to(device)
    train(teacher, student)


if __name__ == '__main__':
    main(args)