import os
import torch
from model import iresnet
from tqdm import tqdm
from torch.optim import Adam
import dataloader
import argparse
import sys
from tools import utils
import torch.nn as nn

parser = argparse.ArgumentParser(description='Trainer for CattleFace')
parser.add_argument('--img_list', default='', type=str,
                    help='训练图片列表')
parser.add_argument('--val_list', default='', type=str,
                    help='验证图片列表')
parser.add_argument('--save_path', default='weights', type=str,
                    help='保存路径')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='训练总轮次')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='训练启动轮次')
parser.add_argument('--device', default=1, type=int, metavar='N',
                    help='GPU')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--input-fc-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--last-fc-size', default=, type=int,
                    help='类别数量')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量的权值')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay',
                    dest='weight_decay')
parser.add_argument('--lr-drop-epoch', default=[30, 60], type=int, nargs='+',
                    help='The learning rate drop epoch')
parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
                    help='The learning rate drop ratio')

args = parser.parse_args()


classifier = iresnet.SoftmaxBuilder(args)


classifier = classifier.cuda(args.device)

opt_c = Adam(classifier.parameters(), args.lr, betas=(0.5, 0.999))

arcloss = iresnet.FocalLoss(gamma=2)

def train():
    
    train_loader, len_train = dataloader.img_loader(args)

    best_acc = 0
   
    for epoch in range(args.start_epoch, args.epochs):
        
        loss_c = utils.AverageMeter('Loss', ':.3f')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        
        classifier.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for i, (data, ground_face, target, img_path) in enumerate(train_bar):
            data = data.cuda(args.device)
            target = target.cuda(args.device)
            
            output, cos = classifier(data, target)
            
            acc1, acc5 = utils.accuracy(args, cos, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))

            loss = arcloss(output, target) 
            loss_c.update(loss.item(), data.size(0))

           
            opt_c.zero_grad()
            loss.backward()
            opt_c.step()
            
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc1:{:.3f}".format(
                epoch + 1, args.epochs, loss_c.avg, top1.avg
                )
        
        classifier.eval()
        val_loader, len_val = dataloader.val_loader(args)
        top1_val = utils.AverageMeter('Acc@1', ':6.2f')
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_img, val_ground, val_labels, path = val_data
                val_img = val_img.cuda(args.device)
                val_labels = val_labels.cuda(args.device)
                output, cos_val = classifier(val_img, val_labels)
                acc_val = utils.accuracy(args, cos_val, val_labels, topk=(1, ))
                top1_val.update(acc_val[0], val_img.size(0))
        
        val_acc = top1_val.avg
        
        print("val acc:{}".format(val_acc))
        if val_acc >= best_acc:
            best_acc = val_acc
            print("保存第{}轮的参数,val_acc={}.".format(epoch + 1, best_acc))
           
            encorder_save_path = "encoder"+".pth"
            encorder_save_path = os.path.join(args.save_path, encorder_save_path)
            torch.save(classifier.encoder.state_dict(), encorder_save_path)

            feature_save_path = "feature"+".pth"
            feature_save_path = os.path.join(args.save_path,feature_save_path)
            torch.save(classifier.feature.state_dict(), feature_save_path)

            fc_save_path = "fc"+".pth"
            fc_save_path = os.path.join(args.save_path,fc_save_path)
            torch.save(classifier.fc.state_dict(), fc_save_path)
            
        
        

           

    print('training complete')


if __name__ == '__main__':
    train()