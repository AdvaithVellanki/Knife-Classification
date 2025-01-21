## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
from torchvision import models
from torch.optim import AdamW
import torch.nn.functional as F
warnings.filterwarnings('ignore')

## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()
log.open("logs/resnext50_25.txt")
log.write("\nModel Architecture: ResNeXt")
log.write("\nOptimiser: Adam")
log.write("\nLearning Rate: 0.00005")
log.write("\nBatch Size: 8")
log.write("\nNumber of Epochs: 25")
log.write("\nLoss Function: Cross Entropy Loss")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |----- Train -----|----- Valid----|---------|\n')
log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
log.write('-------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader,model,criterion,optimizer,epoch,valid_accuracy,start):
    losses = AverageMeter()
    model.train()
    model.training=True
    for i,(images,target,fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(),images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        optimizer.zero_grad()
        scheduler.step()

        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s' % (\
                "train", i, epoch,losses.avg,valid_accuracy[0],time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]

# Validating the model
def evaluate(val_loader,model,criterion,epoch,train_loss,start):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], map.avg,time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
    return [map.avg]
  

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5


## Loss Function code
#class LabelSmoothingCrossEntropy(nn.Module):
 #   def __init__(self, smoothing=0.1):
  #      super(LabelSmoothingCrossEntropy, self).__init__()
   #     self.smoothing = smoothing

    #def forward(self, input, target):
     #   log_prob = F.log_softmax(input, dim=-1)
      #  weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
       # weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        #loss = (-weight * log_prob).sum(dim=-1).mean()
        #return loss

## Focal Loss
#class FocalLoss(nn.Module):
 #   def __init__(self, gamma=2, reduction='mean'):
  #      super(FocalLoss, self).__init__()
   #     self.gamma = gamma
    #    self.reduction = reduction

    #def forward(self, input, target):
     #   cross_entropy = F.cross_entropy(input, target, reduction='none')
      #  pt = torch.exp(-cross_entropy)
       # focal_loss = (1 - pt) ** self.gamma * cross_entropy
        #if self.reduction == 'mean':
         #   return focal_loss.mean()
         #elif self.reduction == 'sum':
          #  return focal_loss.sum()
        #else:
         #   return focal_loss
######################## load file and get splits #############################
train_imlist = pd.read_csv("train.csv")
train_gen = knifeDataset(train_imlist,mode="train")
train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=8)
val_imlist = pd.read_csv("test.csv")
val_gen = knifeDataset(val_imlist,mode="val")
val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=8)

## Loading the model to run
#model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=config.n_classes)
model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=config.n_classes)
#model=models.resnet18(pretrained=True)#Using resnet
#model.fc = torch.nn.Linear(model.fc.in_features, config.n_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Parameters #################################
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=1e-4)     
#optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay= 1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
#scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
criterion = nn.CrossEntropyLoss().cuda()
#criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()
#criterion = FocalLoss(gamma=2, reduction='mean').cuda()
############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()
#train
for epoch in range(0,config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,start)
    val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,start)
    ## Saving the model
    filename = "Knife-ResNeXt50-E" + str(epoch + 1)+  ".pt"
    torch.save(model.state_dict(), filename)
    

   
