## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
#from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
warnings.filterwarnings('ignore')
from itertools import product
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.nn.functional as F

mlflow.set_experiment("Custom LR:0.0001, Epochs:50, BS:16, WD:0.01, AdamW, CosineAnnealingLR")
mlflow.start_run()

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



if __name__ == "__main__":
#    experiment_id = mlflow.create_experiment(f"")    
    

    ## Writing the loss and results
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    log = Logger()
    log.open("logs/%s_log_train.txt")
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    log.write('                           |----- Train -----|----- Valid----|---------|\n')
    log.write('mode     iter     epoch    |       loss      |        mAP    | time    |\n')
    log.write('-------------------------------------------------------------------------------------------\n')


    ######################## load file and get splits #############################
    train_imlist = pd.read_csv("train.csv")
    train_gen = knifeDataset(train_imlist,mode="train")
    train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=8,persistent_workers=True)
    val_imlist = pd.read_csv("test.csv")
    val_gen = knifeDataset(val_imlist,mode="val")
    val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=8,persistent_workers=True)

    ## Loading the model to run
    #model = timm.create_model('tf_efficientnet_b3', pretrained=True,num_classes=config.n_classes) ###efficientnet_b1
    #model = timm.create_model('resnet50', pretrained=True, num_classes=config.n_classes)

    class CustomModel(nn.Module):
        def __init__(self, num_classes):
            super(CustomModel, self).__init__()

            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

            # Max pooling layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            # Fully connected layers
            self.fc1 = nn.Linear(128 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, num_classes)

            # Dropout for regularization (optional)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            # Convolutional layers with activation and pooling
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)

            # Flatten the output for the fully connected layers
            x = x.view(-1, 128 * 28 * 28)

            # Fully connected layers with dropout (optional)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x
        
    num_classes = 192
    model = CustomModel(num_classes)

    # Remove the classification layer
    #model.classifier = nn.Identity()

    #Add layer
    # Get the features before the classifier head
    #num_features = model.classifier.in_features

    # Extract the penultimate layer
    #penultimate = model.conv_head  # Assuming the penultimate layer is the last layer in the 'conv_head'

    # Define the additional layer to be added
    #new_layer = torch.nn.Conv2d(in_channels=penultimate.out_channels, out_channels=1536, kernel_size=3, stride=1, padding=1)

    # Append the new layer after the penultimate layer
    #model.conv_head = torch.nn.Sequential(penultimate, new_layer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ############################# Parameters #################################
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay = 0.01) #weight_decay =
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
    criterion = nn.CrossEntropyLoss().cuda()

    mlflow.log_param("learning_rate", config.learning_rate)
    mlflow.log_param("batch_size", config.batch_size)
    mlflow.log_param("weight_decay", 0.01) ###weight_decay
    mlflow.log_param("optimizer", "AdamW") ###Optimizer
    mlflow.log_param("scheduler", "CosineAnnealingLR") ###Scheduler
    mlflow.log_param("epoch", 50) ###Epoch

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

        if epoch == 49 or epoch == 9 or epoch == 19 or epoch == 29 or epoch == 39: ###Epoch
            mlflow.log_metric(f"train_loss", train_metrics[0])
            mlflow.log_metric(f"val_mAP", val_metrics[0])

        ## Saving the model
        filename = "Knife-Custom-E" + str(epoch + 1)+  ".pt"  ###NAME
        #filename = "Knife-Res50-E" + str(epoch + 1)+  ".pt"
        torch.save(model.state_dict(), filename)
        
mlflow.end_run()  
