import os
import torch
from torch import optim
from torchvision import transforms, models
import logging
import numpy as np
import pandas as pd
import copy
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from thop import profile
from pthflops import count_ops

from data import MyDataset, BalancedBatchSampler
from model import *
from model.convnext import *
from model.tresnet.tresnet import *
from losses import *
from eval import eval_model
from utils import AverageMeter, accuracy, Mixup


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): torch.cuda.set_device(device)
batch_size = 64
epochs = 50
learning_rate = 0.01
warmup_epoch = 0
warmup_lr = 1e-4
num_classes = 48
data_dir = 'D:/OneDrive - Universiti Malaya/PhD/data/mohsin_vmmr'
use_bbox_train = False # for MyDataset class
use_bbox_val = False # for MyDataset class
use_model = True # for MyDataset class
use_make = False # for MyDataset class
use_type = False # for MyDataset class
use_viewpoint = False # for MyDataset class
model_no = 9
use_my_model = False
aux_logits = False # For GoogLeNet only
use_mixup = False
lr_scheduler_epoch_update = True # Set True to update after one epoch, Set False to update after one batch
use_cls_weight_for_loss = False
use_batch_sampler = False # check label
n_img_per_label = 8 # For batchsampler
use_dp = False # For data parallel ops, change gpu id in code below
dp_gpu_id = [0, 1, 2, 3]
use_amp = False


def train_one_epoch(
    dataloaders_train, model_no, model, optimizer_, lr_scheduler_, 
    lr_scheduler_epoch_update):
    
    model.train()
    criterion.train()
    train_epoch_loss = AverageMeter()
    train_epoch_accuracy = AverageMeter()

    for _, inputs, labels in dataloaders_train:
        inputs = inputs.to(device)
        labels = labels[0].to(device)

        labels_mixup = None
        if use_mixup:
            inputs, labels_mixup = mixup_fn(inputs, labels)

        # with torch.no_grad():
        #     viewpoint_emb = model_viewpoint.conv1(inputs)
        #     viewpoint_emb = model_viewpoint.bn1(viewpoint_emb)
        #     viewpoint_emb = model_viewpoint.relu(viewpoint_emb)
        #     viewpoint_emb = model_viewpoint.maxpool(viewpoint_emb)
        #     viewpoint_emb = model_viewpoint.layer1(viewpoint_emb)
        #     viewpoint_emb = model_viewpoint.layer2(viewpoint_emb)
        #     viewpoint_emb = model_viewpoint.layer3(viewpoint_emb)
        #     viewpoint_emb = model_viewpoint.layer4(viewpoint_emb)
        #     viewpoint_emb = model_viewpoint.avgpool(viewpoint_emb).flatten(1)
        #     viewpoint_logits = model_viewpoint(inputs)

        optimizer_.zero_grad()

        with torch.set_grad_enabled(True):
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler
        # with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            if model_no != 2:
                # x3_logits, x4_logits, x5_logits, xcat_logits = model(inputs)
                # train_batch_loss = criterion(x3_logits, labels) + criterion(x4_logits, labels) + criterion(x5_logits, labels) + criterion(xcat_logits, labels)
                # logits = x3_logits + x4_logits + x5_logits + xcat_logits

                # logits = model(inputs)
                features, logits = model(inputs)
                train_batch_loss = criterion(logits, labels_mixup if use_mixup else labels)
                if alt_criterion is not None:
                    train_batch_loss += 0.005*alt_criterion(features, labels, 7, num_classes)

                    # Update Feature Center
                    # feature_center_batch = F.normalize(feature_center[labels], dim=-1)
                    # feature_center[labels] += 5e-2 * (features.detach() - feature_center_batch)
                    # train_batch_loss += 0.1*alt_criterion(features, feature_center_batch)

                # logits, logits3, logits4 = model(inputs)
                # train_batch_loss = (criterion(logits, labels) + criterion(logits3, labels) + criterion(logits4, labels)) / 3
                # logits = (logits + logits3 + logits4) / 3

                # Grad-cam
                # logits_raw = model(inputs, 0)
                # train_batch_loss_raw = criterion(logits_raw, labels)
                # cam = GradCAMPlusPlus(model, [model.module.layer4[-1] if use_dp else model.layer4[-1]], use_cuda=True)
                # targets = [ClassifierOutputTarget(i) for i in labels]
                # grayscale_cam = torch.from_numpy(cam(input_tensor=inputs, targets=targets)).to(inputs.device)
                # if torch.isnan(grayscale_cam).any():
                #     grayscale_cam = torch.ones_like(grayscale_cam)
                # inputs_crop = gradcam_crop(inputs, grayscale_cam)
                # model.train()
                # logits_crop = model(inputs_crop, 1)
                # train_batch_loss = 0.5*criterion(logits_crop, labels) + 0.5*train_batch_loss_raw
                # logits = (logits_raw + logits_crop) / 2

                # WS-DAN
                # attention_map, logits = model(inputs, 0)
                # with torch.no_grad():
                #     crop_images = batch_augment(inputs, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
                #     drop_images = batch_augment(inputs, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
                # inputs = torch.cat([crop_images, drop_images], 0)
                # logits_crop, logits_drop = torch.chunk(model(inputs, 1), 2, dim=0)
                # train_batch_loss = (criterion(logits, labels) + criterion(logits_crop, labels) + criterion(logits_drop, labels))/3
                # logits = (logits + logits_crop + logits_drop) / 3
                    
                acc1 = accuracy(logits, labels.data)[0]

                # acc1 = accuracy((logits + logits_lrau)/2, labels.data)[0]
                
            elif model_no == 2:
                logits = model(inputs)
                train_batch_loss = criterion(logits, labels_mixup if use_mixup else labels)
                if alt_criterion is not None:
                    train_batch_loss += 0.005*alt_criterion(features, labels, 7, num_classes)
                acc1 = accuracy(logits, labels.data)[0]

            if use_amp:
                scaler.scale(train_batch_loss).backward()
                scaler.step(optimizer_)
                scaler.update()
            else:
                train_batch_loss.backward()
                optimizer_.step()
                # criterion.next_epoch()
                # criterion.update()

        if not lr_scheduler_epoch_update:
            lr_scheduler_.step()

        train_epoch_loss.update(train_batch_loss, inputs.size(0))
        train_epoch_accuracy.update(acc1.item(), inputs.size(0))

    return train_epoch_loss.avg, train_epoch_accuracy.avg

    
def train_model(
    dataloaders, model, optimizer_, lr_scheduler_, 
    epochs=25, model_no=0, lr_scheduler_epoch_update=True, warmup=False):

    best_state_dict = copy.deepcopy(model.state_dict())
    best_accuracy = 0
    train_loss, train_accuracy, val_loss, val_accuracy, lr = [], [], [], [], []

    for epoch in range(epochs):

        lr.append(lr_scheduler_.get_last_lr()[0])       
        
        # Train
        train_epoch_loss, train_epoch_accuracy = train_one_epoch(
            dataloaders['train'], model_no, model, optimizer_,
            lr_scheduler_, lr_scheduler_epoch_update)
        if lr_scheduler_epoch_update:
            lr_scheduler_.step()
        train_loss.append(train_epoch_loss.item())
        train_accuracy.append(train_epoch_accuracy)
        logging.info(
            f"Epoch {epoch:3d}/{epochs-1:3d} {'Train':5s}, "
            f"Loss: {train_epoch_loss:.4f}, "
            f"Acc: {train_epoch_accuracy:.4f}")
        if warmup:
            # If warmp, do not proceed with the rest
            continue

        # Validate
        val_epoch_loss, val_epoch_accuracy = torch.tensor(0), 0
        if epoch > 19:
            val_epoch_loss, val_epoch_accuracy = eval_model(
                dataloaders['val'], model_no, model,
                nn.CrossEntropyLoss() if use_mixup else criterion, device,
                model_viewpoint=model_viewpoint, alt_criterion=alt_criterion,
                )
        val_loss.append(val_epoch_loss.item())
        val_accuracy.append(val_epoch_accuracy)
        logging.info(
            f"Epoch {epoch:3d}/{epochs-1:3d} {'Val':5s}, "
            f"Loss: {val_epoch_loss:.4f}, "
            f"Acc: {val_epoch_accuracy:.4f}")
        
        # Check if val_epoch acc > best_acc
        if val_epoch_accuracy > best_accuracy:
            best_accuracy = val_epoch_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())

    if not warmup:
        # Load best model
        model.load_state_dict(best_state_dict)
        
        # Evalaute best model
        logging.info('Best Val Acc: {:4f}'.format(best_accuracy))
        val_epoch_loss, val_epoch_accuracy = eval_model(
            dataloaders['val'], model_no, model,
            nn.CrossEntropyLoss() if use_mixup else criterion, device,
            model_viewpoint=model_viewpoint, alt_criterion=alt_criterion,
            print_cls_report=True,
            )
        
        # Save best model
        model_name = '{}_model{}_{:.4f}'.format(
            os.path.split(data_dir)[1], model_no, best_accuracy)
        if use_dp:
            best_state_dict = copy.deepcopy(model.module.state_dict())    
        else:
            best_state_dict = copy.deepcopy(model.state_dict())
        if os.path.exists(f'{model_name}.pth'):
            model_name = model_name + '1'
        torch.save(best_state_dict, f'{model_name}.pth')
        # shutil.copy('model/model.py', f'{model_name}.py')

        # Save training details
        pd.DataFrame({
            'Epochs': range(epochs), 'Learning Rate': lr, 'Training Loss': train_loss, 
            'Training Accuracy': train_accuracy, 'Validation Loss': val_loss, 
            'Validation Accuracy':val_accuracy}).to_csv(f'{model_name}.csv', index=False)

    return model


if __name__ == '__main__':
    
    logging.info(f"Running model {model_no}")
    
    # Refer CMSEA: Compound Model Scaling With Efficient Attention for Fine-Grained Image Classification
    # Refer ConvNeXt
    # for data aug
    data_transforms = {
        'train': transforms.Compose([
            # CompCars
            transforms.RandomResizedCrop(
                size=224, scale=(0.2, 1),  # 0.2 
            #     # ratio=(0.2, 1.5)
                ),
            transforms.RandomHorizontalFlip(p=0.5), # 0.5
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),

            # Cars
            # transforms.RandomResizedCrop(size=448, scale=(0.8, 1)), # 0.2
            # transforms.Resize((448, 448)), # 224, 448, try 512 cross entropy
            # transforms.RandomCrop(size=224, pad_if_needed=False),
            # transforms.RandomCrop(448, padding=4),
            # transforms.RandomHorizontalFlip(p=0.5), # 0.2, 0.5
            # transforms.AutoAugment(),
            # transforms.RandAugment(),
            # transforms.TrivialAugmentWide(),
            # transforms.AugMix(),
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
               
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.RandomErasing(0.2)
        ]),
        'val': transforms.Compose([
            # CompCars
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            
            # Cars
            # transforms.Resize((448, 448)),
            # transforms.CenterCrop((448, 448)),
          
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {x: MyDataset(
        root=data_dir, is_train=True if x == 'train' else False, 
        use_bbox=use_bbox_train if x == 'train' else use_bbox_val,
        use_model=use_model, use_make=use_make, use_type=use_type,
        use_viewpoint=use_viewpoint, transform=data_transforms[x], 
        # combine_vtr_label=True,
        # exp=3
        # use_top_n_model=[],
        # use_n_img_per_cls=250,
        # small_dataset=True,
        # remove_poor_images=True,
        num_classes=num_classes
        ) for x in ['train', 'val']}
    if use_batch_sampler:
        sampler = {'train': BalancedBatchSampler(
            image_datasets['train'], label=image_datasets['train'].label_model, 
            batch_size=batch_size, n_samples_per_cls=n_img_per_label
            )}
        dataloaders = {
            'train': torch.utils.data.DataLoader(
                image_datasets['train'], batch_sampler=sampler['train'], 
                num_workers=4, pin_memory=True
                ),
            'val': torch.utils.data.DataLoader(
                image_datasets['val'], batch_size=batch_size, shuffle=True, 
                num_workers=4, drop_last=False
                )
            }
    else:
        dataloaders = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, 
            drop_last=False
            ) # False, True if x == 'train' else False
            for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    logging.info(f"Training on {dataset_sizes['train']} images")
    logging.info(f"Validating on {dataset_sizes['val']} images")

    if model_no == 0:
        # alexnet
        if use_my_model:
            mymodel = alexnet(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.alexnet(pretrained=True, num_classes=1000)
            num_ftrs = mymodel.classifier[-1].in_features
            mymodel.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    elif model_no == 1:
        # vgg16
        if use_my_model:
            mymodel = vgg16(pretrained=True, num_classes=num_classes, use_my_model=True)
        else:
            mymodel = vgg16(pretrained=True, num_classes=1000, use_my_model=False)
            num_ftrs = mymodel.classifier[-1].in_features
            mymodel.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            
    elif model_no == 2:
        # Inceptionv3
        if use_my_model:         
            mymodel = inception_v3(
                pretrained=True, num_classes=num_classes,
                aux_logits=aux_logits)
        else:            
            mymodel = models.inception_v3(
                pretrained=True, num_classes=1000, 
                aux_logits=True)
            num_ftrs = mymodel.fc.in_features
            mymodel.fc = nn.Linear(num_ftrs, num_classes)    
            if aux_logits:
                num_ftrs = mymodel.AuxLogits.fc.in_features
                mymodel.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            else:
                mymodel.aux_logits = False
                mymodel.AuxLogits = None

    elif model_no == 3:
        # resnet50
        if use_my_model:
            mymodel = resnet50(pretrained=True, num_classes=num_classes)
            # mymodel = resnet101(pretrained=True, num_classes=num_classes)
            # mymodel = legacy_seresnet50(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.resnet50(pretrained=True, num_classes=1000)
            num_ftrs = mymodel.fc.in_features
            mymodel.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_no == 4:
        # densenet161, densenet169
        if use_my_model:
            mymodel = densenet169(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.densenet169(pretrained=True, num_classes=1000)
            num_ftrs = mymodel.classifier.in_features
            mymodel.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_no == 5:
        raise NotImplementedError("Please use other script for ViT")
    elif model_no == 6:
        raise NotImplementedError("Please use other script for TransFG")

    elif model_no == 7:
        # MobileNet
        if use_my_model:
            mymodel = mobilenetv3_large(pretrained=True, num_classes=num_classes, use_my_model=True)
        else:
            # mymodel = models.mobilenet_v3_small(pretrained=True, num_classes=1000)
            # num_ftrs = mymodel.classifier[-1].in_features
            # mymodel.classifier[-1] = nn.Linear(num_ftrs, num_classes)

            mymodel = mobilenetv3_large(pretrained=True, num_classes=num_classes, use_my_model=False)

    elif model_no == 8:
        # Convnext
        if use_my_model:
            mymodel = convnext_tiny(pretrained=True, in_22k=True, num_classes=num_classes, use_my_model=True)
        else:
            mymodel = convnext_tiny(pretrained=True, in_22k=True, num_classes=num_classes, use_my_model=False)

    elif model_no == 9:
        # TResNet
        if use_my_model:
            mymodel = TResnetM(
                pretrained=True, num_classes=num_classes, remove_aa_jit=False,
                use_my_model=True, use_ml_decoder=False,
                num_of_groups=80, decoder_embedding=768, zsl=0
            )
        else:
            mymodel = TResnetM(
                pretrained=True, num_classes=num_classes, remove_aa_jit=False,
                use_my_model=False, use_ml_decoder=True,
                num_of_groups=80, decoder_embedding=768, zsl=0
                # num_of_group=100 for stanford car
            )

    # logging.info('Load my pretrained weight')
    # state_dict = torch.load('CompCarsWeb_model3_TrainRandomresizecropHflipRandomerase224_TestResizeCentercrop224_0.9800_TestResize448_0.9807_epoch90.pth', map_location='cpu')
    # mymodel.load_state_dict(state_dict)
    mymodel = mymodel.to(device)

    # FixRes
    # for name, params in mymodel.named_parameters():
    #     if 'ms4.combine.1' not in name:
    #         params.requires_grad = False

    # Calculate params and flops
    # input_dummy = torch.rand((1, 3, 448, 448)).to(device)
    # # flops = count_ops(mymodel, input)
    # flops, params = profile(mymodel, inputs=(input_dummy, ))
    # logging.info(f'FLOPs (B): {flops/1e9}, Model Parameters (M): {params/1e6}')

    if use_dp:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=dp_gpu_id)

    # load model for viewpoint
    model_viewpoint = None
    # model_viewpoint = models.resnet18(pretrained=False, num_classes=5).eval()
    # state_dict = torch.load('web_nature_model3_viewpoint_0.9797.pth', map_location='cpu')
    # model_viewpoint.load_state_dict(state_dict)
    # model_viewpoint = model_viewpoint.to(device)

    # Calculate model parameters
    model_parameters = filter(
        lambda p: p.requires_grad, mymodel.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f'Model Parameters (M): {params/1e6}')

    # class weight
    _, cls_sample_count = np.unique(image_datasets['train'].label_model, return_counts=True)
    cls_weight = cls_sample_count.sum() / (len(cls_sample_count) * cls_sample_count)
    cls_weight = torch.tensor(cls_weight, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=cls_weight if use_cls_weight_for_loss else None).to(device)
    # criterion = LabelSmoothingLoss(num_classes, 0.1).cuda()
    alt_criterion = None
    # alt_criterion = CenterLoss()
    # feature_center = torch.zeros(num_classes, 2048).cuda()
    # alt_criterion = CenterLossv1(num_classes, 2048).to(device)
    # alt_criterion = CenterLossv2(num_classes, 2048).to(device)
    # alt_criterion = supervisor()

    if use_mixup:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = Mixup(mixup_alpha=0.1, cutmix_alpha=0.1, cutmix_minmax=None,
                        prob=0.5, switch_prob=0.5, mode='batch',
                        label_smoothing=0.1, num_classes=num_classes)

    params_to_update = mymodel.parameters()
    if warmup_epoch > 0:
        warmup_optimizer_ = optim.SGD(
            params_to_update, lr=warmup_lr, momentum=0.9, 
            weight_decay=5e-4)
        # warmup_lr_scheduler_ = optim.lr_scheduler.StepLR(
        #     warmup_optimizer_, warmup_epoch*len(dataloaders['train'])//3+1, gamma=10,
        #     verbose=False
        # )
        warmup_lr_scheduler_ = optim.lr_scheduler.LinearLR(
            warmup_optimizer_, start_factor=1/3, total_iters=warmup_epoch
        )
        logging.info('Warming up...')
        mymodel = train_model(
            dataloaders, mymodel, warmup_optimizer_, warmup_lr_scheduler_,
            warmup_epoch, model_no, False, True)
        logging.info('Warming up completed')

    optimizer_ = optim.SGD(
        params_to_update, lr=learning_rate, momentum=0.9,
        weight_decay=5e-4)
    # optimizer_ = optim.SGD([
    #     {'params': params_to_update}, 
    #     {'params': alt_criterion.parameters()}],
    #     lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer_ = optim.Adam(
    #    mymodel.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    # if model_no in [9]:
    #     optimizer = optim.SGD([
    #         {'params': mymodel.classifier_concat.parameters(), 'lr': 0.01},
    #         {'params': mymodel.conv_block1.parameters(), 'lr': 0.01},
    #         {'params': mymodel.classifier1.parameters(), 'lr': 0.01},
    #         {'params': mymodel.conv_block2.parameters(), 'lr': 0.01},
    #         {'params': mymodel.classifier2.parameters(), 'lr': 0.01},
    #         {'params': mymodel.conv_block3.parameters(), 'lr': 0.01},
    #         {'params': mymodel.classifier3.parameters(), 'lr': 0.01},
    #         {'params': mymodel.features.parameters(), 'lr': 0.001}]
    # optimizer_ = optim.AdamW(params_to_update, lr=learning_rate, weight_decay=5e-4)

    # Scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    lr_scheduler_ = optim.lr_scheduler.StepLR(
        optimizer_, step_size=20, gamma=0.1)
    # lr_scheduler_ = optim.lr_scheduler.MultiStepLR(
    # #     optimizer_, [10, 40, 80], 0.1)
    # lr_scheduler_ = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_, 
    #     T_max=epochs,
    #     # T_max=epochs*len(dataloaders["train"]), 
    #     eta_min=1e-4, verbose=False)
    # lr_scheduler_ = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer_,
    #     # 10, 80,
    #     10*len(dataloaders['train']), 80*len(dataloaders['train']),
    #     1e-4, verbose=False)
    
    # Check below for distributed data parallel training
    # https://github.com/microsoft/FocalNet/blob/main/main.py
    mymodel = train_model(
        dataloaders, mymodel, optimizer_, lr_scheduler_, 
        epochs, model_no, lr_scheduler_epoch_update, False
    )
