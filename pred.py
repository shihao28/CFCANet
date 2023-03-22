import os
import logging
import torch
from torchvision import transforms
import numpy as np

from data import MyDataset
from model import *
from losses import *


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
batch_size = 64
num_classes = 5
data_dir = '../yolov5-master/data/car'
model_no = 3
is_train = True  # Set True to load train dataset
use_my_model = False
aux_logits = False # For GoogLeNet only
use_dp = True # For data parallel ops, change gpu id in code below
dp_gpu_id = [0, 1]
model_path = "web_nature_model3_0.9797.pth"
save_pred = True
save_path = "label_viewpoint.npy"


def pred_model(dataloaders_val, model_no, model, device, save_path):
    model.eval()

    with torch.set_grad_enabled(False):
        preds_all = dict()

        for img_path, inputs, _ in dataloaders_val:
            inputs = inputs.to(device)
            
            if model_no != 2:               
                logits = model(inputs)
            
            elif model_no == 2:
                pass
            
            # +1 so that label starts from 1
            preds = logits.argmax(1) + 1

            for img_path_tmp, preds_tmp in zip(img_path, preds):
                img_path_tmp = str(img_path_tmp).split('/')[-1]
                preds_all[img_path_tmp] = preds_tmp.item()

    if save_pred:
        save_path = f"{save_path.split('.')[0]}_{'train' if is_train else 'test'}.npy"
        if os.path.exists(save_path):
            raise Exception("Path exists")
        np.save(save_path, preds_all)
        logging.info(f"Prediction saved to {save_path}")

    return preds_all


if __name__ == '__main__':
    
    data_transforms = {
        'pred': transforms.Compose([
            # CompCars
            # transforms.Resize(256),
            # transforms.CenterCrop((224, 224)),
            
            # Cars
            transforms.Resize((224, 224)),
            # transforms.Resize((512, 512)),
            # transforms.CenterCrop((448, 448)),
          
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {x: MyDataset(
        root=data_dir, is_train=is_train, use_bbox=False,
        use_model=False, use_make=False, use_type=False,
        use_viewpoint=False, transform=data_transforms[x], 
        # combine_vtr_label=True,
        # exp=3
        # use_top_n_model=[],
        # use_n_img_per_cls=250,
        # small_dataset=True,
        # remove_poor_images=True
        ) for x in ['pred']}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4, 
        drop_last=False
        )
        for x in ['pred']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['pred']}

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
            mymodel = vgg16(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.vgg16(pretrained=True, num_classes=1000)
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
                aux_logits=aux_logits)
            if aux_logits:
                num_ftrs = mymodel.AuxLogits.fc.in_features
                mymodel.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            num_ftrs = mymodel.fc.in_features
            mymodel.fc = nn.Linear(num_ftrs, num_classes)

    elif model_no == 3:
        # resnet50
        if use_my_model:
            mymodel = resnet50(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.resnet18(pretrained=True, num_classes=1000)
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
            mymodel = densenet169(pretrained=True, num_classes=num_classes)
        else:
            mymodel = models.mobilenet_v3_small(pretrained=True, num_classes=1000)
            num_ftrs = mymodel.classifier[-1].in_features
            mymodel.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    mymodel.load_state_dict(state_dict)

    mymodel = mymodel.to(device)
    if use_dp:
        mymodel = torch.nn.DataParallel(mymodel, device_ids=dp_gpu_id)

    # Calculate model parameters
    model_parameters = filter(
        lambda p: p.requires_grad, mymodel.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f'Model Parameters (M): {params/1e6}')
    params_to_update = mymodel.parameters()

    pred =  pred_model(dataloaders['pred'], model_no, mymodel, device, save_path)
