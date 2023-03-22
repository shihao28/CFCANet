import logging
import torch
from torchvision import transforms
from thop import profile
from pthflops import count_ops
from tqdm import tqdm

from data import MyDataset
from model import *
from model.convnext import *
from model.tresnet.tresnet import *
from losses import *
from utils import AverageMeter, accuracy
from sklearn.metrics import classification_report


# Set log level
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) | %(levelname)-8s | %(module)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): torch.cuda.set_device(device)
batch_size = 2048
num_classes = 196
data_dir = 'D:/OneDrive - Universiti Malaya/PhD/data/Cars'
use_bbox_train = False # for MyDataset class
use_bbox_val = False # for MyDataset class
use_model = True # for MyDataset class
use_make = False # for MyDataset class
use_type = False # for MyDataset class
use_viewpoint = False # for MyDataset class
model_no = 9
use_my_model = False
aux_logits = False # For GoogLeNet only
use_dp = True # For data parallel ops, change gpu id in code below
dp_gpu_id = [0, 1]
model_path = 'D:/tresnet_l_stanford_card_96.41.pth'


def eval_model(dataloaders_val, model_no, model, criterion, device, model_viewpoint=None,
               alt_criterion=None,print_cls_report=False):
    model.eval()
    criterion.eval()
    val_epoch_loss = AverageMeter()
    val_epoch_accuracy = AverageMeter()

    with torch.set_grad_enabled(False):
        labels_all = []
        preds_all = []
        for _, inputs, labels in tqdm(dataloaders_val):
            inputs = inputs.to(device)
            labels = labels[0].to(device)

            # viewpoint_emb = model_viewpoint.conv1(inputs)
            # viewpoint_emb = model_viewpoint.bn1(viewpoint_emb)
            # viewpoint_emb = model_viewpoint.relu(viewpoint_emb)
            # viewpoint_emb = model_viewpoint.maxpool(viewpoint_emb)
            # viewpoint_emb = model_viewpoint.layer1(viewpoint_emb)
            # viewpoint_emb = model_viewpoint.layer2(viewpoint_emb)
            # viewpoint_emb = model_viewpoint.layer3(viewpoint_emb)
            # viewpoint_emb = model_viewpoint.layer4(viewpoint_emb)
            # viewpoint_emb = model_viewpoint.avgpool(viewpoint_emb).flatten(1)
            # viewpoint_logits = model_viewpoint(inputs)
            
            # x3_logits, x4_logits, x5_logits, xcat_logits = model(inputs)
            # val_batch_loss = criterion(x3_logits, labels) + criterion(x4_logits, labels) + criterion(x5_logits, labels) + criterion(xcat_logits, labels)
            # logits = x3_logits + x4_logits + x5_logits + xcat_logits
            
            if model_no != 2:

                logits = model(inputs)
                # features, logits = model(inputs)
                val_batch_loss = criterion(logits, labels.long())
                if alt_criterion is not None:
                    val_batch_loss += 0.005*alt_criterion(features, labels, 7, num_classes)

                # logits, logits_lrau = model(inputs)
                # val_batch_loss = 0.5*criterion(logits, labels) + 0.5*criterion(logits_lrau, labels)

                # logits, logits3, logits4 = model(inputs)
                # val_batch_loss = (criterion(logits, labels) + criterion(logits3, labels) + criterion(logits4, labels)) / 3
                # logits = (logits + logits3 + logits4) / 3
                
                # Grad-CAM
                # logits = model(inputs, 1)
                # val_batch_loss = criterion(logits, labels)

                # WS-DAN
                # attention_map, logits = model(inputs, 0)
                # with torch.no_grad():
                #     crop_images = batch_augment(inputs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
                # logits_crop = model(crop_images, 1)
                # logits = (logits + logits_crop) / 2
                # val_batch_loss = criterion(logits, labels)

                acc1 = accuracy(logits, labels.data)[0]

                # acc1 = accuracy((logits + logits_lrau)/2, labels.data)[0]
            else:
                logits = model(inputs)
                val_batch_loss = criterion(logits, labels)
                if alt_criterion is not None:
                    val_batch_loss += 0.005*alt_criterion(features, labels, 7, num_classes)
                acc1 = accuracy(logits, labels.data)[0]

            val_epoch_loss.update(val_batch_loss, inputs.size(0))
            val_epoch_accuracy.update(acc1.item(), inputs.size(0))
            
            if print_cls_report:
                labels_all.append(labels)
                preds_all.append(logits.argmax(1))

        if print_cls_report:
            labels_all = torch.cat(labels_all, 0).cpu().numpy()
            preds_all = torch.cat(preds_all, 0).cpu().numpy()
            cls_report = classification_report(labels_all, preds_all, digits=4)
            logging.info(cls_report)

    return val_epoch_loss.avg, val_epoch_accuracy.avg


if __name__ == '__main__':
    
    data_transforms = {
        'val': transforms.Compose([
            # CompCars
            # transforms.Resize(256),
            # transforms.CenterCrop((224, 224)),
            
            # Cars
            transforms.Resize((384, 384)),
            # transforms.Resize((512, 512)),
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
        # remove_poor_images=True
        ) for x in ['val']}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4, 
        drop_last=False
        )
        for x in ['val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

    if model_no == 0:
        # alexnet
        if use_my_model:
            mymodel = alexnet(pretrained=False, num_classes=num_classes)
        else:
            mymodel = models.alexnet(pretrained=False, num_classes=1000)
            num_ftrs = mymodel.classifier[-1].in_features
            mymodel.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    elif model_no == 1:
        # vgg16
        if use_my_model:
            mymodel = vgg16(pretrained=False, num_classes=num_classes, use_my_model=True)
        else:
            mymodel = vgg16(pretrained=False, num_classes=1000, use_my_model=False)
            num_ftrs = mymodel.classifier[-1].in_features
            mymodel.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            
    elif model_no == 2:
        # Inceptionv3
        if use_my_model:         
            mymodel = inception_v3(
                pretrained=False, num_classes=num_classes,
                aux_logits=aux_logits)
        else:            
            mymodel = models.inception_v3(
                pretrained=False, num_classes=1000, 
                aux_logits=aux_logits)
            if aux_logits:
                num_ftrs = mymodel.AuxLogits.fc.in_features
                mymodel.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            num_ftrs = mymodel.fc.in_features
            mymodel.fc = nn.Linear(num_ftrs, num_classes)

    elif model_no == 3:
        # resnet50
        if use_my_model:
            mymodel = resnet50(pretrained=False, num_classes=num_classes)
        else:
            mymodel = models.resnet50(pretrained=False, num_classes=1000)
            num_ftrs = mymodel.fc.in_features
            mymodel.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_no == 4:
        # densenet161, densenet169
        if use_my_model:
            mymodel = densenet169(pretrained=False, num_classes=num_classes)
        else:
            mymodel = models.densenet169(pretrained=False, num_classes=1000)
            num_ftrs = mymodel.classifier.in_features
            mymodel.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_no == 5:
        raise NotImplementedError("Please use other script for ViT")
    elif model_no == 6:
        raise NotImplementedError("Please use other script for TransFG")

    elif model_no == 7:
        # MobileNet
        if use_my_model:
            mymodel = mobilenetv3_large(pretrained=False, num_classes=num_classes, use_my_model=True)
        else:
            # mymodel = models.mobilenet_v3_small(pretrained=True, num_classes=1000)
            # num_ftrs = mymodel.classifier[-1].in_features
            # mymodel.classifier[-1] = nn.Linear(num_ftrs, num_classes)

            mymodel = mobilenetv3_large(pretrained=False, num_classes=num_classes, use_my_model=False)

    elif model_no == 8:
        # Convnext
        if use_my_model:
            mymodel = convnext_tiny(pretrained=False, in_22k=True, num_classes=num_classes, use_my_model=True)
        else:
            mymodel = convnext_tiny(pretrained=False, in_22k=True, num_classes=num_classes, use_my_model=False)

    elif model_no == 9:
        # TResNet
        if use_my_model:
             mymodel = TResnetM(
                pretrained=False, num_classes=num_classes, remove_aa_jit=False,
                use_my_model=True, use_ml_decoder=False,
                num_of_groups=-1, decoder_embedding=768, zsl=0
            )
        else:
            mymodel = TResnetL(
                pretrained=False, num_classes=num_classes, remove_aa_jit=False,
                use_my_model=False, use_ml_decoder=True,
                num_of_groups=-1, decoder_embedding=768, zsl=0
            )

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')['model']
    mymodel.load_state_dict(state_dict)

    mymodel = mymodel.to(device)

    # Calculate params and flops
    # input_dummy = torch.rand((1, 3, 224, 224)).to(device)
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
    params_to_update = mymodel.parameters()
    
    criterion = nn.CrossEntropyLoss().to(device)  # LabelSmoothingLoss(num_classes, 0.1)

    val_epoch_loss, val_epoch_accuracy = eval_model(
        dataloaders['val'], model_no, mymodel, criterion, device,
        model_viewpoint=model_viewpoint, print_cls_report=True
    )
