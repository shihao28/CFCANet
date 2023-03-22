import os
import torch
from torchvision import transforms, models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM,
    XGradCAM, EigenCAM, FullGrad)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data import MyDataset
from model import *
# from model.cgnet import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): torch.cuda.set_device(device)
batch_size = 64
num_classes = 196
data_dir = 'D:/OneDrive - Universiti Malaya/PhD/data/Cars'
use_bbox_train = False # for MyDataset class
use_bbox_val = False # for MyDataset class
use_model = True # for MyDataset class
use_make = False # for MyDataset class
use_type = False # for MyDataset class
use_viewpoint = False # for MyDataset class
model_no = 3
use_my_model = True
aux_logits = False # For GoogLeNet only
model_path = 'D:/OneDrive - Universiti Malaya/PhD/CGNet/Stanford Car/car_modelPMG3_TrainResizeRandomcropHflip_Randomerase_Batchsampler_Dilation3_InitweightCar224_PretrainedWeightTrue_initFCWeightFalse_448_nobbox_Labelsmooth0.1_0.9540.pth'
num_images_to_view = 10  # grad-cam
num_classes_to_view = num_classes  # tsne
use_dp = True # For data parallel ops, change gpu id in code below
dp_gpu_id = [6, 7]


def create_exp_dir():
    # Create new dir for new run
    dirs = os.listdir('results')
    dir_count = re.search('\d+', sorted(dirs)[-1])
    if dir_count is None:
        dest_dir = 'run000'
    else:
        exp_name = dir_count.group()
        dest_dir = f'run{int(exp_name) + 1:03d}'
    
    return dest_dir


def viz_gradcam(dataloaders, model, device, num_images_to_view, selected_img_path=[]):
    model.eval()

    # Create new dir for new run
    dest_dir = create_exp_dir()
    os.makedirs(os.path.join('results', dest_dir))

    # Get images
    if len(selected_img_path) == 0:
        inputs_path, inputs, labels  = next(iter(dataloaders['val']))
        inputs_path = inputs_path[:num_images_to_view]
        inputs = inputs[:num_images_to_view]
        if len(labels) > 0:
            labels = labels[0][:num_images_to_view]
    else:
        inputs_path_tmp, inputs_tmp, labels_tmp = [], [], []
        for dataloader_tmp in [dataloaders['train'], dataloaders['val']]:
            for inputs_path, inputs, labels in tqdm(dataloader_tmp):
                inputs_path = np.array(list(inputs_path))
                for selected_img_path_tmp in selected_img_path:
                    idx = np.where(inputs_path==selected_img_path_tmp)[0]
                    if len(idx) == 0:
                        continue
                    idx = idx.item()
                    inputs_path_tmp.append(inputs_path[idx])
                    inputs_tmp.append(inputs[idx])
                    labels_tmp.append(labels[0][idx])
        inputs_path = np.stack(inputs_path_tmp, 0)
        inputs = torch.stack(inputs_tmp, 0)
        labels = torch.stack(labels_tmp, 0)
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Define target layer
    if use_my_model:
        target_layers = [model.ms4]
    else:
        target_layers = [model.layer4[-1]]

    # Get prediction
    # Caution: GradCAM does not accept more than one output returned by model
    logits = model(inputs)
    _, preds = logits.max(-1)

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(label.item()) for label in labels] if len(labels) > 0 else None
    grayscale_cam = cam(input_tensor=inputs, targets=targets, aug_smooth=False, eigen_smooth=False)
    for i, grayscale_cam_tmp in enumerate(grayscale_cam):
        
        trans1 = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop((224, 224)),

            transforms.Resize((448, 448)),
        ])

        rgb_img = Image.open(str(inputs_path[i]))
        rgb_img = trans1(rgb_img)
        rgb_img = np.float32(np.asarray(rgb_img)) / 255
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam_tmp, use_rgb=True)
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        # cv2.putText(
        #     visualization, f'{preds[i]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=0.5, color=(255, 0, 0), thickness=1
        # )

        path_tmp = str(inputs_path[i]).split(os.path.sep)
        path_tmp_start_index = path_tmp.index('data')
        output_path = os.path.join(
            'results', dest_dir, 
            f"model{'' if use_my_model else 'ori'}{model_no}_" +\
                '_'.join(str(inputs_path[i]).split(os.path.sep)[path_tmp_start_index+1:]))
        cv2.imwrite(output_path, visualization)

    return None


@torch.no_grad()
def viz_tsne(dataloaders_val, model, device, num_classes_to_view=20, classes=[]):
    model.eval()

    # Create new dir for new run
    dest_dir = create_exp_dir()
    os.makedirs(os.path.join('results', dest_dir))

    if len(classes) == 0:
        # get classes with most number of images
        classes, classes_count = np.unique(dataloaders_val.dataset.label_model, return_counts=True)
        classes = classes[np.argsort(classes_count)[-num_classes_to_view:]]
    
    idx = list(map(lambda x: dataloaders_val.dataset.label_model == x, classes))
    idx = np.stack(idx, 1)
    idx = idx.any(1)
    dataloaders_val.dataset.img_path = dataloaders_val.dataset.img_path[idx]
    dataloaders_val.dataset.label_model = dataloaders_val.dataset.label_model[idx]
    
    # Get tsne features
    all_features = []
    all_labels = []
    for _, inputs, labels in tqdm(dataloaders_val):
        inputs = inputs.to(device)
        if use_my_model:
            features, _ = model(inputs)
        else:
            # For ResNet50 only
            features = model.conv1(inputs)
            features = model.bn1(features)
            features = model.relu(features)
            features = model.maxpool(features)
            features = model.layer1(features)
            features = model.layer2(features)
            features = model.layer3(features)
            features = model.layer4(features)
            features = model.avgpool(features).flatten(1)

        all_features.append(features)
        all_labels.append(labels[0])
    all_features = torch.cat(all_features, 0).cpu().numpy()
    all_labels = torch.cat(all_labels, 0).numpy()

    # TSNE
    all_features_transformed = TSNE(
        n_components=2, random_state=100
        ).fit_transform(all_features)
    
    # Get vehicle model name
    # if data_dir.find('web') != -1:
    #     class_mapping_model = np.load(os.path.join(
    #         data_dir, 'class_mapping_model_webnature.npy'),
    #         allow_pickle=True).flatten()[0]
    # elif data_dir.find('mohsin_vmmr') != -1:
    #     class_mapping_model = np.load(os.path.join(
    #         data_dir, 'class_mapping_model.npy'),
    #         allow_pickle=True).flatten()[0]
    # class_mapping_model = {v: k for k, v in class_mapping_model.items()}
    # # Need +1 since class index in class_mapping_model starts from 1
    # all_labels_name = [class_mapping_model[i+1] for i in all_labels]

    # Plot and save
    fig, ax = plt.subplots()
    ax.scatter(
        all_features_transformed[:,0], all_features_transformed[:,1],
        c=all_labels, s=5,
        # label=all_labels_name
    )
    ax.set_ylim(-100, 100)
    ax.set_axis_off()
    # ax.legend()
    fig.savefig(os.path.join('results', dest_dir, 'tsne.jpg'), dpi=600, bbox_inches="tight")

    return None


@torch.no_grad()
def viz_conf_matrix(dataloaders_val, model, device):
    model.eval()

    # Create new dir for new run
    dest_dir = create_exp_dir()
    os.makedirs(os.path.join('results', dest_dir))

    # Get prediction and labels
    all_preds = []
    all_labels = []
    for _, inputs, labels in tqdm(dataloaders_val):
        inputs = inputs.to(device)
        labels = labels[0].to(device)

        _, logits = model(inputs)
        _, preds = logits.max(1)

        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = torch.cat(all_preds, 0).cpu().numpy()
    all_labels = torch.cat(all_labels, 0).cpu().numpy()

    # Get vehicle model name
    # if data_dir.find('web') != -1:
    #     class_mapping_model = np.load(os.path.join(
    #         data_dir, 'class_mapping_model_webnature.npy'),
    #         allow_pickle=True).flatten()[0]
    # elif data_dir.find('mohsin_vmmr') != -1:
    #     class_mapping_model = np.load(os.path.join(
    #         data_dir, 'class_mapping_model.npy'),
    #         allow_pickle=True).flatten()[0]
    # class_mapping_model = {v: k for k, v in class_mapping_model.items()}
    # all_labels_name = list(class_mapping_model.values())

    # Plot and save conf matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap='Blues')  # jet
    ax.set_title('Confusion Matrix')
    fig.colorbar(cax)
    fig.savefig(
        os.path.join('results', dest_dir, 'conf_matrix.jpg'),
        dpi=600, bbox_inches="tight")

    # conf_matrix = ConfusionMatrixDisplay(
    #     confusion_matrix=conf_matrix,
    #     # display_labels=all_labels_name,
    #     ).plot(cmap='Reds')
    # conf_matrix.figure_.savefig(
    #     os.path.join('results', dest_dir, 'conf_matrix.jpg'),
    #     dpi=600, bbox_inches="tight")

    return None


if __name__ == '__main__':
    
    data_transforms = {
        'val': transforms.Compose([
            # CompCars
            # transforms.Resize(256),
            # transforms.CenterCrop((224, 224)),
            # transforms.Resize((224, 224)),
            
            # Cars
            transforms.Resize((448, 448)),
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
        use_viewpoint=use_viewpoint, transform=data_transforms['val'], 
        # combine_vtr_label=True,
        # exp=3
        # use_top_n_model=[],
        # use_n_img_per_cls=250,
        # small_dataset=True,
        # remove_poor_images=True
        ) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, 
        drop_last=False
        )
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

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
            # mymodel = models.resnet18(pretrained=False)
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
    
    # Grad-CAM
    selected_img_path = [
        # CompCarsWeb
        # Toyota Prius
        # os.path.join(data_dir, 'image/39/1327/2012/58a5545540e42d.jpg'),
        # os.path.join(data_dir, 'image/39/1327/2012/0581b8771e4862.jpg'),
        # os.path.join(data_dir, 'image/39/1327/2012/a81ee9ebeb9309.jpg'),
        # os.path.join(data_dir, 'image/39/1327/2006/c6122e387c8da5.jpg'),
        # os.path.join(data_dir, 'image/39/1327/2012/f1e0f0100eb926.jpg'),

        # # Honda Accord
        # os.path.join(data_dir, 'image/100/213/2011/c06eb868949c0b.jpg'),
        # os.path.join(data_dir, 'image/100/213/2012/ed6d0e5a66709e.jpg'),
        # os.path.join(data_dir, 'image/100/213/2010/27bb6c344896a5.jpg'),
        # os.path.join(data_dir, 'image/100/213/2012/6a6675293a7e58.jpg'),
        # os.path.join(data_dir, 'image/100/213/2012/f0bfa982cace1a.jpg'),

        # # Citroen Elysee
        # os.path.join(data_dir, 'image/158/1253/2014/700905d6eecd2b.jpg'),
        # os.path.join(data_dir, 'image/158/1253/2014/160322be6d6774.jpg'),
        # os.path.join(data_dir, 'image/158/1253/2013/35e021c0dff0a7.jpg'),
        # os.path.join(data_dir, 'image/158/1253/2013/f3e610713b410e.jpg'),
        # os.path.join(data_dir, 'image/158/1253/2014/fcac1b818b1578.jpg'),

        # # Volvo V40 CrossCrountry
        # os.path.join(data_dir, 'image/111/1704/2014/3e045e7109aa4a.jpg'),
        # os.path.join(data_dir, 'image/111/1704/2014/e303b2428bea80.jpg'),
        # os.path.join(data_dir, 'image/111/1704/2014/158146b14e7148.jpg'),
        # os.path.join(data_dir, 'image/111/1704/2014/82dd723da91cac.jpg'),
        # os.path.join(data_dir, 'image/111/1704/2014/b4e36d16309ebc.jpg'),

        # # Benz S Class
        # os.path.join(data_dir, 'image/77/162/2014/c8b31e3723c1ce.jpg'),
        # os.path.join(data_dir, 'image/77/162/2014/5b4728ab06e447.jpg'),
        # os.path.join(data_dir, 'image/77/162/2014/e9905903cc5bd8.jpg'),
        # os.path.join(data_dir, 'image/77/162/2012/7bdfbd75f6d6d4.jpg'),
        # os.path.join(data_dir, 'image/77/162/2012/10745323bcc2e1.jpg'),

        # Stanford Car
        # Audi S6 Sedan 2011
        os.path.join(data_dir, 'cars_train/04802.jpg'),
        os.path.join(data_dir, 'cars_train/04237.jpg'),
        os.path.join(data_dir, 'cars_train/04195.jpg'),
        os.path.join(data_dir, 'cars_train/03157.jpg'),
        os.path.join(data_dir, 'cars_train/00681.jpg'),

        # Hyundai Sonata Sedan 2012
        os.path.join(data_dir, 'cars_train/03410.jpg'),
        os.path.join(data_dir, 'cars_train/00240.jpg'),
        os.path.join(data_dir, 'cars_train/06562.jpg'),
        os.path.join(data_dir, 'cars_train/00911.jpg'),
        os.path.join(data_dir, 'cars_train/02837.jpg'),
        os.path.join(data_dir, 'cars_test/05599.jpg'),  # front

        # Volkswagen Golf Hatchback 2012
        os.path.join(data_dir, 'cars_train/05766.jpg'),
        os.path.join(data_dir, 'cars_train/00062.jpg'),
        os.path.join(data_dir, 'cars_train/02742.jpg'),
        os.path.join(data_dir, 'cars_train/03922.jpg'),
        os.path.join(data_dir, 'cars_train/02705.jpg'),
        os.path.join(data_dir, 'cars_test/04991.jpg'),  # rear

        # Rolls-Royce Phantom Sedan 2012
        os.path.join(data_dir, 'cars_train/04698.jpg'),
        os.path.join(data_dir, 'cars_train/04465.jpg'),
        os.path.join(data_dir, 'cars_train/04084.jpg'),
        os.path.join(data_dir, 'cars_train/07000.jpg'),
        os.path.join(data_dir, 'cars_train/03368.jpg'),
        os.path.join(data_dir, 'cars_test/07721.jpg'),  # rear-side

        # Land Rover Range Rover SUV 2012
        os.path.join(data_dir, 'cars_train/04320.jpg'),
        os.path.join(data_dir, 'cars_train/06171.jpg'),
        os.path.join(data_dir, 'cars_train/08052.jpg'),
        os.path.join(data_dir, 'cars_train/02090.jpg'),
        os.path.join(data_dir, 'cars_train/03567.jpg'),
    ]
    # viz_gradcam(dataloaders, mymodel, device, num_images_to_view, selected_img_path)

    # TSNE
    # viz_tsne(
    #     dataloaders['val'], mymodel, device,
    #     num_classes_to_view=num_classes_to_view, classes=[]
    # )

    # Conf-matrix
    viz_conf_matrix(dataloaders['val'], mymodel, device)
