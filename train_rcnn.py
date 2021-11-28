import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from data.apple_dataset import AppleDataset
from utility.engine import train_one_epoch, evaluate

import utility.utils as utils
import utility.transforms as T

######################################################
# Entrenar un modelo Faster-RCNN o Mask-RCNN (Mask no ha sido arreglado)
######################################################


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_maskrcnn_model_instance(num_classes):
    # carga una instancia del modelo pre-entrenado con COCO (pycocotools)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # consigue el numero de caracteristicas de entrada para el clasificador
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # remplaza el head pre-entrenado con uno nuevo
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # consigue el numero de caracteristicas de entrada para el clasificador
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    #Remplaza el mask predictor con uno nuevo
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def get_frcnn_model_instance(num_classes):
    # carga una instancia del modelo pre-entrenado con COCO (pycocotools)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # consigue el numero de caracteristicas de entrada para el clasificador
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # remplaza el head pre-entrenado con uno nuevo
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(args):
    print(args)
    device = args.device

    # Codigo de carga de datos
    print("Leyendo datos")
    num_classes = 2
    dataset = AppleDataset(os.path.join(args.data_path, 'train'), get_transform(train=True))
    dataset_test = AppleDataset(os.path.join(args.data_path, 'test'), get_transform(train=False))

    print("Crendo cargadores de datos")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   shuffle=False, num_workers=args.workers,
                                                   collate_fn=utils.collate_fn)

    print("Creando modelo")
    # Crea el correcto tipo de modelo
    args.model == 'frcnn'
    if args.model == 'maskrcnn':
        model = get_maskrcnn_model_instance(num_classes)
    else:
        model = get_frcnn_model_instance(num_classes)

    # Mueve el modelo al device correcto
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print("Comienza el entrenamiento")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()

        if args.output_dir:
            torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },  os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # Evaluacion despues de cada epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Tiempo de entrenamiento {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')
    parser.add_argument('--data_path', default='/home/matiasmoreno/ProyectoML/data', help='dataset')
    parser.add_argument('--dataset', default='AppleDataset', help='dataset')
    parser.add_argument('--model', default='frcnn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/home/matiasmoreno/ProyectoML/resultados', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    args = parser.parse_args()
    print(args.model)
    #assert(args.model in ['frcnn'])

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
