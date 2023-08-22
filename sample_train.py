import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import timm

from tqdm import tqdm
import time

from datasets.cct20 import get_cct20

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 256 # may need to change to 224 for certain models

    trainset, testset_cis, testset_true = get_cct20(img_size)
    num_classes = len(set(trainset.labels)) # 16

    models = ['Resnet50', 'AlexNet', 'VGG13BN', 'PNASnet5', 'Resnet152', 'MobileNetV3Small'] # See below for additional models
    pretrained = True
    opt = 'adamw' # ['adamw', 'sgd']
    scheduler = 'cosine' # ['cosine', 'step']
    if opt == 'sgd':
        lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    elif opt == 'adamw':
        lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    wds = [1e-3, 1e-4, 1e-5, 1e-6, 0]
    for model_name in models:
        grid_cis = []
        grid_test = []
        for lr_init in lrs:
            for wd in wds:
                print(model_name)
                start = time.time()
                if model_name == 'Resnet152':
                    net = torchvision.models.resnet152(pretrained=pretrained)
                    net.fc = nn.Linear(net.fc.in_features, num_classes)
                elif model_name =='Resnet50':
                    net = torchvision.models.resnet50(pretrained=pretrained)
                    net.fc = nn.Linear(net.fc.in_features, num_classes)
                elif model_name == 'PNASnet5':
                    net = timm.create_model('pnasnet5large', pretrained=pretrained, num_classes=num_classes)
                elif model_name =='VGG13BN':
                    net = torchvision.models.vgg13_bn(pretrained=pretrained)
                    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
                elif model_name =='MobileNetV3Small':
                    net = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
                    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
                elif model_name =='AlexNet':
                    net = torchvision.models.alexnet(pretrained=pretrained)
                    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
                elif model_name =='DeiT-tiny':
                    net = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
                elif model_name =='DeiT-small':
                    net = timm.create_model('deit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
                elif model_name == 'InceptionResnetv2':
                    net = timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=num_classes)
                elif model_name == 'VGG16BN':
                    net = torchvision.models.vgg16_bn(pretrained=pretrained)
                    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
                elif model_name == 'EfficientNetB0':
                    net = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
                elif model_name == 'EfficientNetB4':
                    net = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
                elif model_name == 'ConvNext-tiny':
                    net = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
                elif model_name == 'DenseNet121':
                    net = torchvision.models.densenet121(pretrained=pretrained)
                    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
                elif model_name == 'ResNext50-32x4d':
                    net = torchvision.models.resnext50_32x4d(pretrained=pretrained)
                    net.fc = nn.Linear(net.fc.in_features, num_classes)
                elif model_name == 'ShuffleNetv2':
                    net = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
                    net.fc = nn.Linear(net.fc.in_features, num_classes)
                elif model_name == 'SqueezeNet':
                    net = torchvision.models.squeezenet1_1(pretrained=pretrained)
                    net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
                elif model_name == 'SqueezeNetLin':
                    net = torchvision.models.squeezenet1_1(pretrained=pretrained)
                    class Flatten(nn.Module):
                        def forward(self, x): return x.view(x.size(0), x.size(1))
                    net.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)), Flatten(), nn.Linear(512, num_classes))
                elif model_name == 'ShuffleNetv2_05':
                    net = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained)
                    net.fc = nn.Linear(net.fc.in_features, num_classes)
                elif model_name == 'MnasNet05':
                    net = torchvision.models.mnasnet0_5(pretrained=pretrained)
                    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
                elif model_name == "caformer_b36":
                    net = timm.create_model('caformer_b36.sail_in1k', pretrained=pretrained, num_classes=num_classes)
                elif model_name == "swin_base":
                    net = timm.create_model('swin_base_patch4_window7_224.ms_in1k', pretrained=pretrained, num_classes=num_classes)
                elif model_name == "mixer_b16":
                    net = timm.create_model('mixer_b16_224', pretrained=pretrained, num_classes=num_classes)
                elif model_name == "tinynet_e":
                    net = timm.create_model('tinynet_e.in1k', pretrained=pretrained, num_classes=num_classes)
                elif model_name == "tinynet_d":
                    net = timm.create_model('tinynet_d.in1k', pretrained=pretrained, num_classes=num_classes)
                elif model_name == "dla46_c":
                    net = timm.create_model('dla46_c.in1k', pretrained=pretrained, num_classes=num_classes)
                elif model_name == "dla46x_c":
                    net = timm.create_model('dla46x_c.in1k', pretrained=pretrained, num_classes=num_classes)
                batch_size = 128
                testloader_cis = torch.utils.data.DataLoader(testset_cis, batch_size=batch_size,
                                                             shuffle=False, num_workers=16)
                testloader_true = torch.utils.data.DataLoader(testset_true, batch_size=batch_size,
                                                             shuffle=False, num_workers=16)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                            shuffle=True, num_workers=16)
                if torch.cuda.device_count() > 1:
                    print('Multiple GPUs')
                    net = nn.DataParallel(net)
                net = net.to(device)

                criterion = nn.CrossEntropyLoss()

                epochs = 50
                if pretrained:
                    epochs = 30
                best_acc = 0
                cis_best = 0
                true_best = 0
                
                if opt == 'sgd':
                    optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, weight_decay=wd, nesterov=True)
                elif opt == 'adamw':
                    optimizer = optim.AdamW(net.parameters(), lr=lr_init, weight_decay=wd/lr_init)
                if scheduler == 'step':
                    if pretrained:
                        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 25], gamma=0.1, last_epoch=-1)
                    else:
                        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.1, last_epoch=-1)
                elif scheduler == 'cosine':
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

                for epoch in range(epochs):
                    print(epoch)
                    running_loss = 0.0
                    net.train()
                    train_correct = 0
                    train_total = 0
                    for i, data in enumerate(tqdm(trainloader), 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = net(inputs).view(-1, num_classes)
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    lr_scheduler.step()

                    net.eval()

                    test_correct = 0
                    test_total = 0
                    test_loss = 0
                    with torch.no_grad():
                        for inputs, labels in testloader_cis:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = net(inputs).view(-1, num_classes)
                            _, predicted = torch.max(outputs.data, 1)
                            test_total += labels.size(0)
                            test_correct += (predicted == labels).sum().item()
                    cis_acc = test_correct/test_total*100
                    test_correct = 0
                    test_total = 0
                    test_loss = 0
                    with torch.no_grad():
                        for inputs, labels in testloader_true:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = net(inputs).view(-1, num_classes)
                            _, predicted = torch.max(outputs.data, 1)
                            test_total += labels.size(0)
                            test_correct += (predicted == labels).sum().item()
                    true_acc = test_correct/test_total*100
                    print("Seen locations Val")
                    print(cis_acc)
                    if cis_acc > cis_best:
                        cis_best = cis_acc
                        best_params = '_' + str(opt) + '_'+ scheduler + '_lr' + str(lr_init) + '_wd' + str(wd)
                        best_detail = '_epoch' + str(epoch) + '_cis' + str(int(cis_acc*100)) + '_test' + str(int(true_acc*100))
                        curr_model_wts = net.state_dict()
                        true_best = true_acc
                    print("Test")
                    print(true_acc)
                print(model_name)
                print("Train time (min): " + str((time.time() - start)/60))
                print("LR: " + str(lr_init))
                print("WD: " + str(wd))
                print("CIS Best")
                print(round(cis_best, 2))
                print("Test")
                print(round(true_best, 2))
                grid_cis.append(round(cis_best, 2))
                grid_test.append(round(true_best, 2))
                net = net.cpu()
        print(model_name)
        print(np.array(grid_cis).reshape(-1, len(lrs), len(wds)))
        print(np.amax(np.array(grid_cis).reshape(-1, len(lrs), len(wds))))
        print(np.array(grid_test).reshape(-1, len(lrs), len(wds)))
        print(np.amax(np.array(grid_test).reshape(-1, len(lrs), len(wds))))

if __name__ == "__main__":
    main()
