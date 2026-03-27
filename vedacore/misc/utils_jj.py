import copy

import cv2
import torch
import math, os
import torchvision.transforms as transforms

def drawMark_Array(im, peaks, y_yaw, yawLength=30):
    cv2.drawMarker(im, (int(peaks[0]), int(peaks[1])), (0, 255, 0))
    cv2.arrowedLine(im, (100, 100), (100 + int(yawLength * y_yaw[0]), 100 + int(yawLength * y_yaw[1])), (0, 0, 255), 2,
                    4, tipLength=0.3)
    return im


def getStat(img_pths:list):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(img_pths))
    to_tensor=transforms.ToTensor()
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X in img_pths:
        im=cv2.imread(str(X))
        assert im is not None
        
        X=to_tensor(im)
        for d in range(3):
            mean[d] += X[d, :, :].mean()
            std[d] += X[d, :, :].std()
    mean.div_(len(img_pths))
    std.div_(len(img_pths))
    return list(mean.numpy()), list(std.numpy())
 

    # train_dataset = ImageFolder(root=r'D:\cifar10_images\test', transform=None)


def initialize_weight(model, useDefault=True):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            if useDefault:
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5.0))
                if m.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1.0 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(m.bias, -bound, bound)
            else:
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            if useDefault:
                m.running_mean.zero_()
                m.running_var.fill_(1.)
                m.num_batches_tracked.zero_()
            else:
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            if useDefault:
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5.))
                if m.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(m.bias, -bound, bound)
            else:
                torch.nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Sequential):
            Seq_nn_initWeight(m)
        elif (isinstance(m, torch.nn.Module) and (not isinstance(m, type(model)))):
            initialize_weight(m)


def Seq_nn_initWeight(seq):
    for m in seq:
        if isinstance(m, torch.nn.Sequential):
            Seq_nn_initWeight(m)
        else:
            initialize_weight(m)


def saveModel(moudle, targetM, useParal, epo,mid_dir="./weights/", switchDeplay=False):
    # torch.save(moudle.cpu(), targetM + '_epo' + str(epo))
    # print('saved model epo:', str(epo))
    if switchDeplay:
        moudle_copy = copy.deepcopy(moudle)
        moudle_copy.switch_to_deploy()
    else:
        moudle_copy = moudle
    # if os.path.exists('./' + targetM + '_epo' + str(epo) + '_dict'):
    #     os.remove('./' + targetM + '_epo' + str(epo) + '_dict')
    if useParal:
        # print('saving parall ...',useParal,'  ',targetM + '_epo' + str(epo) + '_dict')
        torch.save(moudle_copy.module.state_dict(),mid_dir + targetM + '_epo' + str(epo) + '_dict')
    else:
        # print('saving no parall ...False','  ',targetM + '_epo' + str(epo) + '_dict')
        torch.save(moudle_copy.state_dict(),mid_dir + targetM + '_epo' + str(epo) + '_dict')
    print('Model  has been saved:', targetM + '_epo' + str(epo) + '_dict')
    del moudle_copy


def resizeIm(im, w=320, h=256,pad_value=0):
    origScale = im.shape[0] / im.shape[1]
    targetScale = h / w
    wValue = w
    hValue = h
    if origScale < targetScale:
        hValue = int(im.shape[0] * w / im.shape[1])
        im = cv2.resize(im, (int(w), hValue))
        im = cv2.copyMakeBorder(im, 0, h - hValue, 0, 0, borderType=cv2.BORDER_CONSTANT, value=pad_value)
    else:
        wValue = int(im.shape[1] * h / im.shape[0])
        im = cv2.resize(im, (wValue, int(h)))
        im = cv2.copyMakeBorder(im, 0, 0, 0, w - wValue, borderType=cv2.BORDER_CONSTANT, value=pad_value)
    return im, wValue, hValue


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is torch.nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is torch.nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def calculate_fid( images1, images2):
	# calculate activations
	# calculate mean and covariance statistics
	mu1, sigma1 = images1.mean(dim=1), torch.cov(images1)
	mu2, sigma2 = images2.mean(dim=1), torch.cov(images2)
	# calculate sum squared difference between means
	ssdiff = torch.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = torch.sqrt(torch.dot(sigma1, sigma2))
	# check and correct imaginary numbers from sqrt
	if torch.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# if __name__ == '__main__':
#     im1=torch.rand((2,3,4,5))
#     im2=torch.rand_like(im1)
#     fid = calculate_fid(im1,im2)
