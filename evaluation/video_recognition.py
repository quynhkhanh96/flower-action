import torch 
import mmcv 
from tqdm import tqdm 
from mmcv import Config
from mmaction.core.evaluation import top_k_accuracy
from torch.autograd import Variable

def evaluate_topk_accuracy(model, test_loader, device):
    model.to(device)
    model.eval()

    results = []
    labels = []
    for data in test_loader:
        if isinstance(data, dict):
            imgs, label = data['imgs'], data['label']
        else:
            imgs, label, _ = data

        imgs = imgs.to(device)

        with torch.no_grad():
            imgs = Variable(imgs)
            outputs = model(imgs)	

        results.extend(outputs.cpu().numpy())
        labels.extend(label)

    top1_acc, top5_acc = top_k_accuracy(results, labels, topk=(1, 5))
    return {'top1': top1_acc, 'top5': top5_acc}

