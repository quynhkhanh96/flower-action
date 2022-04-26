import torch 
import mmcv 
from tqdm import tqdm 
from mmcv import Config
from mmaction.core.evaluation import top_k_accuracy
from mmaction.datasets import build_dataloader
from torch.autograd import Variable

def evaluate_video_recognizer(model, test_dataset, device):
    dataloader_setting = dict(
        videos_per_gpu=8,
        workers_per_gpu=2,
        persistent_workers=False,
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        shuffle=False)

    data_loader = build_dataloader(test_dataset, **dataloader_setting)

    model.eval()
    results = []
    labels = []

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            data['imgs'] = data['imgs'].to(device)
            result = model(return_loss=False, **data)
        results.extend(result)
        labels.extend(data['label'].numpy())

        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()

    top1_acc, top5_acc = top_k_accuracy(results, labels, topk=(1, 5))
    return {'top1': top1_acc, 'top5': top5_acc}

def evaluate_topk_accuracy(model, test_loader, device):
    model.to(device)
    model.eval()

    results = []
    labels = []
    for imgs, label, video_paths in test_loader:
        imgs = imgs.to(device)

        with torch.no_grad():
            imgs = Variable(imgs)
            outputs = model(imgs)	

        results.extend(outputs)
        labels.extend(label)

    top1_acc, top5_acc = top_k_accuracy(results, labels, topk=(1, 5))
    return {'top1': top1_acc, 'top5': top5_acc}
