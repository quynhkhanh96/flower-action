import torch 
from tqdm import tqdm 
from mmcv import Config
from mmaction.core.evaluation import top_k_accuracy
# from mmaction.datasets import build_dataset
# from mmaction.models import build_recognizer
# from mmcv.runner import load_checkpoint

def evaluate_video_recognizer(model, test_dataset, device):
    # TODO: validation loss 
    labels = []
    scores = []
    num_samples = len(test_dataset)
    with torch.no_grad():
        for i in tqdm(range(num_samples), total=num_samples):
            data = {'imgs': test_dataset[i]['imgs'][None].to(device)}
            score = model(return_loss=False, **data)[0]
            scores.append(score)
            labels.append(test_dataset[i]['label'])

    top1_acc, top5_acc = top_k_accuracy(scores, labels, topk=(1, 5))

    return {'top1': top1_acc, 'top5': top5_acc}