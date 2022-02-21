import torch.nn.functional as F
import torch 

def test_classifer(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        probs = model(data)
        test_loss += F.cross_entropy(probs, target, reduction='sum').item()
        preds = probs.data.max(1, keepdim=True)[1]
        correct += preds.eq(target.data.view_as(preds)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.00 * correct / len(test_loader.dataset)

    return test_loss, {'accuracy': test_accuracy} 
