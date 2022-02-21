from include import * 
import data_utils

class Configs:
    ada_thresh = True
    T_thresh = 0.05 # fixed threshold
    local_e = 5 # local epoch
    epochs = 25 # communication rounds
    device = 'cuda' # GPU or not
    batch_size = 64 # local batch size
    dataset = 'cifar10' 
    model = 'CNN'
    optimizer = 'Adam'
    lr = 0.001 # learning rate 
    frac = 1 # partipation ratio
    num_C = 5 # client number
    min_sample_size = 2
    min_num_clients = 2 
    Nc = 10 # number of classes on clients
    iid = True 
    seed = 1234 

Args = Configs 
C_iter, train_iter, test_iter, stats = data_utils.get_dataset(args=Args)
G_loss_fun = torch.nn.CrossEntropyLoss()

DEFAULT_SERVER_ADDRESS = "[::]:8080"

if Args.model == 'MLP':
    from models.MLP import MLP as Fed_Model
elif Args.model == 'CNN':
    from models.CNN import CNN as Fed_Model
elif Args.model == 'ResNet':
    from models.resnet import ResNet18 as Fed_Model