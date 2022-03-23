import os 
import pickle 
import shutil 
import sys 
from mmaction.datasets import build_dataset

def data_partition(n_clients, preprocessed_dir, fold):
    """
    Args:
        preprocessed_dir (str): path to directory containing HMDB51's 
                                `rawframes`, `videos`, split files (.txt)
    """
    os.makedirs(preprocessed_dir + f'/partition', exist_ok=True)
    dataset_name = os.path.basename(preprocessed_dir)

    all_actions = {action: [] 
                    for action in os.listdir(preprocessed_dir + '/rawframes')}
    with open(preprocessed_dir + f'/{dataset_name}_train_split_{fold}_rawframes.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        action, _ = line.split('/')
        all_actions[action].append(line)
    for action in all_actions:
        all_actions[action] = [all_actions[action][i::n_clients] for i in range(n_clients)]

    for client_id in range(n_clients):
        client_files = []
        for action in all_actions:
            client_files.extend(all_actions[action][client_id])
        metadata_path = '{}/partition/{}_train_split_{}_client_{}_rawframes.txt'.format(
            preprocessed_dir, dataset_name, fold, client_id
        )
        with open(metadata_path, 'a') as f:
            for client_file in client_files:
                f.write(client_file)

    shutil.copy(preprocessed_dir + f'/{dataset_name}_val_split_{fold}_rawframes.txt',
                preprocessed_dir + f'/partition/{dataset_name}_val_split_{fold}_rawframes.txt')
    
    with open(preprocessed_dir + '/partition/clients.pkl', 'wb') as handle:
        pickle.dump(all_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_client_dataset(client_id, fold, dataset_root, cfg):

    dataset_name = os.path.basename(dataset_root)

    data_root = dataset_root + '/rawframes' 
    ann_file_train = dataset_root + '/partition/{}_train_split_{}_client_{}_rawframes.txt'.format(
        dataset_name, fold, client_id
    )
    ann_file_val = dataset_root + '/partition/{}_val_split_{}_rawframes.txt'.format(
        dataset_name, fold
    )

    data_train_cfg = dict(
        type=cfg.data.dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=cfg.data.train.pipeline
    )
    data_val_cfg = dict(
        type=cfg.data.dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=cfg.data.val.pipeline
    )

    train_dataset = build_dataset(data_train_cfg)
    val_dataset = build_dataset(data_val_cfg)

    return train_dataset, val_dataset

if __name__ == '__main__':

    preprocessed_dir = sys.argv[1]
    n_clients = int(sys.argv[2])
    try:
        fold = int(sys.argv[3])
    except:
        fold = 1
    data_partition(n_clients=n_clients, 
                    preprocessed_dir=preprocessed_dir, fold=fold)


    