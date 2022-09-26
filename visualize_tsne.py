from datasets.frame_dataset import get_client_local_loaders
from models.build import build_model
from evaluation.video_tsne import gen_tsne
import yaml 
from utils.parsing import Dict2Class
import argparse
import torch 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize t-SNE on each client")
    parser.add_argument(
        "--cid", type=int, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--work_dir",
        type=str, required=True,
        help="Working directory, used for saving logs, checkpoints etc.",
    )
    parser.add_argument(
        "--cfg_path",
        default='configs/afosr_movinetA0.yaml',
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="image, metadata directory",
    )
    args = parser.parse_args()
    client_id = args.cid

    # configurations 
    with open(args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    # datasets
    _, val_loader = get_client_local_loaders(client_id, 
                                        args.data_dir,
                                        args.work_dir,
                                        cfgs)

    # model
    model = build_model(cfgs, mode='train')
    ## load checkpoint by round number
    checkpoint = torch.load(args.work_dir + '/best.pth')
    model.load_state_dict(checkpoint['state_dict'])

    # Generate t-SNE visualization
    gen_tsne(model, val_loader, 
            cfgs.device, 
            args.work_dir + f'/client_{client_id}_tSNE.png')