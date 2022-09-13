import os 
import mmcv 
import warnings
import argparse
from collections import defaultdict

def extract_frames(video_path, out_path, new_height, new_width):
    try:
        vr = mmcv.VideoReader(video_path)
        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                if new_width == 0 or new_height == 0:
                    # Keep original shape
                    out_img = vr_frame 
                else:
                    out_img = mmcv.imresize(vr_frame, (new_width, new_height))
                mmcv.imwrite(out_img, f'{out_path}/img_{i}.jpg')
            else:
                warnings.warn(
                    'Length inconsistent!'
                    f'Early stop with {i + 1} out of {len(vr)} frames.'
                )
                break
        run_success = 0
    except Exception:
        run_success = -1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Build rawframe dataset")
    parser.add_argument(
        "--src_dir",
        type=str,
        help="Path to your original dataset",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Path to the preprocessed dataset",
    )
    parser.add_argument(
        "--new_height",
        default=224,
        type=int,
        help="Frame height",
    )
    parser.add_argument(
        "--new_width",
        default=224,
        type=int,
        help="Frame width",
    )
    args = parser.parse_args()

    src_dir, dst_dir = args.src_dir, args.dst_dir
    new_height, new_width = args.new_height, args.new_width
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(dst_dir + '/rgb_frames', exist_ok=True)

    with open(src_dir + '/train.txt', 'r') as f:
        train_person_ids = set([l.strip() for l in f.readlines()])
    with open(src_dir + '/val.txt', 'r') as f:
        val_person_ids = set([l.strip() for l in f.readlines()])    

    video_paths = []
    for person_id in os.listdir(src_dir + '/data'):
        for date_id in os.listdir(src_dir + '/data/' + person_id):
            for fname in os.listdir(src_dir + '/data/' + person_id + '/' + date_id):
                if fname.endswith('.mp4'):
                    video_paths.append(src_dir + '/data/' + person_id + '/' + date_id + '/' + fname)

    train_info, val_info = [], []
    n_videos = len(video_paths)
    print(f'There are {n_videos} videos in the dataset.')
    for i, video_path in enumerate(video_paths):
        out_path = dst_dir + f'/rgb_frames/video_{i}'
        os.makedirs(out_path, exist_ok=True)
        extract_frames(video_path, out_path, new_height, new_width)
        # The original label is 1-indexed
        label = int(video.split('/')[-1].split('.')[0]) - 1

        person_id = video_path.split('/')[-3] 
        if person_id in train_person_ids:
            train_info.append([f'video_{i}', label])
        elif person_id in val_person_ids:
            val_info.append([f'video_{i}', label])
        
        if i+1 % 10 == 0:
            print(f'DONE {i+1} videos.')

    with open(dst_dir + '/train.txt', 'a') as f:
        for video_id, label in train_info:
            f.write(f'{video_id} {label}')
    with open(dst_dir + '/val.txt', 'a') as f:
        for video_id, label in val_info:
            f.write(f'{video_id} {label}')