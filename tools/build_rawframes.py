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
        help="Path to your dataset",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        help="Path to the preprocessed dataset with raw frames",
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
    parser.add_argument(
        "--no_person_ids",
        default=False,
        type=bool,
        help="whether there person ids are provided in your dataset",
    )
    args = parser.parse_args()

    src_dir, dst_dir = args.src_dir, args.dst_dir
    new_height, new_width = args.new_height, args.new_width
    os.makedirs(dst_dir, exist_ok=True)

    person2videos = defaultdict(list)
    if args.no_person_ids:
        # pseudo person id
        for idx, video_id in enumerate(os.listdir(src_dir + '/videos')):
            if video_id.endswith('.mp4'):
                person2videos[f'person_{idx}'].append(video_id)
    else:
        for person_id in os.listdir(src_dir + '/videos'):
            for video_id in os.listdir(src_dir + '/videos/' + person_id):
                if video_id.endswith('.mp4'):
                    person2videos[person_id].append(video_id)

    print('Total person ids =', len(person2videos))

    for i, (person_id, video_ids) in enumerate(person2videos.items()):
        os.makedirs(dst_dir + '/' + person_id, exist_ok=True)
        for video_id in video_ids:
            if args.no_person_ids:
                video_path = src_dir + '/videos/' + video_id
            else:
                video_path = src_dir + '/videos/' + person_id + '/' + video_id
            out_path = dst_dir + '/' + person_id + '/' + video_id.split('.')[0] 
            os.makedirs(out_path, exist_ok=True)
            extract_frames(video_path, out_path, new_height, new_width)

        print(f'DONE {i + 1} PEOPLE.')