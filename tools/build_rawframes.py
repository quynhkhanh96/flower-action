import os 
import sys 
import mmcv 
import warnings

def extract_frames(video_path, out_path):

    try:
        vr = mmcv.VideoReader(video_path)
        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                mmcv.imwrite(vr_frame, f'{out_path}/img_{i}.jpg')
            else:
                warnings.warn(
                    'Length inconsistent!'
                    f'Early stop with {i + 1} out of {len(vr)} frames.'
                )
                break
        run_success = 0
    except Exception:
        run_success = -1

src_dir = sys.argv[1]
dst_dir = sys.argv[2]
os.makedirs(dst_dir, exist_ok=True)

for person_id in os.listdir(src_dir):
    os.makedirs(dst_dir + '/' + person_id, exist_ok=True)
    for timestamp in os.listdir(src_dir + '/' + person_id):
        os.makedirs(dst_dir + '/' + person_id + '/' + timestamp, exist_ok=True)

        videos = os.listdir(src_dir + '/' + person_id + '/' + timestamp)
        videos = [video for video in videos if video.endswith('.mp4')]
        for video in videos:

            video_path = src_dir + '/' + person_id + '/' + timestamp + '/' + video

            vid_name = video.split('.')[0]
            out_path = dst_dir + '/' + person_id + '/' + timestamp + '/' + vid_name
            os.makedirs(out_path, exist_ok=True)

            extract_frames(video_path, out_path)

    print('DONE 01 PERSON.')