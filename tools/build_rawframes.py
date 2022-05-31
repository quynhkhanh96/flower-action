import os 
import sys 
import mmcv 
import warnings

def extract_frames(video_path, out_path, new_height, new_width):

    try:
        vr = mmcv.VideoReader(video_path)
        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                # w, h, _ = np.shape(vr_frame)
                # if args.new_short == 0:
                #     if new_width == 0 or new_height == 0:
                #         # Keep original shape
                #         out_img = vr_frame
                #     else:
                #         out_img = mmcv.imresize(
                #             vr_frame,
                #             (new_width, new_height))
                # else:
                #     if min(h, w) == h:
                #         new_h = args.new_short
                #         new_w = int((new_h / h) * w)
                #     else:
                #         new_w = args.new_short
                #         new_h = int((new_w / w) * h)
                #     out_img = mmcv.imresize(vr_frame, (new_h, new_w))
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

src_dir = sys.argv[1]
dst_dir = sys.argv[2]
try:
    new_height, new_width = sys.argv[3], sys.argv[3]
except:
    new_height, new_width = 0, 0
os.makedirs(dst_dir, exist_ok=True)

person_ids = os.listdir(src_dir)
print('Total person ids =', len(person_ids))
for iter_, person_id in enumerate(person_ids):
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

    print(f'DONE {iter_ + 1} PEOPLE.')