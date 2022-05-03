import os 
import sys 

src_dir = sys.argv[1]
dst_dir = sys.argv[2]
size = sys.argv[3]

for person_id in os.listdir(src_dir):
    os.makedirs(dst_dir + '/' + person_id, exist_ok=True)
    for timestamp in os.listdir(src_dir + '/' + person_id):
        os.makedirs(dst_dir + '/' + person_id + '/' + timestamp, exist_ok=True)

        videos = os.listdir(src_dir + '/' + person_id + '/' + timestamp)
        videos = [video for video in videos if video.endswith('.mp4')]
        for video in videos:
            cmd = 'ffmpeg -i {} -threads 4 -vf "scale={}:{}" {}'.format(
                src_dir + '/' + person_id + '/' + timestamp + '/' + video,
                size, size,
                dst_dir + '/' + person_id + '/' + timestamp + '/' + video
            )
            os.system(cmd)

    print('DONE 01 PERSON.')
    