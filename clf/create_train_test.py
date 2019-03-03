import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

def process_frames(video):
    cap = cv2.VideoCapture(video)
    frames = []
    cnt = 0 # keep track of frames to avoid capturing video intro's
    max_frames = 2000 # read only part of the video
    n_skip = 500 # number of frames to skip at beginning
    n_samples = 300 # number of sampled images

    while len(frames)<max_frames and cap.isOpened():
        ret, frame = cap.read()        
        if ret == True:
            cnt += 1
            if cnt>n_skip:
                frames.append(frame)
        else:
            break
    cap.release()

    # Choose only a random sample of captured frames
    if len(frames)>n_samples:
        sample_idxs = np.random.choice(len(frames), size=n_samples, replace=False)
        sample_frames = [frames[ix] for ix in sample_idxs]
    else:
        print("WARNING: video with less than 300 frames")
        sample_frames = []
    return sample_frames

def process_videos(videos_path,imgs_path):   
    videos = glob.glob(videos_path+'/*')
    for video in tqdm(videos):
        video_name = video.split('\\')[-1]
        print("Processing video: "+video_name)
        sample_frames = process_frames(video)
        for i, sample in enumerate(sample_frames):
            fname = imgs_path+'/'+video_name+str(i)+'.png'
            sample = cv2.resize(src=sample, dsize=(100,100))
            cv2.imwrite(fname, sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select video files to parse")
    parser.add_argument('videos_path', help="Path for source videos")
    parser.add_argument('imgs_path', help="Path to store images")
    args = parser.parse_args()
    process_videos(args.videos_path,args.imgs_path)
    