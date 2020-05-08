import os.path as osp
import argparse
import h5py
import cv2
from tqdm import tqdm

"""
Generate a video summary from the predictions and the frames.
"""

def frm2video(frm_dir, summary, vid_writer):
    for idx, val in tqdm(enumerate(summary), total=len(summary), ncols=80):
        if val == 1:
            # here frame name starts with "000001.jpg"
            frm_name = str(idx+1).zfill(6) + ".jpg"
            frm_path = osp.join(frm_dir, frm_name)
            frm = cv2.imread(frm_path)
            frm = cv2.resize(frm, (args.width, args.height))
            vid_writer.write(frm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to hdfs5 predictions file")
    parser.add_argument("-f", "--frames", type=str, required=True, help="Path to frame directory")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset hdfs5 filename")
    parser.add_argument("-v", "--video", type=str, help="Which video key to choose")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--width", type=int, default=640, help="frame width")
    parser.add_argument("--height", type=int, default=480, help="frame height")
    args = parser.parse_args()

    summary_file = f"summary_{args.video}.mp4"
    summary_path = osp.join(osp.dirname(args.path), summary_file)
    vid_writer = cv2.VideoWriter(
        summary_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height)
    )
    h5_preds = h5py.File(args.path, "r")
    dataset = h5_preds[args.dataset]
    summary = dataset[args.video]["machine_summary"][...]
    h5_preds.close()
    frm2video(args.frames, summary, vid_writer)
    vid_writer.release()
    print(f"Summary saved at {summary_path}")
