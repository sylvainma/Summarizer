import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from summarizer.utils.CNN import FeatureExtractor
from summarizer.utils.KTS.cpd_auto import cpd_auto


class Generate_Dataset:
    def __init__(self, args):
        assert os.path.isfile('utils/KTS/cpd_nonlin.c'), "Error: Cython file utils/KTS/cpd_nonlin.pyx hasn't been built. Please refer to ./utils/KTS/README.md#build for instructions."

        self.args = args

        self.extractor = FeatureExtractor(self.args.extractor, self.args.layer_limit)
        self._set_video_list(self.args.video)

        assert os.path.isdir(self.args.annotations), "Error: --annotations path is not a valid directory."

    def _set_video_list(self, video_path):
        assert os.path.isdir(video_path), "Error: --video path is not a valid directory."

        self.video_path = video_path
        video_list = os.listdir(video_path)
        self.video_list = [l for l in video_list if l.endswith(self.args.video_format)]
        self.video_list.sort()

    def _extract_feature(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        out_pool5 = self.extractor(frame)
        frame_feat = out_pool5.cpu().data.numpy().flatten()

        return frame_feat

    def _get_change_points(self, video_feat, n_frame, fps):
        n = n_frame / fps
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0], change_points, [n_frame-1]))

        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]

            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))

        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = change_points[change_points_idx][1] - change_points[change_points_idx][0]
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        return change_points, n_frame_per_seg

    def release_dataset(self):
        self.h5_file.close()

    def generate_dataset(self):

        h5_basedir = os.path.dirname(self.args.h5)
        if h5_basedir:
            os.makedirs(h5_basedir, exist_ok=True)
        self.h5_file = h5py.File(self.args.h5, 'w' if self.args.overwrite else 'x')

        for video_idx, video_filename in enumerate(tqdm(self.video_list, desc="Videos")):

            video_path = os.path.join(self.args.video, video_filename)
            video_basename = '.'.join(os.path.basename(video_path).split('.')[:-1])

            video_capture = cv2.VideoCapture(video_path)

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            video_scores = np.load(os.path.join(self.args.annotations, video_basename + ".npy"))
            """
            NOTE:
            No other processing on video_scores is performed, assuming our annotations are binary (0/1)
            If your dataset holds more precise information (e.g. actual scores in [0, 1]), you may want to
            load these labels as well, and process user_summary and user_scores differently (below)
            See ./datasets/README.md#documentation for more information on the use of each key of the HDF5 files
            """ 

            # Rectifying shape of annotations
            if video_scores.ndim == 1:
                video_scores = video_scores.reshape((1, len(video_scores)))

            # Rectifying for discrepancies between labels and video
            if len(video_scores) < n_frames:
                video_scores = np.pad(video_scores, (0, n_frames - len(video_scores)), 'constant')

            scores = []
            picks = []
            video_feat = []
            video_feat_for_train = []
            for frame_idx in tqdm(range(n_frames-1), desc="Frames"):
                success, frame = video_capture.read()
                if success:
                    frame_feat = self._extract_feature(frame)

                    if frame_idx % self.args.keyshot_sampling == 0:
                        picks.append(frame_idx)
                        scores.append(video_scores[frame_idx])
                        video_feat_for_train.append(frame_feat)

                    video_feat.append(frame_feat)

                else:
                    print(f"Could not open file: {video_path}")

            video_capture.release()

            video_feat_for_train = np.vstack(video_feat_for_train)
            video_feat = np.vstack(video_feat)
            scores = np.array(list(scores))
            video_scores = video_scores[:n_frames]

            # Computing change points
            if self.args.changepoint_method == "kts":
                change_points, n_frame_per_seg = self._get_change_points(video_feat, n_frames, fps)
            elif self.args.changepoint_method == "uniform":
                segment_limits = picks[::self.args.changepoint_duration][:-1]
                change_points = np.vstack((segment_limits, np.append(picks[::self.args.changepoint_duration][1:len(segment_limits)], [picks[-1]]))).transpose()
                n_frame_per_seg = change_points[:, 1] - change_points[:, 0]

            # Writing to h5
            self.h5_file.create_group('video_{}'.format(video_idx+1))
            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks)).astype(np.int32)
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points.astype(np.int32)
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg.astype(np.int32)
            self.h5_file['video_{}'.format(video_idx+1)]['video_name'] = video_filename
            self.h5_file['video_{}'.format(video_idx+1)]['gtscore'] = scores.astype(np.float32)
            self.h5_file['video_{}'.format(video_idx+1)]['gtsummary'] = scores.astype(np.int32)
            self.h5_file['video_{}'.format(video_idx+1)]['user_summary'] = video_scores.astype(np.int32)
            self.h5_file['video_{}'.format(video_idx+1)]['user_scores'] = video_scores.astype(np.float32)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=True, help="Path to the videos directory")
    parser.add_argument("-a", "--annotations", type=str, required=True, help="Path to the corresponding annotations directory")
    parser.add_argument("-5", "--h5", type=str, default="datasets/summarizer_dataset_mynewdataset_google_pool5.h5", help="Name and path of the HDF5 file to create")
    parser.add_argument("-o", "--overwrite", action='store_true', help="Overwrite HDF5 file if it already exists")
    parser.add_argument("-m", "--changepoint-method", choices=["kts", "uniform"], default="uniform", help="Method to use when determining changepoints (kernel temporal segmentation or uniform segmentation)")
    parser.add_argument("-d", "--changepoint-duration", type=int, default=4, help="For uniform segmentation, number of keyshots per segment")
    parser.add_argument("-s", "--keyshot-sampling", type=int, default=15, help="Sampling rate (nth frame) for feature extraction on keyshots")
    parser.add_argument("-f", "--video-format", type=str, default=".mp4", help="File extension of video files")
    parser.add_argument("-e", "--extractor", type=str, default="googlenet", help="Feature extractor from torchvision.models")
    parser.add_argument("-l", "--layer-limit", type=int, default=-3, help="Index of stop layer for feature extraction")    
    args = parser.parse_args()

    gen = Generate_Dataset(args)
    gen.generate_dataset()
    gen.release_dataset()
