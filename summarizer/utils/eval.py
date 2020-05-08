import os
import sys
import math
import numpy as np
from scipy import stats
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarizer.utils.knapsack import knapsack_ortools

"""
From original implementations of Kaiyang Zhou and Jiri Fajtl
https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
https://github.com/ok1zjf/VASNet
"""

def upsample(scores, n_frames, positions):
    """Upsample scores vector to the original number of frames.
    Input
      scores: (n_steps,)
      n_frames: (1,)
      positions: (n_steps, 1)
    Output
      frame_scores: (n_frames,)
    """
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = scores[i]
    return frame_scores

def generate_scores(probs, n_frames, positions):
    """Set score to every original frame of the video for comparison with annotations.
    Input
      probs: (n_steps,)
      n_frames: (1,)
      positions: (n_steps, 1)
    Output
      machine_scores: (n_frames,)
    """
    machine_scores = upsample(probs, n_frames, positions)
    return machine_scores

def evaluate_scores(machine_scores, user_scores, metric="spearmanr"):
    """Compare machine scores with user scores (keyframe-based).
    Input
      machine_scores: (n_frames,)
      user_scores: (n_users, n_frames)
    Output
      avg_corr, max_corr: (1,)
    """
    n_users, _ = user_scores.shape

    # Ranking correlation metrics
    if metric == "kendalltau":
        f = lambda x, y: stats.kendalltau(stats.rankdata(-x), stats.rankdata(-y))[0]
    elif metric == "spearmanr":
        f = lambda x, y: stats.spearmanr(stats.rankdata(-x), stats.rankdata(-y))[0]
    else:
        raise KeyError(f"Unknown metric {metric}")

    # Compute correlation with each annotator
    corrs = [f(machine_scores, user_scores[i]) for i in range(n_users)]
    
    # Mean over all annotators
    avg_corr = np.mean(corrs)
    return avg_corr

def generate_summary(scores, cps, n_frames, nfps, positions, proportion=0.15, method="knapsack"):
    """Generate keyshot-based video summary i.e. a binary vector.
    Input
      scores: predicted importance scores
      cps: change points, 2D matrix, each row contains a segment
      n_frames: original number of frames
      nfps: number of frames per segment
      positions: positions of subsampled frames in the original video
      proportion: length of video summary (compared to original video length)
      method: defines how shots are selected, ['knapsack', 'rank']
    Output
      summary: binary vector of shape (n_frames,)
    """
    n_segs = cps.shape[0]
    frame_scores = upsample(scores, n_frames, positions)

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError(f"Unknown method {method}")

    # This first element should be deleted
    summary = np.zeros((1), dtype=np.float32)
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    # Delete the first element
    summary = np.delete(summary, 0)
    return summary

def evaluate_summary(machine_summary, user_summary):
    """Compare machine summary with user summary (keyshot-based).
    Input
      machine_summary: (n_frames,)
      user_summary: (n_users, n_frames)
    Output
      avg_f_score, max_f_score: (1,)
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users, n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx, :]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    avg_f_score = np.mean(f_scores)
    max_f_score = np.max(f_scores)
    return avg_f_score, max_f_score