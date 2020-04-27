import os
import json
import tables
import requests
import hdf5storage
import numpy as np
import scipy.io as sio

"""
Helpers for downloading and opening data files.
Google Drive download methods are from https://stackoverflow.com/a/39225272/4520764
`load_summe_mat` and `load_tvsum_mat` are from https://github.com/mayu-ot/rethinking-evs/
"""

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)

def load_summe_mat(dirname):
    mat_list = os.listdir(dirname)
    data_list = []
    for mat in mat_list:
        data = sio.loadmat(os.path.join(dirname, mat))
        item_dict = {
            'video': mat[:-4],
            'length': data['video_duration'],
            'nframes': data['nFrames'],
            'user_anno': data['user_score'],
            'gt_score': data['gt_score']
        }
        data_list.append((item_dict))
    return data_list

def load_tvsum_mat(filename):
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()
    data_list = []
    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item
        item_dict = {
            'video': video[0, 0],
            'category': category[0, 0],
            'title': title[0, 0],
            'length': length[0, 0],
            'nframes': nframes[0, 0],
            'user_anno': user_anno,
            'gt_score': gt_score
        }
        data_list.append((item_dict))
    return data_list