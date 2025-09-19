import json


def get_dynamic_label(mode, degradation, filename):
    with open(f'labels/{mode}.json', 'r') as f:
        data = json.load(f)
    return data[degradation][filename.split('.')[0]]

def get_dynamic_idx(mode, degradation, filename):
    with open(f'labels/{mode}_index.json', 'r') as f:
        data = json.load(f)
    return data[degradation][filename.split('.')[0]]

