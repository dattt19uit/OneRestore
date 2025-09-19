import json


def get_dynamic_text(mode, degradation, filename):
    with open(f'labels/{mode}.json', 'r') as f:
        data = json.load(f)
    return data[degradation][filename]


with open(f'labels/test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    idx = 0
    indexs = {}
    for k, v in data.items():
        deg = {}
        for i, j in v.items():
            print(k, i, idx)
            deg[i] = idx
            idx += 1
        indexs[k] = deg

    print(indexs)

    with open('labels/test_index.json', 'w', encoding='utf-8') as f:
        json.dump(indexs, f, ensure_ascii=False, indent=4)
