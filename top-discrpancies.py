#!/root/ImageEvaluator/.venv/bin/python
import csv, numpy as np
files = [
    'laion_aes.csv',
    'maniqa_scaled_raw.csv',
    'musiq-ava.csv',
    'musiq-paq2piq.csv',
    'clipiqa+_vitL14_512_shift_14_gurushots.csv'
]

def load_scores(path):
    data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = row.get('pyiqa_scaled') or row.get('overall_score') or row.get('score')
            if not score:
                continue
            try:
                data[row.get('file_path', path)] = float(score)
            except ValueError:
                pass
    return data
TOP_K = 25
datasets = {path: load_scores(path) for path in files}
rows = []
for i, a in enumerate(files):
    for b in files[i+1:]:
        common = sorted(set(datasets[a]) & set(datasets[b]))
        if not common:
            continue
        va = np.array([datasets[a][k] for k in common])
        vb = np.array([datasets[b][k] for k in common])
        diff = va - vb
        idx = np.argsort(np.abs(diff))[-TOP_K:][::-1]
        for j in idx:
            rows.append({'dataset_a': a, 'dataset_b': b,
                         'file_path': common[j],
                         'score_a': va[j], 'score_b': vb[j],
                         'difference_a_minus_b': diff[j]})
with open('pyiqa_diff_samples.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print('Wrote pyiqa_diff_samples.csv')
