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

datasets = {path: load_scores(path) for path in files}
rows = []
for i, a in enumerate(files):
    for b in files[i+1:]:
        common = sorted(set(datasets[a]) & set(datasets[b]))
        if not common:
            continue
        va = np.array([datasets[a][k] for k in common])
        vb = np.array([datasets[b][k] for k in common])
        corr = float(np.corrcoef(va, vb)[0, 1]) if len(common) > 1 else float('nan')
        rows.append({'dataset_a': a, 'dataset_b': b,
                     'common_images': len(common),
                     'pearson_corr': corr,
                     'mean_a': float(va.mean()), 'std_a': float(va.std()),
                     'mean_b': float(vb.mean()), 'std_b': float(vb.std())})

with open('pyiqa_comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print('Wrote pyiqa_comparison.csv')

