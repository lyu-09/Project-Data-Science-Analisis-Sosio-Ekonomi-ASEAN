import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('analisis_worldbank_asean_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('analisis_worldbank_asean.ipynb', 'r', encoding='utf-8') as f:
    nb_orig = json.load(f)

orig_codes = [(i,c) for i,c in enumerate(nb_orig['cells']) if c['cell_type']=='code']
new_codes  = [(i,c) for i,c in enumerate(nb['cells'])      if c['cell_type']=='code']

all_match = True
for (io, co), (in_, cn) in zip(orig_codes, new_codes):
    if ''.join(co['source']) != ''.join(cn['source']):
        print(f'MISMATCH at code cell index {io}')
        all_match = False

if all_match:
    print(f'VERIFIKASI OK: Semua {len(orig_codes)} code cells identik dengan asli.')

md_orig = [c for c in nb_orig['cells'] if c['cell_type']=='markdown']
md_new  = [c for c in nb['cells']      if c['cell_type']=='markdown']
print(f'Markdown cells asli : {len(md_orig)}')
print(f'Markdown cells baru : {len(md_new)}')
print(f'Total cells notebook: {len(nb["cells"])}')
