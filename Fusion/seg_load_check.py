"""
seg_load_check.py - does the SegFormer checkpoint really load in THIS environment?


    python seg_load_check.py "C:\\...\\Detection_YOLO\\runs\\seg\\best"


transformers 5.x renamed SegFormer's internal layers (encoder.block -> stages.blocks,
attention.self.query -> attention.q_proj, ...). from_pretrained() converts old checkpoints
automatically, so comparing the FILE's key names with model.state_dict() names reports
~200 "missing" keys even when every weight loaded. The only reliable test is the loading
report from from_pretrained itself, which this script prints.
"""
import json, os, sys
import transformers
from transformers import SegformerForSemanticSegmentation


path = sys.argv[1] if len(sys.argv) > 1 else r'..\Detection_YOLO\runs\seg\best'
saved_with = json.load(open(os.path.join(path, 'config.json'))).get('transformers_version', '?')
model, info = SegformerForSemanticSegmentation.from_pretrained(path, output_loading_info=True)
n = {k: len(info.get(k, [])) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')}


print(f'\ncheckpoint saved with transformers {saved_with}; this environment has {transformers.__version__}')
print(f'tensors in model: {len(model.state_dict())}   loading report: {n}')
if sum(n.values()) == 0:
    print('OK: every weight loaded.')
    if saved_with.split('.')[0] != transformers.__version__.split('.')[0]:
        print('    (saved with a different major version: from_pretrained converted the old layer names.)')
        print('    A "does not match" message from your own loader is a false alarm from comparing key')
        print('    NAMES; check the from_pretrained loading report instead (load_segformer).')
else:
    for k in n:
        for key in info.get(k, [])[:5]:
            print(f'  {k}: {key}')
    print(f'NOT OK: install the version the checkpoint was saved with:  pip install "transformers=={saved_with}"')