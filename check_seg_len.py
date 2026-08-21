import json
import glob
import os

jsons = glob.glob('input/Seq1/imageSequence/Segments/video_0_seg_*.json')
jsons.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

for j in jsons:
    with open(j) as f:
        d = json.load(f)
    print(f"{os.path.basename(j)}: {d[0]['frame_id']} -> {d[-1]['frame_id']} ({len(d)} frames)")
