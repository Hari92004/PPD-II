import json
import os

meta_dir = r"D:\PPD2\dataset mini\v1.0-mini\v1.0-mini"

with open(os.path.join(meta_dir, "scene.json")) as f:
    scenes = json.load(f)
with open(os.path.join(meta_dir, "sample.json")) as f:
    samples = json.load(f)

samples_by_token = {s["token"]: s for s in samples}

print("=" * 80)
print(f"Total Scenes: {len(scenes)} | Total Keyframe Samples: {len(samples)}")
print("=" * 80)

for i, sc in enumerate(scenes):
    first_token = sc["first_sample_token"]
    last_token = sc["last_sample_token"]
    desc = sc["description"]
    
    # Traverse sequential chain using 'next'
    curr = first_token
    chain = []
    timestamps = []
    while curr:
        s = samples_by_token.get(curr)
        if not s:
            break
        chain.append(curr)
        timestamps.append(s["timestamp"] / 1e6) # in seconds
        curr = s["next"]
        
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    dt_list = [timestamps[j+1] - timestamps[j] for j in range(len(timestamps)-1)]
    avg_dt = sum(dt_list)/len(dt_list) if dt_list else 0
    
    print(f"Scene {i+1:2d} | {sc['name']:14s} | {len(chain):2d} frames | Duration: {duration:4.1f}s | Sample Rate: {1/avg_dt:3.1f} Hz (every {avg_dt:.2f}s)")
    print(f"         Context: {desc}")

print("=" * 80)
