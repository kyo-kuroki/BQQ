"""Find unique blocks where any linear layer has epoch5 MSE > epoch4 MSE.
Output: model, bit, block_idx — one line per block."""
import os, re, glob

base = "/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm/scripts/qsub_jobs"

# Already re-run with grad clip (latest code)
already_rerun = {
    ("4B", 3): {3, 4, 11, 15, 27},
    ("2B", 3): {11},
}

results = []

for model in ["2B", "4B", "9B"]:
    for bit in [2, 3]:
        logdir = os.path.join(base, f"blockwise-{model}-bit{bit}", "logs")
        if not os.path.isdir(logdir):
            continue

        block_logs = {}
        for f in sorted(glob.glob(os.path.join(logdir, "*.o*.*"))):
            bn = os.path.basename(f)
            parts = bn.rsplit(".", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            try:
                text = open(f).read()
            except:
                continue
            if "Final MSE" not in text:
                continue
            m = re.search(r"BLOCK_IDX\s*:\s*(\d+)", text)
            if not m:
                m2 = re.search(r"block_(\d+)\.pth", text)
                if not m2:
                    continue
                bidx = int(m2.group(1))
            else:
                bidx = int(m.group(1))
            job_m = re.search(r"\.o(\d+)\.", bn)
            job_id = int(job_m.group(1)) if job_m else 0
            if bidx not in block_logs or job_id > block_logs[bidx][0]:
                block_logs[bidx] = (job_id, f, text)

        for bidx in sorted(block_logs.keys()):
            job_id, _, text = block_logs[bidx]
            epochs = re.findall(r"Epoch (\d+)/5: MSE=([0-9.]+)", text)
            has_regression = False
            for i in range(0, len(epochs), 5):
                chunk = epochs[i:i+5]
                if len(chunk) < 5:
                    continue
                e4 = float(chunk[3][1])
                e5 = float(chunk[4][1])
                if e5 > e4:
                    has_regression = True
                    break
            if has_regression:
                skip = already_rerun.get((model, bit), set())
                status = "SKIP(already rerun)" if bidx in skip else "RERUN"
                results.append((model, bit, bidx, status))
                print(f"{model} {bit}bit block_{bidx:2d}  {status}")

print(f"\nTotal blocks to rerun: {sum(1 for r in results if r[3] == 'RERUN')}")
print("Blocks to rerun per model:")
rerun_map = {}
for model, bit, bidx, status in results:
    if status == "RERUN":
        key = f"{model}-{bit}bit"
        rerun_map.setdefault(key, []).append(bidx)
for key, blocks in sorted(rerun_map.items()):
    print(f"  {key}: {blocks}")
