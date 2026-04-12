"""Find blocks where epoch 5 MSE >= 1.2x epoch 4 MSE, across all models."""
import os, re, glob

base = "/gs/bs/tga-artic/k-kuroki/BQQ/neural_network_compression/lm/scripts/qsub_jobs"
THRESHOLD = 1.2

for model in ["2B", "4B", "9B"]:
    for bit in [2, 3]:
        logdir = os.path.join(base, f"blockwise-{model}-bit{bit}", "logs")
        if not os.path.isdir(logdir):
            continue

        # Collect latest log per block
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
                # Try alternate format (re-run scripts without BLOCK_IDX header)
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
                if e4 > 0 and e5 / e4 >= THRESHOLD:
                    layer_idx = i // 5
                    ratio = e5 / e4
                    print(f"{model} {bit}bit block_{bidx:2d} linear_{layer_idx}: "
                          f"e4={e4:.6f} e5={e5:.6f} x{ratio:.1f}")
                    has_regression = True
