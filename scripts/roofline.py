import torch, time

print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no cuda")
d = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**20:.0f} MiB  {p.multi_processor_count} SMs")
free, total = torch.cuda.mem_get_info()
print(f"free {free/2**20:.0f} MiB / total {total/2**20:.0f} MiB")

def timed(fn, iters, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# --- memory bandwidth: large contiguous copy (reads N bytes, writes N bytes) ---
n = 48 * 2**20 // 2          # 48 MiB of fp16 elements
a = torch.randn(n, dtype=torch.float16, device=d)
b = torch.empty_like(a)
t = timed(lambda: b.copy_(a), 50)
bw = 2 * a.numel() * a.element_size() / t / 1e9
print(f"\ncopy bandwidth : {bw:6.1f} GB/s   ({t*1e6:.0f} us for {a.numel()*2/2**20:.0f} MiB)")

# --- compute: square matmul, big enough to be compute bound ---
for dt, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
    m = 2048
    x = torch.randn(m, m, dtype=dt, device=d)
    y = torch.randn(m, m, dtype=dt, device=d)
    t = timed(lambda: x @ y, 20)
    tflops = 2 * m**3 / t / 1e12
    print(f"matmul {name}   : {tflops:6.2f} TFLOP/s ({m}x{m})")
    del x, y
    torch.cuda.empty_cache()
