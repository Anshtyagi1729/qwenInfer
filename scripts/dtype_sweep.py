import torch, time
d = torch.device("cuda")

def timed(fn, iters, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

print(f"{'size':>6} | {'fp32':>18} | {'fp16':>18} | {'bf16':>18}")
print("-" * 70)
for m in (512, 1024, 2048):
    row = f"{m:>6} |"
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        try:
            x = torch.randn(m, m, dtype=dt, device=d)
            y = torch.randn(m, m, dtype=dt, device=d)
            t = timed(lambda: x @ y, 20)
            row += f" {2*m**3/t/1e12:8.3f} TFLOP/s |"
            del x, y; torch.cuda.empty_cache()
        except Exception as e:
            row += f" {type(e).__name__:>16} |"
    print(row)

# Is it a cuBLAS kernel-selection issue, or the hardware's real fp16 rate?
# Elementwise work has no library kernel choice involved, so it isolates the ALU rate.
print("\nelementwise FMA chain (isolates ALU throughput, no cuBLAS):")
n = 1 << 24
for dt in (torch.float32, torch.float16):
    a = torch.randn(n, dtype=dt, device=d)
    b = torch.randn(n, dtype=dt, device=d)
    c = torch.randn(n, dtype=dt, device=d)
    t = timed(lambda: torch.addcmul(c, a, b), 50)
    print(f"  {str(dt):>16}: {n*2/t/1e12:7.3f} TFLOP/s   {(3*n*a.element_size())/t/1e9:6.1f} GB/s")
    del a, b, c; torch.cuda.empty_cache()
