# Project Rules

## CUDA Kernel Fixes And Optimizations

- When the task is to fix correctness, fix performance, or integrate an optimized CUDA kernel, do not replace the requested optimized path with a fallback to a reference kernel and present that as the completed fix. Reference kernels may be used only as temporary debug or validation oracles; the final change must either fix the optimized kernel/path itself or explicitly report that the optimized path remains unresolved.
