# Vulkan GPU-resident PCG State Design

## Goal

Reduce Vulkan PCG synchronization overhead by keeping scalar recurrence values and convergence state on the device, while preserving the existing `IterativeSolverReport` semantics and numerical behavior.

## Context

The current Vulkan solver executes the CSR SpMV and vector kernels on the GPU, but each dot product copies one scalar result to a host-visible staging buffer and waits on a fence. Those waits make the Vulkan path slower than the NVIDIA OpenCL path even after device-local matrix/vector storage and command-buffer reuse improvements.

`IterativeSolverOptions::convergenceCheckInterval` already describes the desired batching boundary. The Vulkan implementation currently performs a host-visible scalar readback for every dot product regardless of that interval.

## Proposed design

Add a small device-local state buffer for the PCG scalar values:

- `rho`: current `r^T z` value.
- `alpha`: current step size.
- `beta`: next direction coefficient.
- `residualSquared`: latest checked residual norm squared.
- `toleranceSquared`: convergence threshold derived on the device.
- `initialResidualSquared`: immutable residual norm used by the public report.
- `flags`: converged and numerical-breakdown bits.
- `iterations`: device-side iteration counter used by convergence handling.

Extend the Vulkan pipeline set with scalar state update shaders. The reduction pipeline continues to produce partial sums in device-local buffers; scalar update shaders consume the final element of those buffers and write `alpha`, `beta`, `rho`, and flags into the state buffer. The vector update and direction update shaders read `alpha` and `beta` from the state buffer rather than receiving host-computed push constants.

The solver records a batch of iterations in one command buffer. It submits the batch at the configured convergence interval, copies the state buffer to a small host staging buffer, and checks the converged/breakdown flags. A converged flag causes subsequent recorded vector updates to become no-ops, so a batch can finish safely without changing the solution after convergence. At the end of each batch, the host reports the actual iteration count and residual from the state buffer. The final solution copy remains a single host readback.

The initial residual, derived tolerance, and initially-converged decision are stored in the device state before the first iteration batch. Upload, initialization, and that first batch share one command submission, while the batch-boundary readback still preserves the public report. The solver must continue to reject non-finite or non-positive scalar values; failures are represented by the breakdown flag and converted to the existing `std::runtime_error` messages after the batch is read back.

## Compatibility and behavior

- The public `pcg` signature and `IterativeSolverOptions` remain unchanged.
- `convergenceCheckInterval == 1` remains numerically equivalent to the current behavior, but scalar values no longer travel through the CPU every iteration.
- For intervals greater than one, convergence is observed at the end of each batch, matching the existing option contract.
- `IterativeSolverReport::iterations`, `initialResidual`, `finalResidual`, and `converged` retain their current meaning.
- OpenCL, CPU, and CUDA implementations are unchanged.
- Existing descriptor-set caching and command-submission counters continue to report actual Vulkan activity.

## Failure handling

- A shader sets the breakdown bit when `rho`, the denominator, `alpha`, `beta`, or residual norm is non-finite, or when a required denominator is non-positive.
- The host reads the state at the batch boundary, checks the bit before scheduling another batch, and throws the same category of `std::runtime_error` currently emitted by the Vulkan solver.
- Buffer sizes and push constants are validated against `uint32_t` limits as they are today.
- The state buffer is allocated and reused with the existing size-keyed PCG workspace and is protected by the existing solve mutex.

## Testing and measurement

Add Vulkan tests for:

1. Device-side scalar updates converge to the CPU reference on small SPD systems.
2. `convergenceCheckInterval = 1` and a larger interval produce matching solutions and reports within the existing float tolerance.
3. A zero or invalid denominator sets the breakdown path and raises the expected exception.
4. The command-submission count decreases when the interval is larger than one.
5. Initially converged and zero-iteration solves preserve their report and solution semantics.

Run the existing Vulkan and CPU full test suites, the no-OpenCL Vulkan build, and the realistic sparse benchmark suite. Record command submissions and warm medians before and after the change.

## Non-goals

- Replacing the generic one-row-per-invocation CSR SpMV kernel with a subgroup-specialized kernel.
- Changing the public solver API or adding a new asynchronous solver API.
- Changing OpenCL or CUDA behavior.
