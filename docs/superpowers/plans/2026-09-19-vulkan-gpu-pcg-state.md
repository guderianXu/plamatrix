# Vulkan GPU-resident PCG State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep Vulkan PCG scalar recurrence and convergence state on the device so one command submission can cover a configurable batch of iterations.

**Architecture:** Add a fixed-size device-local `PcgState` buffer and shader stages that turn reduction outputs into `alpha`, `beta`, `rho`, residual norm, and status flags. The host records a batch bounded by `convergenceCheckInterval`, reads only the state at the batch boundary, and preserves the existing report and error behavior.

**Tech Stack:** C++20, Vulkan 1.x compute shaders in GLSL 450, CMake shader compilation, GoogleTest, existing `CommandContext` descriptor caching.

**Spec:** `docs/superpowers/specs/2026-09-19-vulkan-gpu-pcg-state-design.md`

## Global Constraints

- Keep the public `pcg` signature and `IterativeSolverOptions` unchanged.
- Modify only the Vulkan implementation, Vulkan shaders, Vulkan tests, and matching benchmark/documentation text.
- Keep all new device buffers in the reusable size-keyed `PcgBuffers` workspace and under the existing solve mutex.
- Preserve the current finite-value, positive-denominator, and `requireConvergence` behavior.
- Maintain C++20/CMake compatibility and format modified C++ with the repository clang-format configuration.

## Review Focus

- A batch that converges before its final scheduled iteration must freeze solution and residual updates; test with `convergenceCheckInterval = 4`.
- A non-positive or non-finite scalar must set the device breakdown flag and raise the existing runtime error; test with a deliberately singular/indefinite input.
- `maxIterations` smaller than the interval must submit one partial batch and report the exact executed iteration count; test with `maxIterations = 3`, interval 8.
- An initially converged system must perform zero PCG iterations and leave the supplied solution unchanged; test with a zero residual.
- State and descriptor buffers must be reused across warm solves without stale flags; run two solves with different RHS and compare both CPU references.

### Task 1: Define GPU state and scalar update shaders

**Files:**
- Create: `src/vulkan/shaders/pcg_state_init.comp`
- Create: `src/vulkan/shaders/pcg_alpha.comp`
- Create: `src/vulkan/shaders/pcg_beta.comp`
- Create: `src/vulkan/shaders/pcg_convergence.comp`
- Modify: `src/vulkan/shaders/update.comp`
- Modify: `src/vulkan/shaders/direction.comp`
- Modify: `src/CMakeLists.txt` shader source list

**Interfaces:**
- `PcgState` is a std430 float/uint buffer with fixed indices: `rho`, `alpha`, `beta`, `residual_squared`, `tolerance_squared`, `flags`, and `iterations`.
- `pcg_alpha.comp` consumes the current `rho` and denominator reduction result, writes `alpha`, and sets the breakdown bit when either scalar is non-finite or the denominator is non-positive.
- `pcg_beta.comp` consumes the next `rho`, writes `beta`, updates `rho`, and sets the breakdown bit on invalid values.
- `pcg_convergence.comp` consumes the residual reduction result, compares it to `tolerance_squared`, sets the converged bit, and writes the iteration count.
- `update.comp` and `direction.comp` receive the state buffer as an additional storage binding and skip writes when the converged or breakdown bit is set.

- [ ] **Step 1: Add shader sources with explicit bindings and guarded arithmetic.**

  Use one `uint flags` bit for converged and one for breakdown. Every scalar shader must check `isnan`/`isinf` through GLSL `isnan`/`isinf` before writing dependent values.

- [ ] **Step 2: Add the shader names to the Vulkan CMake compilation list.**

  Run `cmake --build build-vulkan --target plamatrix_vulkan_shaders -j2` and expect all new `.comp` files to produce SPIR-V output.

- [ ] **Step 3: Run the shader-loading Vulkan test binary.**

  Run `./build-vulkan/test/plamatrix_vulkan_tests --gtest_filter=VulkanRuntime.*` and expect all selected tests to pass.

### Task 2: Integrate state buffers and batched recording into the solver

**Files:**
- Modify: `src/vulkan/iterative_solver.cpp`
- Modify: `include/plamatrix/vulkan/execution.h` only if a small readback helper is required
- Modify: `src/vulkan/execution.cpp` only if the helper needs command-context support

**Interfaces:**
- `PcgBuffers` owns `state`, `stateUpload`, and `stateReadback` buffers; `state` is device-local storage and the two staging buffers are host-visible transfer buffers.
- A `PcgStateHost` C++ struct mirrors the shader layout and is validated with `static_assert` size/offset checks.
- `readState()` submits the current batch, copies the state buffer to `stateReadback`, maps it, and returns a validated `PcgStateHost`.

- [ ] **Step 1: Add reusable state storage and host initialization.**

  Initialize `rho`, tolerance squared, zero flags, and zero iterations in `stateUpload`; copy it to `state` in the initial command batch. Reinitialize the state for every solve so warm solves cannot inherit flags.

- [ ] **Step 2: Replace host scalar calculations in the iteration body.**

  Record SpMV, denominator reduction, `pcg_alpha`, vector update, optional residual reduction/convergence shader, Jacobi, next-rho reduction, `pcg_beta`, and direction update. Push constants should contain only counts; `alpha`, `beta`, and `rho` must come from the state buffer.

- [ ] **Step 3: Implement batch boundaries.**

  Record at most `min(convergenceCheckInterval, maxIterations - iterations)` iterations, submit once, read `PcgStateHost`, and stop on converged/breakdown/max-iterations. If no convergence check is scheduled before the batch ends, run the convergence reduction on the final iteration so `finalResidual` is available.

- [ ] **Step 4: Preserve report and error semantics.**

  Convert state flags into the current runtime errors, set `report.iterations` from the device counter, set `report.finalResidual` from the square root of the checked residual, and honor `requireConvergence` exactly as the current implementation does.

- [ ] **Step 5: Build the affected Vulkan target.**

  Run `cmake --build build-vulkan -j2 --target plamatrix_vulkan_tests plamatrix_backend_compare`; expect both targets to link without OpenCL/CUDA changes.

### Task 3: Add behavioral and batching regression tests

**Files:**
- Modify: `test/unit/vulkan/vulkan_runtime_test.cpp`
- Modify: `test/CMakeLists.txt` only if a new test source is needed

**Interfaces:**
- Tests call the existing `plamatrix::vulkan::pcg` API and inspect `IterativeSolverReport`.

- [ ] **Step 1: Add a batched-vs-unbatched CPU reference test.**

  Solve the existing 257-row SPD tridiagonal fixture with interval 1 and interval 8, compare both Vulkan solutions to the CPU solution within `2e-4f`, and assert the batched report uses no more submissions than the interval-1 report.

- [ ] **Step 2: Add warm-state reset coverage.**

  Solve two different RHS vectors sequentially with interval 4 and assert both match separate CPU references, proving flags and scalar state are reinitialized.

- [ ] **Step 3: Add max-iteration partial-batch coverage.**

  Use `maxIterations = 3`, interval 8, and a strict tolerance; assert `report.iterations == 3` and no extra direction update changes the returned solution.

- [ ] **Step 4: Run the focused Vulkan tests.**

  Run `./build-vulkan/test/plamatrix_vulkan_tests --gtest_filter='VulkanSolver.*'` and expect all selected tests to pass.

### Task 4: Validate performance, documentation, and integration

**Files:**
- Modify: `benchmark/backend_compare.cpp` if the CSV needs a batch/submission field
- Modify: `docs/` benchmark or Vulkan performance documentation that describes scalar readbacks

- [ ] **Step 1: Run formatting and static checks.**

  Run `clang-format --dry-run --Werror` on modified C++ files and `git diff --check`.

- [ ] **Step 2: Run full Vulkan and CPU verification.**

  Build and run `ctest --test-dir build-vulkan --output-on-failure` and `ctest --test-dir build-cpu --output-on-failure`; expect 316/316 Vulkan tests and 288/288 CPU tests, with only the existing hardware-dependent skips.

- [ ] **Step 3: Run no-OpenCL verification.**

  Build `plamatrix_backend_compare` in `build-vulkan-no-opencl` and run the MVS visibility case; expect a Vulkan CSV row and an OpenCL skipped row.

- [ ] **Step 4: Run the realistic benchmark suite.**

  Save `--suite` output under `build/tmp/vulkan-realistic-benchmark/gpu-state.csv`, compare `command_submissions` and `warm_median_ms` to the previous descriptor-cache and direction-chain CSVs, and retain the CSV for review.

- [ ] **Step 5: Commit and push only the submodule changes.**

  Recheck `git status --short`, commit the implementation as `perf: keep Vulkan PCG scalars on device`, push `main`, and verify `git ls-remote origin refs/heads/main` equals the new commit.
