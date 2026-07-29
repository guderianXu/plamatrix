#pragma once

namespace plamatrix
{
namespace iterative_solver_detail
{

#ifdef PLAMATRIX_ITERATIVE_SOLVER_TEST_HOOKS
void setForcedWorkspaceAllocationFailureAfter(int successful_allocations) noexcept;
void setFixedSolverCompletionGate(void* event) noexcept;
#endif

} // namespace iterative_solver_detail
} // namespace plamatrix
