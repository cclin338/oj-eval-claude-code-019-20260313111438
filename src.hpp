#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // For round i (0-based), we have i+1 keys and values
    // Q has shape [i+1, d], each K and V has shape [1, d]
    // Compute: Answer = sum_{j=0}^{i} Softmax(Q @ K_j^T) @ V_j
    // Where Q @ K_j^T gives [i+1, 1], softmax on this, then @ V_j [1, d] -> [i+1, d]

    // Move Q to SRAM once
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Accumulate result
    Matrix *result = nullptr;

    for (size_t j = 0; j <= i; ++j) {
      // Move K_j to SRAM
      gpu_sim.MoveMatrixToSharedMem(keys[j]);

      // Copy and transpose K_j
      Matrix *k_t = matrix_memory_allocator.Allocate("k_t");
      gpu_sim.Copy(keys[j], k_t, kInSharedMemory);
      gpu_sim.Transpose(k_t, kInSharedMemory);

      // Compute Q @ K_j^T -> [i+1, 1]
      Matrix *qk = matrix_memory_allocator.Allocate("qk");
      gpu_sim.MatMul(current_query, k_t, qk);
      gpu_sim.ReleaseMatrix(k_t);

      // Compute exp(qk)
      Matrix *qk_exp = matrix_memory_allocator.Allocate("qk_exp");
      gpu_sim.MatExp(qk, qk_exp);
      gpu_sim.ReleaseMatrix(qk);

      // Sum to get normalizer
      Matrix *normalizer = matrix_memory_allocator.Allocate("normalizer");
      gpu_sim.Sum(qk_exp, normalizer);

      // Divide to get softmax
      Matrix *softmax_vec = matrix_memory_allocator.Allocate("softmax_vec");
      gpu_sim.MatDiv(qk_exp, normalizer, softmax_vec);
      gpu_sim.ReleaseMatrix(qk_exp);
      gpu_sim.ReleaseMatrix(normalizer);

      // Move V_j to SRAM
      gpu_sim.MoveMatrixToSharedMem(values[j]);

      // Compute softmax @ V_j -> [i+1, d]
      Matrix *attn_out = matrix_memory_allocator.Allocate("attn_out");
      gpu_sim.MatMul(softmax_vec, values[j], attn_out);
      gpu_sim.ReleaseMatrix(softmax_vec);

      // Move K_j and V_j back to HBM
      gpu_sim.MoveMatrixToGpuHbm(keys[j]);
      gpu_sim.MoveMatrixToGpuHbm(values[j]);

      // Accumulate
      if (j == 0) {
        result = attn_out;
      } else {
        Matrix *new_result = matrix_memory_allocator.Allocate("new_result");
        gpu_sim.MatAdd(result, attn_out, new_result);
        gpu_sim.ReleaseMatrix(result);
        gpu_sim.ReleaseMatrix(attn_out);
        result = new_result;
      }
    }

    // Move Q back to HBM
    gpu_sim.MoveMatrixToGpuHbm(current_query);

    // Move result to HBM for submission
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu