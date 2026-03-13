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
    // Concatenate all K[0..i] to form K_matrix [i+1, d]
    // Concatenate all V[0..i] to form V_matrix [i+1, d]
    // Compute Softmax(Q @ K^T) @ V

    // First, concatenate all keys vertically
    Matrix *K_matrix = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        K_matrix = matrix_memory_allocator.Allocate("K_matrix");
        gpu_sim.Copy(keys[j], K_matrix, kInGpuHbm);
      } else {
        Matrix *new_K = matrix_memory_allocator.Allocate("new_K");
        gpu_sim.Concat(K_matrix, keys[j], new_K, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(K_matrix);
        K_matrix = new_K;
      }
    }

    // Concatenate all values vertically
    Matrix *V_matrix = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        V_matrix = matrix_memory_allocator.Allocate("V_matrix");
        gpu_sim.Copy(values[j], V_matrix, kInGpuHbm);
      } else {
        Matrix *new_V = matrix_memory_allocator.Allocate("new_V");
        gpu_sim.Concat(V_matrix, values[j], new_V, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(V_matrix);
        V_matrix = new_V;
      }
    }

    // Move Q, K, V to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_matrix);
    gpu_sim.MoveMatrixToSharedMem(V_matrix);

    // Transpose K to get K^T
    gpu_sim.Transpose(K_matrix, kInSharedMemory);

    // Compute Q @ K^T -> shape [i+1, i+1]
    Matrix *QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_matrix, QK);

    // We can release K_matrix now
    gpu_sim.ReleaseMatrix(K_matrix);

    // Compute exp(QK) for softmax
    Matrix *QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);
    gpu_sim.ReleaseMatrix(QK);

    // For softmax, we need to divide each row by its row sum
    // QK_exp is [i+1, i+1]
    // We need to compute softmax row-wise
    Matrix *softmax_matrix = matrix_memory_allocator.Allocate("softmax_matrix");

    // For each row, compute the sum and divide
    for (size_t row = 0; row <= i; ++row) {
      // Get the row
      Matrix *row_vec = matrix_memory_allocator.Allocate("row_vec");
      gpu_sim.GetRow(QK_exp, row, row_vec, kInSharedMemory);

      // Sum the row
      Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_vec, row_sum);

      // Divide row by sum
      Matrix *normalized_row = matrix_memory_allocator.Allocate("normalized_row");
      gpu_sim.MatDiv(row_vec, row_sum, normalized_row);

      gpu_sim.ReleaseMatrix(row_vec);
      gpu_sim.ReleaseMatrix(row_sum);

      // Concatenate to softmax_matrix
      if (row == 0) {
        gpu_sim.Copy(normalized_row, softmax_matrix, kInSharedMemory);
      } else {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("new_softmax");
        gpu_sim.Concat(softmax_matrix, normalized_row, new_softmax, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_matrix);
        softmax_matrix = new_softmax;
      }
      gpu_sim.ReleaseMatrix(normalized_row);
    }

    gpu_sim.ReleaseMatrix(QK_exp);

    // Compute softmax_matrix @ V_matrix
    Matrix *result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_matrix, V_matrix, result);

    gpu_sim.ReleaseMatrix(softmax_matrix);
    gpu_sim.ReleaseMatrix(V_matrix);

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