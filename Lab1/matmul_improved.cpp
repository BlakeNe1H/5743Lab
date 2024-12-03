#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cmath>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int n = 1024;
int A[n][n];
int B[n][n];
int BT[n][n];
int AT[n][n];
int C[n][n];
int C_groundtruth[n][n];

void init() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = rand(); 
      B[i][j] = rand(); 
    } 
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

void matmul() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_ikj() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_AT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      AT[i][j] = A[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += AT[k][i] * B[k][j];    
      }   
    }
  }
}

void matmul_BT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * BT[j][k];    
      }   
    }
  }
}

void matmul_improve1() {
  memset(C, 0, sizeof(C));
  // improve 1
  // Loop unrolling
  for (int i = 0; i < n; i+=4) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j+=4) {
        C[i][j] += A[i][k] * B[k][j];
        C[i][j+1] += A[i][k] * B[k][j+1];
        C[i][j+2] += A[i][k] * B[k][j+2];
        C[i][j+3] += A[i][k] * B[k][j+3];

        C[i+1][j] += A[i+1][k] * B[k][j];
        C[i+1][j+1] += A[i+1][k] * B[k][j+1];
        C[i+1][j+2] += A[i+1][k] * B[k][j+2];
        C[i+1][j+3] += A[i+1][k] * B[k][j+3];

        C[i+2][j] += A[i+2][k] * B[k][j];
        C[i+2][j+1] += A[i+2][k] * B[k][j+1];
        C[i+2][j+2] += A[i+2][k] * B[k][j+2];
        C[i+2][j+3] += A[i+2][k] * B[k][j+3];

        C[i+3][j] += A[i+3][k] * B[k][j];
        C[i+3][j+1] += A[i+3][k] * B[k][j+1];
        C[i+3][j+2] += A[i+3][k] * B[k][j+2];
        C[i+3][j+3] += A[i+3][k] * B[k][j+3];
      }
    }
  }
}

void matmul_improve2() {
  memset(C, 0, sizeof(C));
  // improve 2
  // Tiling
  int tile_size = 32;
  for (int i0 = 0; i0 < n; i0 += tile_size) {
        for (int k0 = 0; k0 < n; k0 += tile_size) {
            for (int j0 = 0; j0 < n; j0 += tile_size) {
                for (int i = i0; i < i0 + tile_size; i++) {
                    for (int k = k0; k < k0 + tile_size; k++) {
                        for (int j = j0; j < j0 + tile_size; j++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

void test_ijk_time() {
  // init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    // matmul_ikj();
    matmul(); 
    // matmul_AT();
    // matmul_BT();
    test();
    // printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for ijk Algorithm: %f\n", avg_time / 32);
}

void test_ikj_time() {
  // init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    matmul_ikj();
    // matmul(); 
    // matmul_AT();
    // matmul_BT();
    test();
    // printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for ikj Algorithm: %f\n", avg_time / 32);
}

void test_ATijk_time() {
  // init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    // matmul_ikj();
    // matmul(); 
    matmul_AT();
    // matmul_BT();
    test();
    // printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for ATijk Algorithm: %f\n", avg_time / 32);
}

void test_BTijk_time() {
  // init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    // matmul_ikj();
    // matmul(); 
    // matmul_AT();
    matmul_BT();
    test();
    // printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for BTijk Algorithm: %f\n", avg_time / 32);
}

void test_improve1_time() {
  // init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    matmul_improve1();
    test();
    // printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Loop Unrolling Algorithm: %f\n", avg_time / 32);
}

void test_improve2_time() {
  // init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    matmul_improve2();
    test();
    // printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Tiling Algorithm: %f\n", avg_time / 32);
}

int main() {
  init();
  test_ijk_time();
  test_ikj_time();
  // test_ATijk_time();
  // test_BTijk_time();
  test_improve1_time();
  test_improve2_time();
  return 0;
}
