// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int n = 256;
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

int main() {
  init();
  test_ijk_time();
  test_ikj_time();
  test_ATijk_time();
  test_BTijk_time();
  return 0;
}

