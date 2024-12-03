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

int** alloc_m(int l) {
  int** m = new int*[l];
  for (int i = 0; i < l; i++) {
    m[i] = new int[l];
  }
  return m;
}

void delete_m(int** m, int l) {
  for (int i = 0; i < l; i++) {
    delete[] m[i];
  }
  delete[] m;
}

// return A + B
int** sum_m(int** A, int** B, int l) {
  int** C = alloc_m(l);
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < l; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
  return C;
}

// return A - B
int** sub_m(int** A, int** B, int l) {
  int** C = alloc_m(l);
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < l; j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
  return C;
}

int** matmul_ordinary(int** A, int** B, int l) {
  int** temp = alloc_m(l);
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < l; j++) {
      temp[i][j] = 0;
    }
  }
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < l; j++) {
      for (int k = 0; k < l; k++) {
        temp[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
  return temp;
}

int** one_strassen(int** A, int** B, int l) {
  if (l <= 64) {
    return matmul_ordinary(A, B, l);
  }
  // init block matrix
  int** A11 = alloc_m(l / 2);
  int** A12 = alloc_m(l / 2);
  int** A21 = alloc_m(l / 2);
  int** A22 = alloc_m(l / 2);
  int** B11 = alloc_m(l / 2);
  int** B12 = alloc_m(l / 2);
  int** B21 = alloc_m(l / 2);
  int** B22 = alloc_m(l / 2);

  for (int i = 0; i < l / 2; i++) {
    for (int j = 0; j < l / 2; j++) {
      A11[i][j] = A[i][j];
      A12[i][j] = A[i][j + l / 2];
      A21[i][j] = A[i + l / 2][j];
      A22[i][j] = A[i + l / 2][j + l / 2];
      B11[i][j] = B[i][j];
      B12[i][j] = B[i][j + l / 2];
      B21[i][j] = B[i + l / 2][j];
      B22[i][j] = B[i + l / 2][j + l / 2];
    }
  }

  // M1 = (A11 + A22) * (B11 + B22)
  // M2 = (A21 + A22) * B11
  // M3 = A11 * (B12 - B22)
  // M4 = A22 * (B21 - B11)
  // M5 = (A11 + A12) * B22
  // M6 = (A21 - A11) * (B11 + B12)
  // M7 = (A12 - A22) * (B21 + B22)
  // C11 = M1 + M4 - M5 + M7
  // C12 = M3 + M5
  // C21 = M2 + M4
  // C22 = M1 - M2 + M3 + M6
  int** M1 = one_strassen(sum_m(A11, A22, l / 2), sum_m(B11, B22, l / 2), l / 2);
  int** M2 = one_strassen(sum_m(A21, A22, l / 2), B11, l / 2);
  int** M3 = one_strassen(A11, sub_m(B12, B22, l / 2), l / 2);
  int** M4 = one_strassen(A22, sub_m(B21, B11, l / 2), l / 2);
  int** M5 = one_strassen(sum_m(A11, A12, l / 2), B22, l / 2);
  int** M6 = one_strassen(sub_m(A21, A11, l / 2), sum_m(B11, B12, l / 2), l / 2);
  int** M7 = one_strassen(sub_m(A12, A22, l / 2), sum_m(B21, B22, l / 2), l / 2);

  int** C11 = sum_m(sub_m(sum_m(M1, M4, l / 2), M5, l / 2), M7, l / 2);
  int** C12 = sum_m(M3, M5, l / 2);
  int** C21 = sum_m(M2, M4, l / 2);
  int** C22 = sum_m(sub_m(sum_m(M1, M3, l / 2), M2, l / 2), M6, l / 2);

  int** C = alloc_m(l);
  for (int i = 0; i < l / 2; i++) {
    for (int j = 0; j < l / 2; j++) {
      C[i][j] = C11[i][j];
      C[i][j + l / 2] = C12[i][j];
      C[i + l / 2][j] = C21[i][j];
      C[i + l / 2][j + l / 2] = C22[i][j];
    }
  }
  
  delete_m(A11, l / 2);
  delete_m(A12, l / 2);
  delete_m(A21, l / 2);
  delete_m(A22, l / 2);
  delete_m(B11, l / 2);
  delete_m(B12, l / 2);
  delete_m(B21, l / 2);
  delete_m(B22, l / 2);
  delete_m(M1, l / 2);
  delete_m(M2, l / 2);
  delete_m(M3, l / 2);
  delete_m(M4, l / 2);
  delete_m(M5, l / 2);
  delete_m(M6, l / 2);
  delete_m(M7, l / 2);
  delete_m(C11, l / 2);
  delete_m(C12, l / 2);
  delete_m(C21, l / 2);
  delete_m(C22, l / 2);

  // if (l >= 32) {
  //   printf("%d\n", l);
  // }

  return C;
}

void matmul_strassen() {
  memset(C, 0, sizeof(C));
  // strassen algorithm
  int** a = alloc_m(n);
  int** b = alloc_m(n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i][j] = A[i][j];
      b[i][j] = B[i][j];
    }
  }
  int ** c = one_strassen(a, b, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      C[i][j] = c[i][j];
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

void test_strassen_time() {
  // init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    matmul_strassen();
    test();
    // printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Strassen Algorithm: %f\n", avg_time / 32);
}

int main() {
  init();
  test_ijk_time();
  test_strassen_time();
  return 0;
}
