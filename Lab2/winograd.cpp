#include<iostream>
#include<random>
#include<stdlib.h>
#include<sys/time.h>
using namespace std;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}


const int batch = 1;
const int height_feature = 56;
const int width_feature = 56;
const int in_channels = 3;
const int out_channels = 64;
const int kernel_size = 3;
const int stride = 1;
const int padding = 0;

// winograd tiling data reuse overlap
const int overlap = 2;
// winograd tiled matrix 'd'
int tiled_input[4][4];
// winograd filter 'g'
int kernel[kernel_size][kernel_size];

// initial middle varibles to reduce alloc time
int mid_g[4][3];
int trans_g[4][4];
int mid_d[4][4];
int trans_d[4][4];
int mid_emul[4][4];
int mid_a[2][4];
int gamma[2][2];

// winograd standard matrix
float A[4][2] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};
float AT[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};
float B[4][4] = {{1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
float BT[4][4] = {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
float G[4][3] = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
float GT[3][4] = {{1, 0.5, 0.5, 0}, {0, 0.5, -0.5, 0}, {0, 0.5, 0.5, 1}};

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

// standard matrix multiplication
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

// tiled input should already assigned
// tiled_input[4][4], kernel[3][3]
void cal_gamma(int** kernel) {

    memset(mid_g, 0, sizeof(mid_g));
    memset(trans_g, 0, sizeof(trans_g));
    memset(mid_d, 0, sizeof(mid_d));
    memset(trans_d, 0, sizeof(trans_d));
    memset(mid_emul, 0, sizeof(mid_emul));
    memset(mid_a, 0, sizeof(mid_a));
    memset(gamma, 0, sizeof(gamma))

    // G * g
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
                mid_g[i][j] += G[i][k] * kernel[k][j];
            }
        }
    }

    // G * g * GT
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 4; j++) {
                trans_g[i][j] += mid_g[i][k] * GT[k][j];
            }
        }
    }

    // BT * b
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 4; j++) {
                mid_d[i][j] += BT[i][k] * tiled_input[k][j];
            }
        }
    }

    // BT * b * B
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 4; j++) {
                trans_d[i][j] += mid_d[i][k] * B[k][j];
            }
        }
    }

    // element-wise multiplication
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mid_emul[i][j] = trans_g[i][j] * trans_d[i][j];
        }
    }

    // AT * element-wise multiplication
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 4; j++) {
                mid_a[i][j] += AT[i][k] * mid_emul[k][j];
            }
        }
    }

    // AT * element-wise multiplication * A
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 2; j++) {
                gamma[i][j] += mid_a[i][k] * A[k][j];
            }
        }
    }
}

int*** im2col(int**** input, int kernel_size) {
    int middle_height = in_channels * kernel_size * kernel_size;
    int output_height = (height_feature - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width_feature - kernel_size + 2 * padding) / stride + 1;
    int middle_width = output_height * output_width;
    int*** middle = new int**[batch];
    for (int i = 0; i < batch; i++) {
        middle[i] = new int*[middle_height];
        for (int j = 0; j < middle_height; j++) {
            middle[i][j] = new int[middle_width];
        }
    }
    
    cout << "................start im2col................" << endl;
    int x = 0, y = 0, z = 0;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < middle_height; j++) {
            for (int m = 0; m < output_height; m++) {
                for (int n = 0; n < output_width; n++) {
                    // if (z > in_channels) {cout<< "FATAL ERROR!!!!!!!!!!!!"<<endl;}
                    middle[i][j][m * output_height + n] = input[i][z][x + m][y + n];
                }
            }
            y += stride;
            if (y >= kernel_size) {
                y = 0;
                x += stride;
                if (x >= kernel_size) {
                    x = 0;
                    z++;
                }
            }
        }
    }

    // print middle
    cout << "middle matrix:" << endl;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < middle_height; j++) {
            for (int m = 0; m < middle_width; m++) {
                cout << middle[i][j][m] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    return middle;
}

int**** wino_im2col(int**** input, int kernel_size) {
    int middle_height = kernel_size * kernel_size;
    int output_height = (height_feature - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width_feature - kernel_size + 2 * padding) / stride + 1;
    int middle_width = output_height * output_width;

    int**** output = new int***[batch];
    for (int i = 0; i < batch; i++) {
        middle[i] = new int**[in_channels];
        for (int j = 0; j < in_channels; j++) {
            middle[i][j] = new int*[middle_height];
            for (int k = 0; k < middle_height; k++) {
                middle[i][j][k] = new int[middle_width];
            }
        }
    }
    
    // cout << "................start im2col................" << endl;
    int x = 0, y = 0;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < in_channels; j++){
            for (int k = 0; k < middle_height; k++) {
                for (int m = 0; m < output_height; m++) {
                    for (int n = 0; n < output_width; n++) {
                        // if (z > in_channels) {cout<< "FATAL ERROR!!!!!!!!!!!!"<<endl;}
                        middle[i][j][k][m * output_height + n] = input[i][j][x+m][y+n];
                    }
                }
                y += stride;
                if (y >= kernel_size) {
                    y = 0;
                    x += stride;
                }
            }
        }
    }

    // // print middle
    // cout << "middle matrix:" << endl;
    // for (int i = 0; i < batch; i++) {
    //     for (int j = 0; j < middle_height; j++) {
    //         for (int m = 0; m < middle_width; m++) {
    //             cout << middle[i][j][m] << " ";
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }
    return middle;
}

int main() {
    // init input feature and kernel
    // input[batch][in_channels][height_feature][width_feature];
    int**** input = new int***[batch];
    for (int i = 0; i < batch; i++) {
        input[i] = new int**[in_channels];
        for (int j = 0; j < in_channels; j++) {
            input[i][j] = new int*[height_feature];
            for (int k = 0; k < height_feature; k++) {
                input[i][j][k] = new int[width_feature];
            }
        }
    }

    // kernel[out_channels][in_channels][kernel_size][kernel_size];
    int**** kernel = new int***[out_channels];
    for (int i = 0; i < out_channels; i++) {
        kernel[i] = new int**[in_channels];
        for (int j = 0; j < in_channels; j++) {
            kernel[i][j] = new int*[kernel_size];
            for (int k = 0; k < kernel_size; k++) {
                kernel[i][j][k] = new int[kernel_size];
            }
        }
    }

    // init them with random value
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < height_feature; k++) {
                for (int l = 0; l < width_feature; l++) {
                    input[i][j][k][l] = rand() % 10;
                }
            }
        }
    }
    for (int i = 0; i < out_channels; i++) {
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    kernel[i][j][k][l] = rand() % 10;
                }
            }
        }
    }
    
    // print input feature
    cout << "................input................" << endl;
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < in_channels; j++) {
            cout << "channel: " << j << endl;
            for (int k = 0; k < height_feature; k++) {
                for (int l = 0; l < width_feature; l++) {
                    cout << input[i][j][k][l] << " ";
                }
                cout << endl;
            }
        }
    }
    // print kernel
    cout << "................kernel................" << endl;
    for (int i = 0; i < out_channels; i++) {
        cout << "out channel: " << i << endl;
        for (int j = 0; j < in_channels; j++) {
            cout << "in channel: " << j << endl;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    cout << kernel[i][j][k][l] << " ";
                }
                cout << endl;
            }
        }
    }

    // reshape kernel to (out_channels, in_channels * kernel_size * kernel_size)
    int** kernel_reshape = new int*[out_channels];
    for (int i = 0; i < out_channels; i++) {
        kernel_reshape[i] = new int[in_channels * kernel_size * kernel_size];
    }
    for (int i = 0; i < out_channels; i++) {
        for (int j = 0; j < in_channels; j++) {
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    kernel_reshape[i][j * kernel_size * kernel_size + k * kernel_size + l] = kernel[i][j][k][l];
                }
            }
        }
    }

    // calculate output feature H & W using padding and stride
    int output_height = (height_feature - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width_feature - kernel_size + 2 * padding) / stride + 1;
    // init output[batch][out_channels][output_height][output_width];
    int*** output = new int**[batch];
    for (int i = 0; i < batch; i++) {
        output[i] = new int*[out_channels];
        for (int j = 0; j < out_channels; j++) {
            output[i][j] = new int[output_height * output_width];
        }
    }

    // im2col
    int*** middle_matrix = im2col(input, kernel_size);

    // // mat multiple
    // // middle_matrix (batch, in_channels * kernel_size * kernel_size, output_height * output_width)
    // // kernel_reshape (out_channels, in_channels * kernel_size * kernel_size)
    // // output (batch, out_channels, output_height * output_width)
    // for (int n = 0; n < batch; n++) {
    //     for (int i = 0; i < out_channels; i++) {
    //         for (int j = 0; j < output_height * output_width; j++) {
    //             output[n][i][j] = 0;
    //             for (int k = 0; k < in_channels * kernel_size * kernel_size; k++) {
    //                 output[n][i][j] += kernel_reshape[i][k] * middle_matrix[n][k][j];
    //             }
    //         }
    //     }
    // }

    // winograd
    // 4 * 4 tiled input
    float time = 0.0f;
    auto t = get_time();
    int**** wino_middle = wino_im2col(input, kernel_size);
    for (int b = 0; b < batch; b++) {
        for (int out = 0; out < out_channels; out++) {
            int x = 0, y = 0;
            for (int in = 0; in < in_channels; in++) {
                // assign 4 * 4 tiled input
                for (int m = 0; m < 4; m++) {
                    for (int n = 0; n < 4; n++) {
                        tiled_input[m][n] = input[x + m][y + n];
                    }
                }

                cal_gamma(kernel[out][in]);

                for (int i = 0; i < 4; i++){
                    for (int j = 0; j < 4; j++) {
                        output[b][out][x+i][y+j] = gamma[i][j];
                    }
                }
                
                if (y < output_width - 4) {
                    y += overlap;
                } else {
                    y = 0;
                    x += overlap;
                }
            }

        }
    }

    time = get_time() - t;
    cout << "time: " << time << endl;

    // // print output
    // cout << "................output................" << endl;
    // for (int n = 0; n < batch; n++) {
    //     cout << "batch: " << n << endl;
    //     for (int i = 0; i < out_channels; i++) {
    //         cout << "out_channels: " << i << endl;
    //         for (int j = 0; j < output_height; j++) {
    //             for (int k = 0; k < output_width; k++) {
    //                 cout << output[n][i][j * output_width + k] << " ";
    //             }
    //             cout << endl;
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }

    // verify
    // int sum = 0;
    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         for (int k = 0; k < 3; k++) {
    //             sum += kernel[0][i][j][k] * input[0][i][j][k];
    //         }
    //     }
    // }
    // cout << "sum: " << sum << endl;
    return 0;
}
