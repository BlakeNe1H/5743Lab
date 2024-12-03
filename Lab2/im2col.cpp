#include<iostream>
#include<random>
#include<stdlib.h>
#include<vector>
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

// vector<vector<float>> im2colv(vector<vector<float>>& input, int kernel_size) {
//     vector<vector<float>> middle;
//     int output_height = (height_feature - kernel_size + 2 * padding) / stride + 1;
//     int output_width = (width_feature - kernel_size + 2 * padding) / stride + 1;
    
//     return middle;
// }

int*** im2col(int**** input, int kernel_size) {
    int middle_height = in_channels * kernel_size * kernel_size;
    int output_height = (height_feature - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width_feature - kernel_size + 2 * padding) / stride + 1;
    int middel_width = output_height * output_width;
    int*** middle = new int**[batch];
    for (int i = 0; i < batch; i++) {
        middle[i] = new int*[middle_height];
        for (int j = 0; j < middle_height; j++) {
            middle[i][j] = new int[middel_width];
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
            for (int m = 0; m < middel_width; m++) {
                cout << middle[i][j][m] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    return middle;
}


int main() {
    float time = 0.0f;

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

    auto t = get_time();
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

    
    // mat multiple
    // middle_matrix (batch, in_channels * kernel_size * kernel_size, output_height * output_width)
    // kernel_reshape (out_channels, in_channels * kernel_size * kernel_size)
    // output (batch, out_channels, output_height * output_width)
    for (int n = 0; n < batch; n++) {
        for (int i = 0; i < out_channels; i++) {
            for (int j = 0; j < output_height * output_width; j++) {
                output[n][i][j] = 0;
                for (int k = 0; k < in_channels * kernel_size * kernel_size; k++) {
                    output[n][i][j] += kernel_reshape[i][k] * middle_matrix[n][k][j];
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