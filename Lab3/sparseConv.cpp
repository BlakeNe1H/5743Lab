#include<iostream>
#include<random>
#include<stdlib.h>
#include<sys/time.h>
#include<vector>
#include<unordered_set>
#include<unordered_map>
#include<string>
#include<fstream>
#include<sstream>
using namespace std;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}


const int batch = 1;
const int height_feature = 64;
const int width_feature = 4096;
const int in_channels = 1;
const int out_channels = 512;
const int kernel_size = 3;
const int stride = 1;
const int padding = 0;

// Store sparse matrix as Triplet data structure
struct sparseMat {
    vector<float> values;
    vector<int> rowIndex;
    vector<int> columnIndex;
    int rowCnt;
    int columnCnt;

    sparseMat(int row, int column) : rowCnt(row), columnCnt(column) {}
    
    // add new value in sparse matrix
    void add(int row, int column, float value) {
        values.push_back(value);
        rowIndex.push_back(row);
        columnIndex.push_back(column);
    }

    // get target value in sparse matrix
    float get(int row, int col) const {
        for (uint i = 0; i < values.size(); i++) {
            if (rowIndex[i] == row && columnIndex[i] == col) {
                return values[i];
            }
        }
        return 0.0f;
    }
};


// data structure of rulebook
struct Rulebook {
    // <in, out, kernel_index>
    vector<tuple<int, int, int>> rules;

    void addRule(int inputIndex, int outputIndex, int kernelIndex) {
        rules.emplace_back(inputIndex, outputIndex, kernelIndex);
    }

    void print() const {
        cout << "Rulebook:" << endl;
        for (const auto& rule : rules) {
            cout << "Input Index: " << get<0>(rule)
                 << ", Output Index: " << get<1>(rule)
                 << ", Kernel Index: " << get<2>(rule) << endl;
        }
    }
};


// build Rulebook
Rulebook buildRulebook(const sparseMat& input, const sparseMat& kernel, const sparseMat& output) {
    Rulebook rulebook;

    unordered_map<int, unordered_map<int, int>> inputIndexMap;
    for (uint i = 0; i < input.values.size(); i++) {
        inputIndexMap[input.rowIndex[i]][input.columnIndex[i]] = i; // <行, 列> -> 索引
    }

    for (uint i = 0; i < input.values.size(); i++) {
        int inputRow = input.rowIndex[i];
        int inputCol = input.columnIndex[i];

        for (uint k = 0; k < kernel.values.size(); k++) {
            int kernelRow = kernel.rowIndex[k];
            int kernelCol = kernel.columnIndex[k];

            int outputRow = inputRow + kernelRow - kernel.rowCnt / 2;
            int outputCol = inputCol + kernelCol - kernel.columnCnt / 2;

            if (outputRow >= 0 && outputRow < output.rowCnt && outputCol >= 0 && outputCol < output.columnCnt) {
                if (inputIndexMap.count(outputRow) && inputIndexMap[outputRow].count(outputCol)) {
                    int outputIndex = inputIndexMap[outputRow][outputCol];
                    rulebook.addRule(i, outputIndex, k);
                }
            }
        }
    }

    return rulebook;
}


// Submanifold Sparse Convolution with Rulebook
sparseMat submanifoldSparseConvolutionWithRulebook(const sparseMat& input, const sparseMat& kernel, const Rulebook& rulebook) {
    sparseMat output(input.rowCnt, input.columnCnt);

    unordered_map<int, float> outputValues;

    for (const auto& rule : rulebook.rules) {
        int inputIndex = get<0>(rule);
        int outputIndex = get<1>(rule);
        int kernelIndex = get<2>(rule);

        outputValues[outputIndex] += input.values[inputIndex] * kernel.values[kernelIndex];
    }

    for (const auto& entry : outputValues) {
        int outputIndex = entry.first;
        float value = entry.second;
        output.add(input.rowIndex[outputIndex], input.columnIndex[outputIndex], value);
    }

    return output;
}


// generate random sparse matrix for kernel
// input matrix shape and non-zero ratio, range [0, 1]
sparseMat generate_sparse_mat(int rows, int columns, float non_zero_ratio) {
    sparseMat sparse(rows, columns);
    // calculate non-zero values number
    uint non_zero_cnt = rows * columns * non_zero_ratio;

    random_device rd;
    mt19937 gen(rd());
    // generate random value in [0, 1]
    uniform_real_distribution<float> dis(0, 1); 
    // generate random position in matrix
    uniform_int_distribution<int> row_dis(0, rows - 1);
    uniform_int_distribution<int> col_dis(0, columns - 1);

    unordered_set<int> positions;

    while (positions.size() < non_zero_cnt) {
        int row = row_dis(gen);
        int col = col_dis(gen);
        int pos = row * columns + col;

        // check if there is an exist value in this position
        if (positions.find(pos) == positions.end()) {
            // generate random value in [0, 1] and add to matrix
            positions.insert(pos);
            sparse.rowIndex.push_back(row);
            sparse.columnIndex.push_back(col);
            sparse.values.push_back(dis(gen)); 
        }
    }

    return sparse;
}

// transform dense storage structure to sparse storage structure
// input dense matrix and its shape
sparseMat dense_to_sparse(float** dense, int rows, int columns) {
    sparseMat sparse(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            // if non-zero value, add it to sparse matrix
            if (dense[i][j] != 0) {
                sparse.add(i, j, dense[i][j]);
            }
        }
    }

    return sparse;
}


// transform sparse storage structure to dense storage structure
// input sparse matrix
float** sparse_to_dense(const sparseMat& sparse, int rows, int columns) {
    float** dense = new float*[rows];
    for (int i = 0; i < rows; i++) {
        dense[i] = new float[columns];
        for (int j = 0; j < columns; j++) {
            dense[i][j] = 0;
        }
    }

    for (uint i = 0; i < sparse.values.size(); i++) {
        dense[sparse.rowIndex[i]][sparse.columnIndex[i]] += sparse.values[i];
    }

    return dense;
}

// visualize sparse matrix
void visualize_sparse(const sparseMat& sparse){
    for (uint i = 0; i < sparse.values.size(); i++) {
        cout << "Value: " << sparse.values[i] << " at (" << sparse.rowIndex[i] << ", " << sparse.columnIndex[i] << ")" << endl;
    }
}


// visualize dense matrix
void visualize_dense(float** dense, int rows, int columns){
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            cout << dense[i][j] << " ";
        }
        cout << endl;
    }
}


// main function
int main(){
    // initialize pcd matrix (64, 4096)
    float** pcd = new float*[height_feature];
    for (int i = 0; i < height_feature; i++) {
        pcd[i] = new float[width_feature];
    }
    
    // read pcd.txt
    string filename = "pointcloud.txt";
    ifstream file(filename);
    string line;

    int line_num = 0, fig = 0;
    while (getline(file, line)) {
        vector<float> row;
        istringstream words(line);
        string word;
        
        while (getline(words, word, ' ')) {
            // printf("%f", stof(word));
            pcd[line_num][fig] = stof(word);
            fig++;
            // row.push_back(stof(word));
        }

        line_num++;
        fig = 0;
    }
    file.close();


    // float*** pcd;
    // float** pcd_reshaped;
    // pcd = new float**[height_feature];
    // for (int i = 0; i < height_feature; i++) {
    //     pcd[i] = new float*[height_feature];
    //     for (int j = 0; j < height_feature; j++) {
    //         pcd[i][j] = new float[height_feature];        
    //     }
    // }

    // random_device rd;
    // mt19937 gen(rd());
    // uniform_real_distribution<float> dis(0, 1); 
    // // generate random data
    // for (int i = 0; i < height_feature; i++) {
    //     for (int j = 0; j < height_feature; j++) {
    //         for (int k = 0; k < height_feature; k++) {
    //             pcd[i][j][k] = dis(gen);
    //         }
    //     }
    // }

    // // initial pcd height_feature * width_feature
    // pcd_reshaped = new float*[height_feature];
    // for (int i = 0; i < height_feature; i++) {
    //     pcd_reshaped[i] = new float[width_feature];
    // }
    
    
    // // transform pcd (height_feature, height_feature, height_feature) to pcd_reshaped (height_feature, width_feature)
    // for (int i = 0; i < height_feature; i++) {
    //     for (int j = 0; j < height_feature; j++) {
    //         for (int k = 0; k < height_feature; k++) {
    //             pcd_reshaped[i][height_feature * j + k] = pcd[i][j][k];
    //         }
    //     }
    // }

    // visualize_dense(pcd_reshaped);
    sparseMat input = dense_to_sparse(pcd, height_feature, width_feature);

    // sparseMat input(5, 5);
    // input.add(0, 0, 1.0);
    // input.add(1, 1, 2.0);
    // input.add(2, 2, 3.0);
    // input.add(3, 3, 4.0);
    // input.add(4, 4, 5.0);

    // float** dense = new float*[5];
    // for (int i = 0; i < 5; i++) {
    //     dense[i] = new float[5];
    // }

    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         dense[i][j] = 1;
    //     }
    // }

    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         dense[i][j] = 
    //     }
    // }

    // sparseMat input_sparse = dense_to_sparse(dense, 5, 5);
    // visualize_sparse(input);
    
    // sparseMat kernel(3, 3);
    // kernel.add(0, 0, 1.0);
    // kernel.add(1, 1, 1.0);
    // kernel.add(2, 2, 1.0);
    // visualize_sparse(kernel);
    // float** temp = sparse_to_dense(kernel);
    // visualize_dense(temp, 3, 3);

    // initial kernel in each out channel
    vector<sparseMat> kernel_list;
    for (int i = 0; i < out_channels; i++) {
        kernel_list.push_back(generate_sparse_mat(kernel_size, kernel_size, 0.4));
    }

    // for (int i = 0; i < out_channels; i++) {
    //     cout << "Kernel channel: " << i << endl;
    //     for (uint j = 0; j < kernel_list[i].values.size(); j++) {
    //         cout << "Value: " << kernel_list[i].values[j] << " at (" << kernel_list[i].rowIndex[j] << ", " << kernel_list[i].columnIndex[j] << ")" << endl;
    //     }
    // }

    // get time before convolution
    float time = 0.0f;
    auto t = get_time();

    // sparse convolution
    // sparseMat output = sparse_convolution(input, kernel);

    // cout << "input" << endl;
    // visualize_dense(sparse_to_dense(input,5,5), 5, 5);


    // cout << "kernel" << endl;
    // visualize_dense(sparse_to_dense(kernel,3,3), 3, 3);

    // sparseMat output = submanifoldSparseConvolution(input, kernel);
    for (int i = 0; i < out_channels; i++) {
        sparseMat output(height_feature, width_feature);
        Rulebook rulebook = buildRulebook(input, kernel_list[i], output);
        // rulebook.print();
        output = submanifoldSparseConvolutionWithRulebook(input, kernel_list[i], rulebook);
    }
    

    
    // cout << "output:" << endl;
    // visualize_sparse(output);
    // visualize_dense(sparse_to_dense(output,5,5), 5, 5);

    // get time after convolution
    time = get_time() - t;
    cout << "Time: " << time << "s" << endl;
    

    return 0;
}