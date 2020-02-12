#ifndef _PERCEPTRON_h_
#define _PERCEPTRON_h_
#include <fstream>
#include <vector>

class Perceptron
{
private:
    std::vector<double [3]> past;
    std::vector<std::vector<double>> iris_setosa;
    std::vector<std::vector<double>> iris_versicolor;
    std::vector<std::vector<double>> iris_virginica;
    double input_layer [4], 
            hidden_layer [6], 
            hidden_weights [6],
            output_layer [3], 
            bias,
            learning_rate;
    int epochs;
    // double randomize();
    double dot_val(const double &node);
    double sigmoid(const double &node);
    void parse_file(std::string path);
public:
    Perceptron(std::string file_path);
    ~Perceptron();
    void engage_perceptron();
    void display();
};

#endif