#ifndef _PERCEPTRON_h_
#define _PERCEPTRON_h_
#include <fstream>
#include <vector>
#include <array>

class Perceptron
{
private:
    std::vector<std::vector<double>> // Data for training
            iris_setosa,
            iris_versicolor,
            iris_virginica;
    // {input_layer} pos 0 of every secondary sub array is the val and pos 1 is the weight
    std::array<std::array<double, 2>, 4> input_layer; 
    // {hidden_layer} pos 0 of every secondary sub array is the val and pos 1 is the weight
    // pos 2 is the bias
    std::array<std::array<double, 3>, 6> hidden_layer;
    std::array<std::array<double, 2>, 4> output_layer;
    size_t epochs = 1000;
    template<size_t H, size_t V>
    double dot_val(const std::array<std::array<double, H>, V>);
    double sigmoid(const double temp);
    double random_number();
    void parse_file(std::string path);
    void initialize_weights_bias();
public:
    Perceptron(std::string file_path);
    ~Perceptron();
    void run();
    void display();
};

#endif