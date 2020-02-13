#ifndef _PERCEPTRON_h_
#define _PERCEPTRON_h_
#include <vector>
#include <array>

class Perceptron
{
private:
    std::array<std::array<double, 4>, 50> // Data for training
            iris_setosa,
            iris_versicolor,
            iris_virginica;
    std::array<std::array<double, 4>, 90> training_set; // 30 setosa -> 30 versicolor -> 30 virginica
    std::array<std::array<double, 4>, 60> test_set; // 20 setosa -> 20 versicolor -> 20 virginica

    // {input_layer} pos 0 of every secondary sub array is the val and pos 1 is the weight
    std::array<std::array<double, 2>, 4> input_layer; 
    // {hidden_layer} pos 0 of every secondary sub array is the val, pos 1 is the weight and pos 2 is the bias
    std::array<std::array<double, 3>, 6> hidden_layer;
    // {ouput_layer} pos 0 = setosa, pos 1 = versicolor, pos 2 = virginica
    std::array<std::array<double, 3>, 3> output_layer;
    size_t epochs = 1;

    template<size_t H, size_t V>
    double dot_val(const std::array<std::array<double, H>, V>);
    double sigmoid(const double &num);
    double relu(const double &num);
    double cost(double y, double y_hat);
    double random_number();
    void parse_file(std::string path);
    void initialize_weights_bias();
    void create_data_sets();
    double epoch();
public:
    Perceptron(std::string file_path);
    ~Perceptron() = default;
    void train(size_t epochs);
    void test();
    void display();
    void display_accuracy();
};

#endif