#ifndef _PERCEPTRON_h_
#define _PERCEPTRON_h_
#include <vector>
#include <array>
#include <map>

class Perceptron
{
private:
    std::array<std::array<double, 4>, 50> // Data for training
            iris_setosa,
            iris_versicolor,
            iris_virginica;
    std::array<std::array<double, 4>, 90> training_set; // 30 setosa -> 30 versicolor -> 30 virginica
    std::array<std::array<double, 4>, 60> test_set; // 20 setosa -> 20 versicolor -> 20 virginica
    std::array<double, 4> input_layer;
    // {hidden_layer} pos 0 of every secondary sub array is the val, pos 1 is the weight and pos 2 is the bias
    std::array<std::map<std::string, double>, 5> hidden_layer;
    // {ouput_layer} pos 0 = setosa, pos 1 = versicolor, pos 2 = virginica
    std::array<std::map<std::string, double>, 3> output_layer;
    size_t epochs = 1;
    template<size_t N>
    double dot_val(const std::array<std::map<std::string, double>, N> layer);
    double sigmoid(const double &num);
    double relu(const double &num);
    double random_number();
    void parse_file(std::string path);
    template<size_t N>
    void initialize(std::array<std::map<std::string, double>, N> &layer);
    void create_data_sets();
    void epoch();
    void set_hidden_values();
    void set_output_values(double y, double &y_hat);
public:
    Perceptron(std::string file_path);
    ~Perceptron() = default;
    void train(size_t epochs);
    void test();
    void display();
};

#endif