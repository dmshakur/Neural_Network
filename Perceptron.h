#ifndef _PERCEPTRON_h_
#define _PERCEPTRON_h_
#include <vector>
#include <array>
#include <map>

class Perceptron
{
private:
    std::array<std::array<double, 4>, 50> iris_setosa, iris_versicolor, iris_virginica;
    std::array<std::array<double, 4>, 90> training_set; // 30 setosa -> 30 versicolor -> 30 virginica
    std::array<std::array<double, 4>, 60> test_set; // 20 setosa -> 20 versicolor -> 20 virginica
    std::vector<std::map<std::string, double>> input_layer;
    // {hidden_layer} pos 0 of every secondary sub array is the val, pos 1 is the weight and pos 2 is the bias
    std::vector<std::map<std::string, double>> hidden_layer;
    // {ouput_layer} pos 0 = setosa, pos 1 = versicolor, pos 2 = virginica
    std::vector<std::map<std::string, double>> output_layer;
    size_t epochs = 1;
    double dot_val(std::vector<std::map<std::string, double>> &layer);
    double sigmoid(const double num);
    double relu(const double num);
    double random_number();
    void parse_file(std::string path);
    void initialize(std::vector<std::map<std::string, double>> &layer, size_t size, std::string layer_type = "");
    void create_data_sets();
    void epoch();
    void forward_prop(double y, double &y_hat);
    void back_prop(std::array<double, 3> expected);
    void update_network(double l_rate);
public:
    Perceptron(std::string file_path);
    ~Perceptron() = default;
    void train(size_t epochs = 1);
    void test();
    void display();
};

#endif