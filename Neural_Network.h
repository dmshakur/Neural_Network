#ifndef _NEURAL_NETWORK_h_
#define _NEURAL_NETWORK_h_
#include <vector>
#include <array>
#include <map>

class Neural_Network
{
private:
    std::array<std::array<double, 4>, 50> iris_setosa, iris_versicolor, iris_virginica; // an array of all the data for each flower
    std::array<std::array<double, 4>, 90> training_set; // 30 setosa -> 30 versicolor -> 30 virginica
    std::array<std::array<double, 4>, 60> test_set; // 20 setosa -> 20 versicolor -> 20 virginica
    std::vector<std::vector<double>> values, bias, errors;
    std::vector<std::vector<std::vector<double>>> weights;
    size_t net_size = 0;
    double dot_val(std::vector<double> val, std::vector<double> weights);
    double sigmoid(const double num);
    double random_number();
    void parse_file(std::string path);
    void initialize(std::vector<size_t> layers);
    void create_data_sets();
    void epoch();
    void forward_prop(double &y_hat, std::vector<double> &expected, double &cost);
    void back_prop(std::vector<double> expected);
    void update_network(double l_rate);
public:
    Neural_Network(std::string file_path);
    ~Neural_Network() = default;
    void train(size_t epochs = 1);
    void display();
};

#endif