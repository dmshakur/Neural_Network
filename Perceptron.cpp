#include "Perceptron.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

Perceptron::Perceptron(std::string file_path)
{
    initialize_weights_bias();
    parse_file(file_path);
    std::cout << "File parsed, weights and bias randomized" << std::endl;
}
// number function and operations
template<size_t H, size_t V>
double Perceptron::dot_val(const std::array<std::array<double, H>, V> layer)
{
    double output;
    for (size_t i = 0; i < V; ++i)
        output += layer[i][0] * layer[i][1]; // output += val * weight
    return output;
}

double Perceptron::sigmoid(const double &num)
{ 
    return (1 / (1 + exp(-num))); 
}

double Perceptron::relu(const double &num)
{
    return (num > 0) ? num : 0;
}

double Perceptron::cost(double y, double y_hat)
{ 
    return 0.5 * ((y - y_hat) * (y - y_hat)); 
}

double Perceptron::random_number() 
{ 
    return (double)rand() / (double)RAND_MAX; 
}

void Perceptron::test()
{

}

void Perceptron::parse_file(std::string path)
{
    std::ifstream in_file;
    in_file.open(path);
    if (!in_file)
    {
        std::cout << "Unable to open file";
        return;
    }
    double sepal_length, sepal_width, petal_length, petal_width;
    std::string iris_type;
    std::string line;
    size_t line_number = 0;

    while(!in_file.eof())
    {
        std::getline(in_file, line);
        std::istringstream iss {line};

        iss >> sepal_length >> sepal_width >> petal_length >> petal_width >> iris_type;
        
        if (iris_type == "iris-setosa")
            iris_setosa[line_number] = {sepal_length, sepal_width, petal_length, petal_width};
        else if (iris_type == "iris-versicolor")
            iris_versicolor[line_number - 50] = {sepal_length, sepal_width, petal_length, petal_width};
        else if (iris_type == "iris-virginica")
            iris_virginica[line_number - 100] = {sepal_length, sepal_width, petal_length, petal_width};
        ++line_number;
    }
    for (size_t i = 0; i < 30; ++i)
    {
        training_set[i] = iris_setosa[i];
        training_set[i + 30] = iris_versicolor[i];
        training_set[i + 60] = iris_virginica[i];
    }
    for (size_t i = 30; i < 50; ++i)
    {
        test_set[i - 30] = iris_setosa[i];
        test_set[i - 10] = iris_versicolor[i];
        test_set[i + 10] = iris_virginica[i];
    }
}

void Perceptron::display()
{
    std::cout << "iris_setosa.size(): " << iris_setosa.size() << ", iris_versicolor.size(): " << iris_versicolor.size() << ", iris_virginica.size(): " << iris_virginica.size();
}

void Perceptron::display_accuracy()
{

}

void Perceptron::initialize_weights_bias()
{
    for (auto &elem : input_layer)
        elem[1] = random_number(); // weight
    for (auto &elem : hidden_layer)
    {
        elem[1] = random_number(); // weight
        elem[2] = random_number(); // bias
    }
    for (auto &elem : output_layer)
    {
        elem[1] = random_number(); // weight
        elem[2] = random_number(); // bias
    }
}

void Perceptron::train(size_t epoch_init)
{
    epochs = epoch_init;
    double total_acc;
    while (epochs > 0)
    {
        std::cout << "Epoch " << epochs << "\n\n";
        
        total_acc += epoch();

        std::cout << "Accuracy: " << (total_acc / 90.0) << "%\n\n";
        --epochs;
    }
}

double Perceptron::epoch()
{
    for (size_t i = 0; i < 90; ++i)
    {
        double y_hat, y;
        std::string y_str;
        if (i < 30)
        {
            y = 0;
            y_str = "iris_setosa";
        }
        else if (i < 60)
        {
            y = 1;
            y_str = "iris_versicolor";
        }
        else if (i < 90)
        {
            y = 2;
            y_str = "iris_virginica";
        }
        for (size_t j = 0; j < 4; ++j)
            input_layer[j][0] = training_set[i][j]; // initialize input layer
        for (size_t j = 0; j < 6; ++j)
            hidden_layer[j][0] = relu(dot_val(input_layer) + hidden_layer[j][2]); // hidden node = dot product + bias
        for (size_t j = 0; j < 3; ++j)
            output_layer[j][0] = sigmoid(dot_val(hidden_layer) + output_layer[j][2]); // setting output node
        
        std::cout << output_layer[0][0] << " " << output_layer[1][0] << " " << output_layer[2][0] << std::endl;
        return 0.0;
    }
}
