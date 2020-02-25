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
    parse_file(file_path);
    initialize(hidden_layer);
    initialize(output_layer);
    std::cout << "File parsed, weights and bias randomized" << std::endl;
}
// number function and operations
template<size_t N>
double Perceptron::dot_val(const std::array<std::map<std::string, double>, N> layer)
{
    double output;
    for (size_t i = 0; i < V; ++i)
        output += layer[i]["value"] * layer[i]["weight"]; // output += val * weight
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
    double sepal_length, sepal_width, petal_length, petal_width, iris_type;
    std::string line;
    size_t line_number = 0;

    while(!in_file.eof())
    {
        std::getline(in_file, line);
        std::istringstream iss {line};

        iss >> sepal_length >> sepal_width >> petal_length >> petal_width >> iris_type;
        
        if (iris_type == 0)
            iris_setosa[line_number] = {sepal_length, sepal_width, petal_length, petal_width};
        else if (iris_type == 1)
            iris_versicolor[line_number - 50] = {sepal_length, sepal_width, petal_length, petal_width};
        else if (iris_type == 2)
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
    std::setprecision(2);
    std::cout << "input_layer:" << std::endl;
    std::cout << "value, weight, bias" << std::endl;
    for (const auto &node : input_layer)
        std::cout << "[" << node << "], ";
    std::cout << std::endl;
    std::cout << "hidden_layer:" << std::endl;
    for (auto &node : hidden_layer)
        std::cout << "[" << node["value"] << ", " << node["weight"] << ", " << node["bias"] << "], ";
    std::cout << std::endl;
    std::cout << "output_layer:" << std::endl;
    for (auto &node : output_layer)
        std::cout << "[" << node["value"] << ", " << node["weight"] << ", " << node["bias"] << "], ";
    std::cout << std::endl;
}

template<size_t N>
void Perceptron::initialize(std::array<std::map<std::string, double>, N> &layer)
{
    for (auto &elem : layer)
    {
        elem["value"] = 0; // value
        elem["weight"] = random_number(); // weight
        elem["bias"] = random_number(); // bias
    }
}

void Perceptron::train(size_t epoch_init)
{
    epochs = epoch_init;
    while (epochs > 0)
    {
        std::cout << "Epoch " << epochs << "\n\n";
        
        epoch();

        std::cout << "Accuracy: " << "?" << "%\n\n";
        --epochs;
    }
}

void set_y_val(std::string &y_str, double &y, const size_t i)
{
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
}

void Perceptron::epoch()
{
    for (size_t i = 0; i < 90; ++i)
    {
        double y_hat, y;
        std::string y_str;
        set_y_val(y_str, y, i);

        input_layer = training_set[i]; // Initialize input layer
        set_hidden_values();
        set_output_values(y, y_hat);

        std::cout << i << " " << std::boolalpha << (y_hat == y) << ", y^: " << y_hat << " y: " << y << std::endl;
    }
}

void Perceptron::set_hidden_values()
{
    for (size_t i = 0; i < 6; ++i)
        hidden_layer[i]["value"] = sigmoid( + hidden_layer[i]["bias"]); // hidden node = dot product + bias
}

void Perceptron::set_output_values(double y, double &y_hat)
{
    for (size_t i = 0; i < 3; ++i)
    {
        output_layer[i]["value"] = sigmoid(dot_val(hidden_layer) + output_layer[i]["bias"]); // setting output node
        if (i == y)
            output_layer[i]["y"] = 1;
        else
            output_layer[i]["y"] = 0;

        if (output_layer[i]["value"] > y_hat) // setting actual value
            y_hat = (double)i;
    }
}
