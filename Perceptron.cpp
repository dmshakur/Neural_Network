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
    std::cout << "File parsed, weights and bias' randomized" << std::endl;
}

template<size_t H, size_t V>
double Perceptron::dot_val(const std::array<std::array<double, H>, V> layer)
{
    double output;

    for (size_t i = 0; i < V; ++i)
        output += layer[i][0] * layer[i][1];

    return sigmoid(output);
}

double Perceptron::sigmoid(const double temp )//[])
{
    // double temp;
    // for (size_t i = 0; i < 4; ++i)
    //     temp += layer[i];
    return (1 / (1 + exp(-temp)));
}

double Perceptron::random_number()
{
    return (double)rand() / (double)RAND_MAX;
}

void Perceptron::run()
{
    std::setprecision(5);
    for (auto vec_of_doubles : iris_setosa)
    {
        for (size_t i = 0; i < 6; ++i)
            hidden_layer[i][0] = dot_val(input_layer);

        for (size_t i = 0; i < 4; ++i)
            output_layer[i][0] = dot_val(hidden_layer);

        // std::cout << output_layer;
    }
}

void Perceptron::parse_file(std::string path)
{
    std::ifstream in_file;
    in_file.open(path);
    if (!in_file)
        throw "Unable to open file";
    double sepal_length, sepal_width, petal_length, petal_width;
    std::string iris_type;
    std::string line;

    while(!in_file.eof())
    {
        std::getline(in_file, line);
        std::istringstream iss {line};

        iss >> sepal_length >> sepal_width >> petal_length >> petal_width >> iris_type;
        
        if (iris_type == "iris-setosa")
            iris_setosa.push_back({sepal_length, sepal_width, petal_length, petal_width});
        else if (iris_type == "iris-versicolor")
            iris_versicolor.push_back({sepal_length, sepal_width, petal_length, petal_width});
        else if (iris_type == "iris-virginica")
            iris_virginica.push_back({sepal_length, sepal_width, petal_length, petal_width});
    }
}

void Perceptron::display()
{
    std::cout << "iris_setosa.size(): " << iris_setosa.size() << ", iris_versicolor.size(): " << iris_versicolor.size() << ", iris_virginica.size(): " << iris_virginica.size();
}

void Perceptron::initialize_weights_bias ()
{
    for (size_t i = 0; i < 4; ++i)
    {
        input_layer[i][1] = random_number(); // Weight
        output_layer[i][1] = random_number(); // Weight
    }
    for (auto &elem : hidden_layer)
    {
        elem[1] = random_number(); // Weight
        elem[2] = random_number(); // Bias
    }
}