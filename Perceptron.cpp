#include "Perceptron.h"
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>

Perceptron::Perceptron(std::string file_path)
    : {
        for (auto node : input_layer)
            node = (double)rand() / (double)RAND_MAX;
        bias = (double)rand() / (double)RAND_MAX;
        parse_file(file_path);
        std::cout << "File parsed, weights and bias randomized" << std::endl;
    }

double Perceptron::dot_val(const double &weight)
{
    double output = bias;

    for (size_t i = 0; i < 5; ++i)
        output += (input_layer[i] * weight);

    return output;
}

double Perceptron::sigmoid(const double &node)
{
    return (1 / (1 + exp(-node)))
}

void Perceptron::engage_perceptron()
{
    for (size_t i = 0; i < 7; ++i)
    {
        hidden_layer[i] = dot_val(hidden_weights[i]);
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