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
    initialize(input_layer, 4, "input");
    initialize(hidden_layer, 6);
    initialize(output_layer, 3, "output");
    std::cout << "\nFile parsed, weights and bias randomized\n\n";
}
// number function and operations
double Perceptron::dot_val(std::vector<std::map<std::string, double>> &layer)
{
    double output;
    for (size_t i = 0; i < layer.size(); ++i)
        output += layer[i]["value"] * layer[i]["weight"]; // output += val * weight
    return output;
}

double Perceptron::sigmoid(const double num)
{ 
    return (1 / (1 + exp(-num))); 
}

double Perceptron::relu(const double num)
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
    std::cout << "value, weight, bias" << std::endl;
    std::cout << "input_layer:" << std::endl;
    for (auto &node : input_layer)
        std::cout << "[value: " << node["value"] << ", weight: " << node["weight"] << "], ";
    std::cout << std::endl;
    std::cout << "hidden_layer:" << std::endl;
    for (auto &node : hidden_layer)
        std::cout << "[value: " << node["value"] << ", weight: " << node["weight"] << ", " << node["bias"] << "], ";
    std::cout << std::endl;
    std::cout << "output_layer:" << std::endl;
    for (auto &node : output_layer)
        std::cout << "[value: " << node["value"] << ", bias: " << node["bias"] << "], ";
    std::cout << std::endl;
}

void Perceptron::initialize(std::vector<std::map<std::string, double>> &layer, size_t size, std::string layer_type)
{
    for (size_t i = 0; i < size; ++i)
    {
        layer.push_back({});
        layer[i]["value"] = 0; // value
        if (layer_type != "output")
            layer[i]["weight"] = random_number(); // weight
        if (layer_type != "input")
            layer[i]["bias"] = random_number(); // bias
    }
}

void Perceptron::train(size_t epoch_init)
{
    epochs = epoch_init;
    while (epochs > 0)
    {
        std::cout << "Epoch " << epochs << "\n\n";
        
        epoch();

        --epochs;
    }
}

void set_y_val(std::string &y_str, double &y, const size_t i, std::array<double, 3> &expected)
{
        if (i < 30)
        {
            y = 0;
            y_str = "iris_setosa";
            expected = {1, 0, 0};
        }
        else if (i < 60)
        {
            y = 1;
            y_str = "iris_versicolor";
            expected = {0, 1, 0};
        }
        else if (i < 90)
        {
            y = 2;
            y_str = "iris_virginica";
            expected = {0, 0, 1};
        }
}

void Perceptron::epoch()
{
    // double correct;
    for (size_t i = 0; i < 90; ++i)
    {
        double y_hat, y;
        std::array<double, 3> expected;
        std::string y_str;
        set_y_val(y_str, y, i, expected);

        for (size_t j = 0; j < 4; ++j) // Initialize input layer values
            input_layer[j]["value"] = training_set[i][j];

        forward_prop(y, y_hat);
        back_prop(expected);

        // std::cout << i << " " << std::boolalpha << (y_hat == y) << ", y^: " << y_hat << " y: " << y << std::endl;
        // if (y == y_hat)
        //     correct += 1;
    }
    // std::setprecision(2);
    // std::cout << "Accuracy: " << (correct / 90) << "%\n\n";
}

void Perceptron::forward_prop(double y, double &y_hat)
{
    for (size_t i = 0; i < 6; ++i)
        hidden_layer[i]["value"] = sigmoid(dot_val(input_layer) + hidden_layer[i]["bias"]); // hidden node = dot product + bias
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

void Perceptron::back_prop(std::array<double, 3> expected)
{
    std::vector<double> output_errors;
    for (size_t i = 0; i < output_layer.size(); ++i)
    {
        output_errors.push_back(expected[i] - output_layer[i]["value"]);
        output_layer[i]["error"] = output_errors[i] * (output_layer[i]["value"] * (1 - output_layer[i]["value"]));
    }
    // output layer finished

    std::vector<double> hidden_errors;
    for (size_t i = 0; i < hidden_layer.size(); ++i)
    {
        double error = 0;
        for (auto &node : output_layer)
            error += (hidden_layer[i]["weight"] * node["error"]);
        hidden_errors.push_back(error);
    }
    for (size_t i = 0; i < hidden_layer.size(); ++i)
        hidden_layer[i]["error"] = hidden_errors[i] * (hidden_layer[i]["value"] * (1 - hidden_layer[i]["value"]));
    // hidden layer finished

    std::vector<double> input_errors;
    for (size_t i = 0; i < input_layer.size(); ++i)
    {
        double error = 0;
        for (auto &node : hidden_layer)
            error += (input_layer[i]["weight"] * node["error"]);
        input_errors.push_back(error);
    }
    for (size_t i = 0; i < input_layer.size(); ++i)
        input_layer[i]["error"] = input_errors[i] * (input_layer[i]["value"] * (1 - input_layer[i]["value"]));
    // input layer finished
}
