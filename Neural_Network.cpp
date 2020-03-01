#include "Neural_Network.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

Neural_Network::Neural_Network(std::string file_path)
{
    parse_file(file_path);
    initialize(input_layer, 4, "input");
    initialize(hidden_layer, 6);
    initialize(output_layer, 3, "output");
    std::cout << "\nFile parsed, weights and bias randomized\n\n";
}
// number function and operations
double Neural_Network::dot_val(std::vector<std::map<std::string, double>> &layer)
{
    double output;
    for (auto node : layer)
        output += node["value"] * node["weight"]; // output += val * weight
    return output;
}

double Neural_Network::sigmoid(const double num)
{ 
    return (1 / (1 + exp(-num))); 
}

double Neural_Network::random_number() 
{ 
    return (double)rand() / (double)RAND_MAX; 
}

void Neural_Network::parse_file(std::string path)
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

void Neural_Network::display()
{
    // std::setprecision(2);
    // std::cout << "value, weight, bias" << std::endl;
    // std::cout << "input_layer:" << std::endl;
    // for (auto &node : input_layer)
    //     std::cout << "[value: " << node["value"] << ", weight: " << node["weight"] << "], ";
    // std::cout << std::endl;
    // std::cout << "hidden_layer:" << std::endl;
    // for (auto &node : hidden_layer)
    //     std::cout << "[value: " << node["value"] << ", weight: " << node["weight"] << ", bias: " << node["bias"] << ", error " << node["error"] << "], ";
    // std::cout << std::endl;
    // std::cout << "output_layer:" << std::endl;
    for (auto &node : output_layer)
        std::cout << "[value: " << node["value"] << ", bias: " << node["bias"] << ", error: " << node["error"] << "], ";
    std::cout << std::endl;
}

void Neural_Network::initialize(std::vector<std::map<std::string, double>> &layer, size_t size, std::string layer_type)
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

void Neural_Network::train(size_t epoch_init)
{
    const size_t count = epoch_init;
    while (epoch_init > 0)
    {
        std::cout << "Epoch " << 1 + (count - epoch_init) << "\n\n";
        epoch();
        display();
        --epoch_init;
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

void Neural_Network::epoch()
{
    size_t correct = 0;
    for (size_t i = 0; i < 90; ++i)
    {
        double y_hat, y;
        std::array<double, 3> expected {0, 0, 0}, actual {0, 0, 0};
        std::string y_str;
        set_y_val(y_str, y, i, expected);

        for (size_t j = 0; j < 4; ++j) // Initialize input layer values
            input_layer[j]["value"] = training_set[i][j];

        forward_prop(y_hat, actual);
        back_prop(expected);
        update_network(0.125);
    }
}

void Neural_Network::forward_prop(double &y_hat, std::array<double, 3> &expected)
{
    for (auto &node : hidden_layer)
        node["value"] = sigmoid(dot_val(input_layer) + node["bias"]);
    for (size_t i = 0; i < output_layer.size(); ++i)
    {
        output_layer[i]["value"] = sigmoid(dot_val(hidden_layer) + output_layer[i]["bias"]);
        if (output_layer[i]["value"] > y_hat)
            y_hat = (double)i;
    }
}

void Neural_Network::back_prop(std::array<double, 3> expected) // work backwards from the output layer
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

void Neural_Network::update_network(double l_rate)
{
    std::vector<double> input_vals;
    for (auto &node : input_layer)
        input_vals.push_back(node["value"]);
    for (auto &node : hidden_layer)
    {
        for (size_t i = 0; i < input_vals.size(); ++i)
            input_layer[i]["weight"] += l_rate * node["error"] * input_vals[i];
        node["bias"] += l_rate * node["error"];
    }
    // updated weights and bias for hidden layer
    std::vector<double> hidden_vals;
    for (auto &node : hidden_layer)
        hidden_vals.push_back(node["value"]);
    for (auto &node : output_layer)
    {
        for (size_t i = 0; i < hidden_vals.size(); ++i)
            node["weight"] += l_rate * node["error"] * hidden_vals[i];
        node["bias"] += l_rate * node["error"];
    }
    // updated weights and bias for output layer
}

