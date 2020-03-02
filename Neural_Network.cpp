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
    initialize({4, 6, 3});
    std::cout << "\nFile parsed, weights and bias randomized\n\n";
}
// number function and operations
double Neural_Network::dot_val(std::vector<double> val,std::vector<double> weights)
{
    double output;
    for (size_t i = 0; i < weights.size(); ++i)
        output += val[i] * weights[i];
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
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "values:\n";
    for (size_t i = 0; i < values.size(); ++i)
    {
        std::cout << "layer " << i << "\n[ ";
        for (size_t j = 0; j < values[i].size(); ++j)
            std::cout << values[i][j] << " ";
        std::cout << " ]\n";
    }

    std::cout << std::endl;

    std::cout << "bias:\n";
    for (size_t i = 0; i < bias.size(); ++i)
    {
        std::cout << "layer " << i << "\n[ ";
        for (size_t j = 0; j < bias[i].size(); ++j)
            std::cout << bias[i][j] << " ";
        std::cout << " ]\n";
    }

    std::cout << std::endl;

    std::cout << "errors:\n";
    for (size_t i = 0; i < errors.size(); ++i)
    {
        std::cout << "layer " << i << "\n[ ";
        for (size_t j = 0; j < errors[i].size(); ++j)
            std::cout << errors[i][j] << " ";
        std::cout << " ]\n";
    }

    std::cout << std::endl;

    std::cout << "weights:\n";
    for (size_t i = 0; i < weights.size(); ++i)
    {
        std::cout << "layer " << i << "[\n";
        for (size_t j = 0; j < weights[i].size(); ++j)
        {
            std::cout << "node [ ";
            for (size_t k = 0; k < weights[i][j].size(); ++k)
                std::cout << weights[i][j][k] << " ";
            std::cout << " ]\n";
        }
        std::cout << "]\n";
    }

    std::cout << std::endl;
}

void Neural_Network::initialize(std::vector<size_t> layers)
{
    for (size_t i = 0; i < layers.size(); ++i)
    {
        std::vector<double> v, b, e;
        std::vector<std::vector<double>> w;
        //initializing the nodes in the layers
        for (size_t j = 0; j < layers[i]; ++j)
        {
            v.push_back(0);
            b.push_back(random_number());
            e.push_back(1);
            std::vector<double> inner_w;
            if (i != 0) // checking if the current layer is the input
                for (size_t k = 0; k < layers[i - 1]; ++k) // adding weights to the current layer to the amount of nodes in the next layer
                    inner_w.push_back(random_number()); // adding a weight to the current layer for a node in the next layer
            w.push_back(inner_w);
        }
        values.push_back(v);
        bias.push_back(b);
        errors.push_back(e);
        weights.push_back(w);
        ++net_size;
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

void set_y_val(std::string &y_str, double &y, const size_t i, std::vector<double> &expected)
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
    double cost;
    for (size_t i = 0; i < 90; ++i)
    {
        std::vector<double> expected {0, 0, 0}, actual {0, 0, 0};
        std::string y_str;
        double y_hat, y;
        set_y_val(y_str, y, i, expected);
        for (size_t j = 0; j < values[0].size(); ++j) // Initialize input layer values
            values[0][j] = training_set[i][j]; // value[0] is the input layer, j is the node

        forward_prop(y_hat, actual, cost);
        actual[(size_t)y_hat] = 1;
        if (expected == actual)
            ++correct;
        back_prop(expected);
        update_network(0.125);
    }
    std::cout << "cost: " << cost / 90 << std::endl;
}

void Neural_Network::forward_prop(double &y_hat, std::vector<double> &expected, double &cost)
{
    for (size_t i = 1; i < net_size - 1; ++i) // looping through every layer except the first and last
        for (size_t j = 0; j < values[i].size(); ++j) // looping through every node in the current non input/output layer
            values[i][j] = sigmoid(dot_val(values[i - 1], weights[i][j]) + bias[i][j]); // assigning node j of layer i a sigmoided val that is the dotval + the associated bias
    for (size_t i = 0; i < values[net_size - 1].size(); ++i) // looping through the ouptut layer
    {
        values[net_size - 1][i] = sigmoid(dot_val(values[net_size - 2], weights[net_size - 1][i]) + bias[net_size - 1][i]);
        if (values[net_size - 1][i] > y_hat)
            y_hat = (double)i;
        cost += pow(values[net_size - 1][i] - expected[i], 2);
    }
}

double transfer_derivitive(double num)
{
    return num * (1 - num);
}

void Neural_Network::back_prop(std::vector<double> expected) // work backwards from the output layer
{
    std::vector<double> output_errors;
    for (size_t i = 0; i < errors[net_size - 1].size(); ++i) // looping through the output layer
    {
        output_errors.push_back(expected[i] - values[net_size - 1][i]);
        errors[net_size - 1][i] = output_errors[i] * transfer_derivitive(values[net_size - 1][i]);
    } // output layer finished
    for (size_t i = net_size - 2; i >= 0; --i) // looping through the non output layers backwards
    {
        std::vector<double> layer_errors;
        for (size_t j = 0; j < errors[i].size(); ++j) // looping through the current layer's nodes
        {
            double error = 0;
            for (size_t k = 0; k < weights[i + 1][j].size(); ++k) // looping through the current set of weights
                error += weights[i + 1][j][k] * errors[i][j];
            layer_errors.push_back(error);
        }
        for (size_t j = 0; j < layer_errors.size(); ++j)
        {
            errors[i][j] = layer_errors[j] * transfer_derivitive(values[i][j]);
            std::cout << layer_errors[j] * transfer_derivitive(values[i][j]) << " ";
        }
        std::cout << "testing ";
    }
}

void Neural_Network::update_network(double l_rate)
{
    for (size_t i = 0; i < net_size; ++i)
        for (size_t j = 0; j < weights[i].size(); ++j)
        {
            for (size_t k = 0; k < weights[i][j].size(); ++k)
                weights[i][j][k] += l_rate * errors[i][j] * values[i - 1][j];
            bias[i][j] += l_rate * errors[i][j];
        }
}
