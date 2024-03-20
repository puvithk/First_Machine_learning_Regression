# First_Machine_learning_Regression
Importing Libraries: The code imports necessary libraries like NumPy, pandas, and Matplotlib for numerical computations, data handling, and visualization.

#Function Definitions:

#compute_cost: 
Computes the cost function for the given weights, biases, input features, and target values.
train: Performs training of the model using gradient descent, updating weights and biases iteratively for a certain number of epochs.
predict: Predicts the target values based on the trained weights and biases.
#readcsv:
Reads the data from a CSV file containing columns for income, savings, debt, and credit values.
dydx: Computes the partial derivatives of the cost function with respect to weights and bias.
Main Execution:

Reads the CSV file containing the data.
Sets the number of epochs for training.
Initializes weights, learning rate (alpha), and runs the training process.
Predicts the target values for the input data.
Plots the cost function over epochs and the predicted versus actual values.
README:

Simple Linear Regression Model for Credit Prediction
This Python script implements a simple linear regression model to predict credit values based on income, savings, and debt. Here's a brief overview of the components:

Dependencies: Ensure you have NumPy, pandas, and Matplotlib installed in your Python environment.
Data: The model requires a CSV file named PuvithMl_Credit.csv with columns for income, savings, debt, and credit values.
Functionality:
compute_cost: Computes the cost function for the model.
train: Trains the model using gradient descent.
predict: Predicts credit values based on input features.
readcsv: Reads the data from the CSV file.
dydx: Computes partial derivatives of the cost function.
plot_cost: Plots the cost function over epochs and the predicted versus actual credit values.
Instructions for Use:
Ensure the PuvithMl_Credit.csv file is in the same directory as the script.
Adjust the epochs variable to set the number of training iterations.
Run the script to train the model and visualize the results.
Interpret the plotted graphs to analyze the model's performance.