%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

fprintf('Loading Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
training = sel(1:round(m*8/10));
testing = sel((round(m*8/10)+1):end);

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

iters = [10 50 400];
lambdas = [0 1 100];

predTrains = zeros(length(lambdas), length(iters));
predTests = zeros(length(lambdas), length(iters));

for i_iter = 1:length(iters),
	for i_lambda = 1:length(lambdas),

iter = iters(i_iter);
lambda = lambdas(i_lambda);
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', iter);

%  You should also try different values of lambda
%lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
								   X(training, :), ...
								   y(training), ...
								   lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

fprintf('\nLambda: % 8.4f Iter: % 6i\n', lambda, iter);

predTrain = predict(Theta1, Theta2, X(training, :));
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predTrain == y(training))) * 100);
predTrains(i_lambda, i_iter) = mean(double(predTrain == y(training))) * 100;

predTest = predict(Theta1, Theta2, X(testing, :));
fprintf('\nTesting Set Accuracy: %f\n\n', mean(double(predTest == y(testing))) * 100);
predTests(i_lambda, i_iter) = mean(double(predTest == y(testing))) * 100;

end
end

fprintf('    Lambda  Iter ');
for iter = iters,
	fprintf(' % 6i ', iter);
end
fprintf('\n');
for i_lambda = 1:length(lambdas),
	lambda = lambdas(i_lambda);
	printf('% 10.4f ', lambda);
	printf('Train ');
	for i_iter = 1:length(iters),
		iter = iters(i_iter);
		printf(' % 6.2f%%', predTrains(i_lambda, i_iter))
	end
	printf('\n');
	printf('           ');
	printf('Test  ');
	for i_iter = 1:length(iters),
		iter = iters(i_iter);
		printf(' % 6.2f%%', predTests(i_lambda, i_iter))
	end
	printf('\n');
end
