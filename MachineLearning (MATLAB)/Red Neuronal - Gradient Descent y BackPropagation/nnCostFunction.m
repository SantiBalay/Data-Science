function [J ,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%%propagation

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';

a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1), a2];

z3 = a2*Theta2';
a3 = sigmoid(z3);

hx = a3; %listo

yv = [1:num_labels] == y;

theta_1 = Theta1(:,2:end);
theta_2 = Theta2(:,2:end);

%J = (1/m) * sum(sum((-yvec.*log(hx))-((1.-yvec).*log(1-hx)),2)); sin reg ->  %hago todo row*row, y los sumo al final.

J = (1/m) * sum(sum((-yv.*log(hx))-((1.-yv).*log(1-hx)),2)) + (lambda/(2*m)) * (sum(sum(theta_1.^2,2)) + sum(sum(theta_2.^2,2)));

%k = size(unique(y),1); %numero de clases

error3 = a3 - yv;
aux = Theta2(:,2:end);
error2 = error3*aux .* sigmoidGradient(z2);

DeltaTheta1 = error2'*a1;
DeltaTheta2 = error3'*a2;

m1 = size(theta_1,1);
m2 = size(theta_2,1);

regTerm1 = [zeros(m1,1),theta_1];
regTerm2 = [zeros(m2,1),theta_2];

Theta1_grad = 1/m * DeltaTheta1 + lambda/m * regTerm1;
Theta2_grad = 1/m * DeltaTheta2 + lambda/m * regTerm2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end