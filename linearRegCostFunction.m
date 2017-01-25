function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n= size(theta,1);

J = 0;
grad = zeros(size(theta));


predictions= X*theta;
J= (1/(2*m))* sum((predictions - y).^2) + ((lambda/(2*m))*sum(theta(2:n,1).^2));

grad  = (1/m)*(((predictions - y)' * X)');

grad(2:n,1)=grad(2:n,1) + (lambda/m)*(theta(2:n,1));








% =========================================================================

grad = grad(:);

end
