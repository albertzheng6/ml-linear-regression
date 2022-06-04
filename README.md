# Machine-Learning-Using-Linear-Regression
This code demonstrates the basics of machine learning by implementing algorithms for linear regression of multiple variables. Given a training set of data with any number of features, it can learn parameters that fit an optimal linear function to the data and make predictions. While this code can work for any number of variables, linear regression is best used when there are only a few features because the data can be easily plotted, and because neural networks are better "in general" for data sets with a high number of features.

Steps to implement linear regression using this code:
1. Obtain raw initial data from somewhere and put it into a csv file.
2. Modify the data by adding the x_0 feature. Also, feature normalize the data if necessary. Note that this code separates the initial X and feature normalized X into two distinct variables.
3. Choose a learning rate by plotting the speed of convergence of the error function with respect to the number of iterations.
4. Depending on the situation, apply a gradient descent algorithm to return the optimal parameters that minimizes the error function.
5. Plot linear regression using the optimal parameters.

Reference
- xj_i is the ith element of the jth training example
- Training examples range from x1 to xm
- Each training example has elements ranging from x_0 to x_n+1
- Use numpy 2d arrays as preferred data structure whenever possible
- If learning rates are too large, J can diverge and blow up, resulting in really large values the computer cannot comprehend
- Csv files MUST include headers (x_0, x_1, ..., y) or else the first training ex will be cut off

Data
- Data Set 1: linReg1.csv, do not do feat norm
- Data Set 2: linReg2.csv, requires feat norm

Examples

Testing different learning rates for data set 1
![Figure_2](https://user-images.githubusercontent.com/106856325/171986347-c2cd9df1-be5a-4bca-a167-e4b1b1a9de29.png)

Plotting linear functions to predict data set using the learned parameters
![Figure_1](https://user-images.githubusercontent.com/106856325/171986069-04d53806-2c37-4858-8e1f-cfa8a276ec6e.png)

Visualizing data set 2
![Figure_3](https://user-images.githubusercontent.com/106856325/171986797-762ad215-e493-4b7a-bf33-bce8c7e9356d.png)
![Figure_4](https://user-images.githubusercontent.com/106856325/171986800-63eca4bd-30b5-4baf-b628-ccd3b9f6ed97.png)

Testing different learning rates for data set 2
![Figure_5](https://user-images.githubusercontent.com/106856325/171986821-25dd6164-92ab-4a5a-b528-2624ed030a7d.png)

