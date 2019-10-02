# Introduction to Machine Learning with Python
# Chapter 2: Supervised Learning
# 
import mglearn
import matplotlib.pyplot as plt

### Supervised Learning
## 2.1 Classification and Regression
# An easy way to distinguish between classification and regression tasks is to ask whether there is some kind of ordering 
# or continuity in the output. 
# If there is an ordering, or a continuity between possible outcomes, then the problem is a regression problem.

## 2.2 Generalization, Overfitting, and Underfitting

## 2.3 Supervised Machine Learning Algorithms
# 2.3.1 Some Sample Datasets
# An example of a synthetic two-class classification dataset is the forge dataset, which has two features. 
# Below is a scatter plot visualizing all of the data points in this dataset. 
# The plot has the first feature on the x-axis and the second feature on the y-axis. 
# As is always the case in in scatter plots, each data point is represented as one dot. 
# The color of the dot indicates its class, with red meaning class 0 and blue meaning class 1.

# mglearn.plots.plot_grid_search_overview()

# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)   # X[:, 0]      取所有行的第0个数据
                                                # X[:, 1]      取所有行的第1个数据    
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("\nX.shape:", X.shape)
