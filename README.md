# Workshop-Activity-on-Genetic-Algorithm-and-Neural-Network
# Aim:
To understand and implement the concepts of Genetic Algorithm (GA) and Neural Network (NN) for solving optimization and prediction problems.

# Objectives:

To learn the basic working principles of Genetic Algorithms.

To understand how Neural Networks can be trained and used for prediction.

To implement GA to optimize a problem or set of parameters.

To implement a simple Neural Network for classification or regression tasks.

To observe and analyze the output of GA and NN in Python.

# Program:

```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

def fitness_function(x):
    return x**2

def genetic_algorithm(pop_size=10, generations=20):
    population = np.random.randint(0, 31, size=pop_size)
    for gen in range(generations):
        fitness = np.array([fitness_function(ind) for ind in population])
        selected = population[np.argsort(fitness)][-pop_size//2:]
        offspring = []
        while len(offspring) < pop_size//2:
            parents = np.random.choice(selected, 2, replace=False)
            cross_point = np.random.randint(1, len(parents))
            child = int(str(parents[0])[:cross_point] + str(parents[1])[cross_point:])
            offspring.append(child)
        offspring = [child + np.random.randint(-2, 3) for child in offspring]
        population = np.concatenate([selected, offspring])
    best_individual = population[np.argmax([fitness_function(ind) for ind in population])]
    return best_individual

best_solution = genetic_algorithm()
print("Best solution from GA:", best_solution)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nn = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=42)
nn.fit(X_train, y_train)

predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print("Neural Network accuracy:", accuracy)

```

# Output:

<img width="1402" height="95" alt="image" src="https://github.com/user-attachments/assets/9a3faa4b-ec7b-4a01-8587-daea339299b9" />



# Result:

The Genetic Algorithm successfully found the optimal value of the function.

The Neural Network achieved high accuracy in classifying the Iris dataset.

The activity demonstrates the integration of GA for optimization and NN for predictive modeling.
