# Workshop-Activity-on-Genetic-Algorithm-and-Neural-Network
# Aim:
To study and implement optimization techniques using Genetic Algorithm (GA) and Artificial Neural Network (ANN) in Python for solving computational problems through learning and evolution-based approaches.

# Objectives:

To understand the working principles of Genetic Algorithms and Neural Networks.

To implement a GA to find the optimal set of parameters that minimize the output error for a given function.

To develop a Neural Network model capable of learning input-output relationships through training and backpropagation.

To compare how both approaches achieve optimization—GA through evolution and ANN through learning.

# Program:

#  Genettic Algorithm :
```
!pip install pygad

import pygad
import numpy as np

X = np.array([1, 2, 4, -2])
target_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    y = np.sum(solution * X)
    fitness = 1.0 / (1.0 + abs(target_output - y))
    return fitness

num_genes = len(X)

ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=2,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=num_genes,
    init_range_low=-10,
    init_range_high=10,
    mutation_percent_genes=20,
    mutation_type="random",
    crossover_type="single_point"
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
predicted_output = np.sum(solution * X)

print("Best Weights (w1, w2, w3, w4):", solution)
print("Predicted Output y =", predicted_output)
print("Target Output =", target_output)
print("Error =", abs(target_output - predicted_output))

```

# Output:

<img width="1259" height="98" alt="image" src="https://github.com/user-attachments/assets/cde617df-e296-44fd-9e85-7bfebf8df00f" />


# Neural Network :
```
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# Build the MLP model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu')) # Hidden layer with ReLU
model.add(Dense(1, activation='sigmoid')) # Output layer with Sigmoid
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X, y, epochs=10, verbose=0)
# Evaluate and test
_, accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy * 100:.2f}%")
predictions = model.predict(X)
predictions = np.round(predictions).astype(int)
for i in range(len(X)):
   print(f"Input: {X[i]} => Predicted Output: {predictions[i][0]}, Actual Output: {y[i][0]}")
```
# Output :

<img width="872" height="164" alt="image" src="https://github.com/user-attachments/assets/f455d257-2301-4969-b19a-74f0dc01d69d" />


# Result:

The Genetic Algorithm successfully optimized the solution by evolving the parameters to achieve the desired output with minimal error.

The Neural Network effectively learned the input-output relationship through iterative training, reducing prediction error and accurately mapping inputs to outputs.
