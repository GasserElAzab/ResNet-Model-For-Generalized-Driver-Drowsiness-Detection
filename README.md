# Generalized-Driver-Drowsiness-Detection

There are two available versions for this model, a jupyter notebook for better visualization for each step, and a main file that can be easier for trying differnet architecture. 

The main files two argument:-

1:- a list of numbers, the length of the list is the number of groups, while each number is the number of ResNet blocks per each element
2:- learning rate for Adam optimizer

As an example, to train your own network:-
python main.py -b 1 1 1 -lr 0.001

Which is training a network of 3 groups, each group has one ResNet block. The Adam optimizer is going to learn with a rate of 0.001
