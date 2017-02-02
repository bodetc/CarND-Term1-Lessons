"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)
g = Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], forward_pass(g, sorted_nodes)))
