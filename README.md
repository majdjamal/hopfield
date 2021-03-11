# Hopfield Network

## Overview
Hopfield Network is an auto-associative memory. It is a neural network that can remember data and reduce noise. This network is similar to a human brain that can remember objects and recall them from different states. For example, if our friends dress differently, we would still recognize them.
This is an implementation of the Hopfield Network, and the architecture is demonstrated in figure 1.

<img src="https://i.ibb.co/B61C250/hn-removebg-preview.png" width="350" height="350">
(Figure 1. The architecture of a Hopfield Network.)

## Setup
This experiment requires NumPy and Matplotlib.

## Demonstration
We train the model with an image of a heart, as seen in figure 2.

<img src="https://i.ibb.co/XYffQxS/original.png" width="350" height="350">
(Figure 2. Original data point. An image of a heart.)

We add noise to the heart and re-create it to its original state using a Hopfield Network. The result is demonstrated in figure 3.

<img src="https://media.giphy.com/media/umIi973GvKxKcHvGXn/giphy.gif" width="350" height="350">
(Figure 3. Adding noise to the original image and using Hopfield Network to re-create it to its original state. The image is re-created using asynchronous update and each state is annotated with the current step.)

## Testing

To test the model, navigate to the repository in your terminal and type:

```bash
python experiment.py
```
