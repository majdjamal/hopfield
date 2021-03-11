
from data import heart, broken
import numpy as np
from hopfield import Hopfield
import matplotlib.pyplot as plt

h, w = heart.shape

hpd = Hopfield()

# Reshaping the data to match requirements of Hopfield.
full = heart.flatten().reshape(-1,1).T
brkn = broken.flatten().reshape(-1,1).T

hpd.fit(full)

recreated_heart = hpd.sequential_predict(brkn, 800)

plt.imshow(recreated_heart.reshape(h,w), cmap="Reds")
plt.show()
