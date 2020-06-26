Choosing the right architecture for your deep learning model can drastically change the results from training. Using too few neurons can lead to the model not finding complex relationships in the data, whereas using too many neurons can lead to an overfitting effect.



One important factor to note is that with tabular data, most of the time one hidden layer is enough.



However, how do you pick the number of neurons in that layer?

The choice of the number of neurons in your data is related to the degrees of freedom in your dataset and how many degrees of freedom you want to allow in your model.

One formula found during research of this problem is:

$$ N_{h} = \frac{N{s}}{\alpha (N_{i} + N_{o})} $$

$N_{h}$ is the number of neurons, $N_{s}$ the number of training samples $N_{i}$ represents the number of input neurons, $N_{o}$ is the number of output neurons and $\alpha$ is a hyperparameter to be selected.

The degrees of freedom in your dataset is $N_{s}(N_{i} + N_{o})$, and the aim is to limit the number of free parameters in your model to a small portion of the degrees of freedom in your data. If your data is a good representation, this should allow the model to generalise well.

It cannot overfit as the degrees of freedom are limited to a number proportional to the degrees of freedom of the dataset.

An $\alpha$ value of 2-10 is recommended, and it is best to loop through to find an optimal. An easy way to understand this is to let $\beta = \frac{1}{\alpha}$ and increasing $\beta$ represents an increasing complexity. A more nonlinear problem will need a larger $\beta$.
