---
layout: post
title: Machine and Deep Learning for Stock Price Prediction
mathjax: true
project: true
description: Comparison of treating stock price prediction as a classification or regression problem. This report formed part of my 3rd research project at the University of Leeds.
---

<div class="message">
  The objective of this report is to compare the use of classification models and regression models from Machine and Deep Learning are used to predict the price trend of a stock. For the corresponding Github repository click <a href="https://github.com/henriwoodcock/Stock-Price-Prediction">here</a>. You can also view the <b>full</b> report which formed part of my degree as a pdf <a href="{{ site.url }}/downloads/Henri Woodcock - Math3001.pdf">here</a>.
</div>

The objective of this report is to compare the use of classification models and regression models from Machine and Deep Learning are used to predict the price trend of a stock. In this report the stock that is used is the Koninklijke KPN N.V. (KPN) stock from the Amsterdam Exchange (AEX) Index. Technical features and the price of 4 other stocks from the AEX Index are used to train 12 models. The 12 models are 6 regression and 6 classification models, these are Support Vector Machines/Regression, Feedforward Neural Networks, Recurrent Neural Networks and auto-encoded variants of those models. Each model trained to tackle the classification problem and the regression problem. It is found on average that classification models are able to predict price movements better, however this comes at the cost of only achieving higher results when the models are trained on more data, regression models actually outperformed classification models when less training data is used. However, it is also found that regression models can accurately predict the actual price with an average mean absolute error as low as 0.0867 which could be of more value than trend prediction to certain investors. Specific models are then compared Overall it is concluded that classification models work best in predicting trend with the best trend prediction model being a Recurrent Neural Network which predicted the correct trend 55.46%.

# Contents
- [Introduction](#introduction)
- [Modern Best Practices](#modern-best-practices)
  - [Transfer Learning](#transfer-learning)
  - [Feature Loss](#feature-loss)
  - [Pixel Shuffle](#pixel-shuffle)
  - [Progressive Resizing](#progressive-resizing)
- [Experiment](#experiment)
  - [Models](#models)
  - [Measures](#measures)
  - [Results](#results)
  - [Discussion](#discussion)
- [Conclusion](#conclusion)
  - [Further Research](#further-research)
- [References](#references)

# Introduction
Time series prediction has always been of interest especially in finance, the ability to be able to predict the price movements of a stock correctly can lead to an investor being able to make profits with no limit. Machine and Deep learning are growing fields in computer science, in which artificial learning techniques are used to find patterns in data. Deep learning can find relationships between asset returns no matter how complex and nonlinear [1], this can lead to predictions being made from deeper meaningful relationships between assets returns. This has led to an increase in the interest in Machine and Deep Learning techniques in finance, this report will be explaining Machine and Deep Learning from a financial time series perspective to discuss which models work well for time series prediction and how models are trained before finally comparing different models for stock price prediction.

# K-Fold Validation
One way to make sure that the model is not overfit to the data is to use a method called cross-validation. $K$-fold cross validation shuffles the data so it’s in a random order and then splits the data into K subsets. Label these $F\_{1},F\_{2},...,F\_{K}$. Then for $i = 1,...,K$, keep $F\_{i}$ as the testing set and the other K − 1 folds for training. Train K different models and compare the test results. The best model can now be chosen or an average of all the models can be made. This means the model can perform well on the entire sample without overfitting any particular training set.

Time series data cannot be shuffled because it is time indexed, and so shuffling the data could mean the model is trained on data thats indexed after the data its tested on. To overcome this, [27] compared a few different methods of K-fold validation in time series data. The method that will be used in the experiment is as follows: split the data into K + 1 subsets, then the folds are:
```python
Fold 1: Training [1], test [2]
Fold 2: Training [1, 2], test [3]
...
Fold K: Training [1, 2, ..., K], test [K + 1]
```
This method also allows comparison of how dependant the performance of a model is on the size of the data.

# Experiment
This experiment compares different algorithms for predicting a stocks price and its trend, it also compares the use of classification and regression algorithms for predicting the trend while also analysing other problems like the importance of the amount data and the use of auto-encoders.
This section will start with a literature review to discuss developments in financial time series prediction, then discuss the models to be compared and the tools used to create them, followed by an explanation of the conducted experiment and finally a discussion on the results.

## Literature Review
The 21st century has stimulated a huge growth in the accessibility of data and the amount of data collected. This has brought around a rise in Machine and Deep Learning. Machine Learning algorithms can automatically detect patterns in data and in this era, with vast amounts of data, algorithms that can automatically analyse data are needed.

In finance, investors can now download years of data for hundreds of securities, from stock prices to accounting ratios, leading to the use of Machine and Deep Learning techniques in finance. Stock price prediction was (and still is to an extent) dominated by the use of statistical time series models like ARIMA, both [28],[29] both found that the ARIMA method is still effective, with [28] getting a mean absolute percentage error as low as 21% for the Nifty 50 index. However, [29] showed that ARIMA models can only produce satisfactory results with short-term predictions – this is also supported by the results in [30].
The use of Machine and Deep Learning algorithms to predict financial time series is growing. [13] used SVMs to predict trend movements and was able to achieve a hit ratio of 54% on the Korean stock index over a far greater time frame of 581 steps (20% of the data) instead of a standard 10-20 steps used in ARIMA research. [13] also showed the sensitivity of the SVM parameters in this context. This was then extended by [14] where a denoising auto-encoder was used prior to SVM algorithm and managed to achieve a hit ratio of 67.2%. However, this was tested on far fewer time steps: [31] used a hybrid ARIMA and SVM algorithm which achieved better results than ARIMA, but still was only performed on so few time steps that long-term accuracies could not be seen.
A slightly different approach was performed by [32], which – inspired by recent development in Convolutional Neural Networks (CNNs) (and the AlphaGo project) for image recognition – converted financial time series data into 2D images for a CNN to classify the trend of the Taiwan stock exchange. A hit rate of 57.88% on one of their examples was achieved, but concluded that more data could reduce the difficulty of classification. [33] compared the use of a simple RNN to a LSTM model and found that a simple RNN had a trend classification accuracy of 75%, 1% higher than an LSTM. He also found that implementing strategy used by following trend prediction can lead to profits exceeding buy and hold strategy on the S&P500 index with a prediction on 480 steps (20% of the data).

On the regression side, [34] found that RNNs outperform in prediction compared to feed-forward neural networks. However, RNNs have a far greater training time, which could be a factor in model selection. [35] managed to achieve a mean absolute percentage error as low as 0.71% when predicting 60 future values after training on 1000 examples.

One final important paper is [36] which creates a huge deep learning model trained on a large dataset of “billions of orders and transactions” [36] for 1000 US stocks. This gives remarkable results in which the authors believe they have uncovered evidence “universal price formation mechanism”[36] by showing their model can accurately predict prices of unseen stocks after being trained on multiple stocks, outperforming models trained on the singular stock. [36] concludes that creating one universal model is less costly than training thousands of stock specific models.

## Models for Comparison
The experiment will use 3 classification and 3 regression models as well as auto-encoded variants of these models and so there are 12 models in total.
The models are: SVM (and SVR), FFN (feedforward network), RNN (which uses an LSTM layer). All 12 models have been written in Python using Spyder 3.3.3 [37]. To implement the SVM and SVR models, scikit-learn library [38] is used, and Keras [39] is used to implement the FFN, RNN and auto-encoder.

## Data
Data used is daily closing price of KPN, AKZA, RAND, VPK, HEIA and the KPN trading volume from the 4th Febuary 2008 to 16th January 2019 in the Amsterdam stock exchange, taken from Yahoo Finance [40], as well as technical indicators applied to the KPN price and trading volume which can be found in Appendix E. The use of multiple stock prices to predict one is inspired by [36]. The use of techincal indicators was inspired by [13, 14] who were able to achieve high results with just the use of techincal indicators.

For the classification data set, the output, $y\_{i}$ at each time step was created as follows ($S\_{t}$ = closing price at time $t$):

$$
y_{i} =
    \begin{cases}
      1, & \text{if}\ S_{i+1} > S_{i} \\
      0, & \text{otherwise}
    \end{cases}
$$
where $S_{t}$ is the closing price at time $t$. So the classification problem is attempting to predict the trend of the next timestep. For the regression problem, $y_{i} = S_{i+1}$ and so the regression models are attempting to predict the actual real value of the stock price at the next timestep.

The data is then split into $6$ subsets and setup like K-fold validation for time series discussed in the [previous section](#k-fold-validation), with each subset being 467 time steps.

The input data ($\mathbf{X}$) for each fold is standardised, this is so all features have a mean of $0$ and variance of $1$, this makes training perform better as the algorithms are not skewed by features which are alot larger than others, for example volume traded $\gg$ close price (the data before preprocessing can be seen in Appendix~\ref{a:data}. This is done by:
$$
\tilde{\mathbf{x}} = \frac{\mathbf{x} - \mu\_{x}}{\sigma\_{x}}
$$
The testing data input is standardised using the same $\mu\_{x}$ and $\sigma_{x}$ as the training data so that model knows how to use the data with respect to the training data.

## Performance Measures
The models will compare the models using 3 different prediction performances. The first measure is the _hit ratio_. This measures how many times the models predict the next trend correctly. For regression models, the output will be converted into a trend by:
$$
\text{trend}\_{i} =
    \begin{cases}
      1, & \text{if}\ \hat{y}\_{i} > S\_{i} \\
      0, & \text{otherwise}
    \end{cases}
$$
i.e. a $1$ is the model predicts that the next days closing price is higher than todays closing price. The hit ratio is calculated as follows:
$$
\text{hit}\_{i} =
    \begin{cases}
      1, & \text{if}\ \text{trend}\_{i} = \text{actual trend}\_{i} \\
      0, & \text{otherwise}
    \end{cases}
$$
$$
\text{hit ratio} = \frac{1}{m}\sum\_{i=1}^{m} \text{hit}\_{i}
$$
where $m$ is the number of testing samples.

The other two measures are the mean squared error (MSE) and the mean absolute error (MAE):
$$
\text{MSE} = \sum\_{i=1}^{m}(y\_{i} - \hat{y}\_{i})^2
$$
$$
\text{MAE} = \sum\_{i=1}^{m} |y\_{i}-\hat{y}\_{i}|
$$
both are used as it can allow for comparison with anomalies. The MSE weights anomalies high and so if a model has a high MSE it can be seen it had a few anomaly results in which the distance from the actual value was high. This is important in stock price prediction because if there is a lot of anomalies the model is risky and should not be used as it could risk investments.


## Results and Discussion

### MAE Table
| __MAE__        | Fold 1    | Fold 2    | Fold 3      | Fold 4             | Fold 5             |
|------------|-----------|-----------|-------------|--------------------|--------------------|
| SVM        | ﻿0.5053533 | ﻿0.4775161 | ﻿0.4946467   | ﻿0.4860813704496788 | ﻿0.4925053533190578 |
| SVR        | ﻿0.3973456 | ﻿2.3802462 | ﻿0.4000280   | 0.133773577        | ﻿0.1329901634655249 |
| MLP        | 0.4998837 | 0.5122232 | 0.49999425  | 0.49633452         | 0.49297398         |
| MLP Reg    | 0.5839373 | 1.2310393 | 0.0857462   | 0.05617182         | 0.02886502         |
| RNN        | 0.4967482 | 0.4985013 | 0.50321287  | 0.5078179          | 0.4868381          |
| RNN Reg    | 0.388919  | 1.5988204 | 0.53422076  | 0.07032181         | 0.09193794         |
| AE SVM     | 0.4732334 | 0.496788  | 0.466880942 | 0.464668094        | 0.488222698        |
| AE SVR     | 0.4159817 | 3.0418163 | 1.10663284  | 0.552469041        | 1.192020797        |
| AE MLP     | 0.4953396 | 0.4956971 | 0.49009582  | 0.49047193         | 0.4952373          |
| AE MLP Reg | 0.3251435 | 2.9395235 | 0.700055    | 0.46293378         | 0.6654148          |
| AE RNN     | 0.4978627 | 0.4968552 | 0.5008697   | 0.4987097          | 0.49928015         |
| AE RNN Reg | 0.5420215 | 2.8771923 | 0.79907376  | 0.35173625         | 0.58343124         |

### MSE Table
| __MSE__        | Fold 1      | Fold 2      | Fold 3      | Fold 4      | Fold 5      |
|------------|-------------|-------------|-------------|-------------|-------------|
| SVM        | 0.505353319 | 0.47751606  | 0.494646681 | 0.48608137  | 0.492505353 |
| SVR        | 0.218232737 | 9.223313087 | 0.24325343  | 0.027218199 | 0.024403618 |
| MLP        | 0.26779357  | 0.33536112  | 0.2816803   | 0.25790796  | 0.27341437  |
| MLP Reg    | 0.3979998   | 2.67062     | 0.01061037  | 0.00535633  | 0.00140317  |
| RNN        | 0.25128642  | 0.34675285  | 0.32399637  | 0.30122712  | 0.25336137  |
| RNN Reg    | 0.17189465  | 4.643264    | 0.32230368  | 0.00817965  | 0.01458676  |
| AE SVM     | 0.473233405 | 0.496788009 | 0.466809422 | 0.464668094 | 0.488222698 |
| AE SVR     | 0.267383649 | 11.53000469 | 2.463742763 | 0.454032826 | 1.995170974 |
| AE MLP     | 0.2626788   | 0.26310447  | 0.2624369   | 0.26902062  | 0.27926183  |
| AE MLP Reg | 0.16338402  | 10.861558   | 0.6763687   | 0.28178778  | 0.5230367   |
| AE RNN     | 0.25046968  | 0.24803352  | 0.2522605   | 0.24930151  | 0.24994816  |
| AE RNN Reg | 0.47966784  | 10.426144   | 0.8275157   | 0.16618541  | 0.41073775  |

### Hit Ratio Table
| __HITS__       | Fold 1              | Fold 2              | Fold 3              | Fold 4              | Fold 5              |
|------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| SVM        | ﻿0.49464668094218417 | ﻿0.5224839400428265  | ﻿0.5053533190578159  | ﻿0.5139186295503212  | ﻿0.5074946466809421  |
| SVR        | ﻿0.5257510729613734  | ﻿0.5493562231759657  | ﻿0.4871244635193133  | ﻿0.4978540772532189  | ﻿0.5515021459227468  |
| MLP        | ﻿0.47109207708779444 | ﻿0.47109207708779444 | ﻿0.4989293361884368  | ﻿0.5032119914346895  | ﻿0.5331905781584583  |
| MLP Reg    | ﻿0.5236051502145923  | ﻿0.5236051502145923  | ﻿0.51931330472103    | ﻿0.4957081545064378  | ﻿0.4871244635193133  |
| RNN        | ﻿0.5010706638115632  | ﻿0.5053533190578159  | ﻿0.48822269807280516 | ﻿0.5010706638115632  | ﻿0.5546038543897216  |
| RNN Reg    | ﻿0.5364806866952789  | ﻿0.5021459227467812  | ﻿0.4699570815450644  | ﻿0.49356223175965663 | ﻿0.5064377682403434  |
| AE SVM     | ﻿0.5267665952890792  | ﻿0.5032119914346895  | ﻿0.5331905781584583  | ﻿0.5353319057815846  | ﻿0.5117773019271948  |
| AE SVR     | ﻿0.5021459227467812  | ﻿0.5085836909871244  | ﻿0.5364806866952789  | ﻿0.5236051502145923  | ﻿0.5150214592274678  |
| AE MLP     | ﻿0.5353319057815846  | ﻿0.5053533190578159  | ﻿0.5246252676659529  | ﻿0.5396145610278372  | ﻿0.5353319057815846  |
| AE MLP Reg | ﻿0.5515021459227468  | ﻿0.5515021459227468  | ﻿0.4957081545064378  | ﻿0.5128755364806867  | ﻿0.48068669527896996 |
| AE RNN     | ﻿0.5203426124197003  | ﻿0.5460385438972163  | ﻿0.48822269807280516 | ﻿0.5267665952890792  | ﻿0.5139186295503212  |
| AE RNN Reg | ﻿0.5343347639484979  | ﻿0.51931330472103    | ﻿0.44849785407725323 | ﻿0.5321888412017167  | ﻿0.48497854077253216 |

The hit ratio, MAE and MSE results can be seen in the tables above.
The highest hit ratio was a classification RNN model at 0.555 in fold5
with a close second of a 0.552% for the Auto-Encoded feedforward network
in fold2 and the Support Vector Regression in fold5. The models with the
lowest MAE and MSE were all regression models, with the lowest being the
feedforward regression network with a MAE of 0.02886502 and an MSE of
0.00140317.

On average, classification models had a higher hit rate, as more
training data was used -- going from an average 0.508 in Fold 1 to 0.526
in Fold 5. This was expected, as can be seen in [@Halevy2009], and
relationships begin to be found with increased data. As data is increased, it becomes easier to generalise, because you learn relationships in more situations. However, in the regression algorithms, the average hit rate actually decreased as the folds increased, suggesting that more data was, in fact, problematic for predicting the trend. However, when looking at the other measurements (mean squared error and mean absolute error), the lowest error was actually in Fold 4. This means that the models were, in reality, incredibly close to predicting the actual price (an MSE of 0.157 and a MAE 0.271) of the stock but not the trend. One surprising result is that in Fold 1 and Fold 2 regression models on average achieved a higher hit rate than the classification models suggesting that if little data obtainable then regression models will perform better. In Fold 1 and 2 regression models achieved an average hit rate of 0.529 and 0.526 respectively compared to the average classifcation hit rates in Fold 1 and 2 which was 0.508 and 0.509 respectively.

The hit rate is a good measurement as it demonstrates how an investor
could use these models. If an uptrend is predicted, then the investor
will know to buy the stock and can sell it at the next time step for a
profit (if the prediction is correct). However, the hit rate makes it
difficult to compare models in any other way, as mentioned before.
Regression models had a decreasing hit rate over increasing the training
size, but the MSE and MAE reached a low in Fold 4, suggesting that more
training data was helping prediction. MSE and MAE can be used to assess
the model's accuracy in predicting its intended output.

It became clear that, as the training data increased, classification
models had a decreasing MAE until Fold 4 (similar to regression). That
being said, the MSE actually increased with more training data,
suggesting that the model was more affected by outliers, which could
mean that the models began to overfit with more data. Regression and
Classification models achieved similar MAE and MSE on average, yet
regression was the most accurate in both of these measurements,
suggesting that it is more accurate to predict actual price than price
trend.

Another important factor to note is the effect of auto-encoding. To do
this, an average is taken for each of the performance measures when
auto-encoded and when not auto-encoded for both classification and
regression. Doing this showed that, on average, the auto-encoded
classification models outperformed the standard models in all three
performance measures. Also, it showed that, when the training data
increased, the performance increased in both standard and auto-encoded
models. However, for regression, the results were slightly different: On
average, the auto-encoded models outperformed the standard models in the
hit ratio in all but Fold 5, suggesting that auto-encoding helps to
improve the models' accuracy in predicting movements in the price and
its ability to predict the trend. On the other hand, the standard models
outperformed the auto-encoded models in both MSE and MAE in all five
folds. In fact, on average, the standard models were able to achieve a
MAE of 0.0867 suggesting that the models were, on average, 0.0867 away
from the actual price. Figure 1 of the Feed-Forward Network in Fold 5 shows how
close the regression models were to actually predicting the true price.

{% include image.html url="/assets/stock-price/ffn_fold5.png" description="Figure 1: The Price Prediction of The Feed-Forward Network vs The Actual Price in Fold 5." %}

A reason for the auto-encoder making the performance of regression models worse for the MAE and MSE can be explained by [@maas2009]. Regression in time series requires long term dependencies which is not represented through standard auto-encoders. [@maas2009] presents a recurrent auto-encoder which can allow for these long-term dependencies, and thus represent relationships over time in the compressed (encoded) data.

Looking at specific models, the best model for the hit ratio was the classification RNN with a hit ratio of 0.555 (3 s.f.) (as mentioned at the beginning), suggesting that, for the standard investor who will simply want to predict an uptrend or downtrend to make profit, a classification algorithm will work best. This also occurred in Fold 5, suggesting that the more data there is, the better. The best MSE and MAE was achieved by the regression feedforward neural network, with 0.0289(3 s.f.) and 0.00140(3 s.f.), respectively. This is incredibly close, and a plot of this can be seen in Figure 1 to show how close this was to predicting the actual price.

# Conclusion

To conclude, if the user is trying to achieve the highest hit rate to profit from the model, then the best model to use would be a classification RNN. This model performed better as the training data increased so it is assumed that, if using this model, then the more training data there is, the better the outcome will be. However, if little training data is available, then the user may be more inclined to use a regression model: Fold 1, 2 and 3 all had a regression model for the top spot. It should be noted, though, that if the auto-encoded feedforward network or the auto-encoded support vector regression model is used, then the prices predicted should not be taken as an accurate prediction of the actual price. In short, the models are good at predicting the movement (hit ratio) but not the actual price (having a high MSE and MAE).

However, if wanting to predict the actual price, then a classification model cannot be used. This experiment has shown that regression models are able to do this with a high accuracy, and that results improve with more training data. In this experiment, a regression feedforward network achieved the best results and are far quicker to train than the LSTM due to there being fewer components to learn. However, LSTMs have proved to be more successful in other areas such as in [@sakda] where \"state-of-the-art\"[@sakda] performance is achieved in acoustic modelling, and so more research needs to be done here to see if their performance can be enhanced by other features or a different architecture to the LSTM used.

Auto-encoding proved to be successful for classification algorithms but not regression algorithms, suggesting more research could be done to find different types of auto-encoders. This could then enhance the performance of regression models in the scenario, like the recurrent auto-encoder mentioned in [Results Section](#results-and-discussion).

Finally, this experiment could have been extended to try different stocks, so as to see if these results generalise to many different stocks or whether the results achieved only happen on this specific stock. To see how these results can be useful for investors see Appendix  which discusses trading strategies that could be implemented with these results.

# References

Heaton, J.B., Polson, N.G., Witte, J.H. Deep Learning For Finance: Deep
Portfolios. . 2017, **33**(1), pp.3-12.

Murphy, K. . Massachusetts: The MIT Press, 2012.

Nocedal, J. and Wright, S. . New York: Springer, 2000.

Bottou, L. Stochastic Gradient Learning in Neural Networks. . 1991,
**91**(8), p.8.

James, G., Witten, D., Hastie, T., Tibshirani, R. . 7th ed. New York:
Springer, 2014.

Manning, C.D., Raghavan, P., Schütze, H. . Reprint. Cambridge: Cambridge
University Press, 2017.

Goodfellow, I., Bengio, Y., Courville, A. . Massachusetts: The MIT
Press, 2016.

Han, J., Kamber, M., Pei, J. . Waltham: Elsevier, 2012.

Chang, Y. W., Hsieh, C. J., Chang, K. W., Ringgaard, M., Lin, C. J.
Training and Testing Low-degree Polynomial Data Mappings via Linear SVM.
. 2010, **11**(4), pp.1471-1490.

Géron, A. . Sebastopol: O'Reilly, 2017.

Hastie, T., Tibshirani, R., Friedman, J.H. . 12th ed. New York:
Springer, 2017.

Cortes, C. and Vapnik, V. Support-Vector Networks. . 1995, **20**(3),
pp. 273-297.

Kim, K. J. Financial Time Series Forecasting Using Support Vector
Machines. . 2003, **55**(1), pp. 307-319.

Qian, X.Y. and Gao, S. \[E-print\]. Financial Series Prediction:
Comparison Between Precision of Time Series Models and Machine Learning
Methods. . arXiv:1706.00948. 2017.

Vapnik, V., Golowich, S., Smola, A. Support Vector Method for Function
Approximation, Regression Estimation and Signal Processing. In: Mozer,
M.C., Jordan, M.I., Petsche, T. eds. . Massachusetts: The MIT Press,
1997, pp. 281-287.

Schoelkopf, B. and Smola, A. . Massachusetts: The MIT Press, 2002.

LeNail, A. NN-SVG: Publication-Ready Neural Network Architecture
Schematics. . 2019, **4**(33), p. 747.

Hornik, K. Approximation Capabilities of Multilayer Feedforward
Networks. . 1991, **4**(2), pp. 251-257.

Bengio, Y., Simar, P., Frasconi, P. Learning Long-Term Dependencies with
Gradient Descent is Difficult. . 1994, **5**(2), pp. 157-166.

Hochreiter, S. and Schmidhuber, J. Long Short-term Memory. . 1997,
**9**(8), pp. 1735-1780.

Sak, H., Senior, A., Beaufays, F. Long short-term memory recurrent
neural network architectures for large scale acoustic modeling. . 2014,
**15**(1), pp. 338-342.

Zaremba, W., Sutskever, I., Vinyals, O. Recurrent Neural Network
Regularization. . 2014, **abs/1409.2329**.

Erhan, D., Bengio, Y., Courville, A., Manzagol, P.A., Vincent, P.,
Bengio, S. Why Does Unsupervised Pre-training Help Deep Learning? .
2010, **11**(3), pp. 625-660.

Heaton, J.B, Polson, N.G., Witte, J.H. \[E-print\]. Deep Learning in
Finance. . arXiv:1602.06561. 2016.

Chen, J., Sathe, S., Aggarwal, C., Turaga, D. Outlier Detection with
Autoencoder Ensembles. In: . 2017, pp. 90-98.

Guo, Y., Liao, W., Wang, Q., Yu, L., Ji, T., Li, P. Multidimensional
Time Series Anomaly Detection: A GRU-based Gaussian Mixture Variational
Autoencoder Approach. . 2018, **95**(1), pp. 97-112.

Bergmeir, C. and Benı́tez, J.M. On the use of cross-validation for time
series predictor evaluation. . 2012, **191**(1), pp. 192-213.

Uma, D.B., Sundar, D., Alli, P. An Effective Time Series Analysis for
Stock Trend Prediction Using ARIMA Model for Nifty Midcap-50. . 2013,
**3**(1), pp. 65-78.

Adebiyi, A., Adewumi, A., Ayo, C. Stock price prediction using the ARIMA
model. In: *UKSim-AMSS 16th International Conference on Computer
Modelling and Simulation, 26 - 28 March 2014, Cambridge*. Washington:
IEEE Computer Society, 2014, pp. 106-112.

Ho, S.L., Xie, M., Goh, T.N. A Comparative Study of Neural Network and
Box-Jenkins ARIMA Modeling in Time Series Prediction. . 2002, **42**(2),
pp. 371 - 375.

Pai, P.F. and Lin, C.S. A hybrid ARIMA and support vector machines model
in stock price forecasting. . 2005, **33**(6), pp. 497-505.

Chen, J., Chen, W., Huang, C., Huang, S., Chen, A. Financial Time-Series
Data Analysis Using Deep Convolutional Neural Networks. In: *2016 7th
International Conference on Cloud Computing and Big Data (CCBD), 16 - 18
November 2016, Macau*. Washington: IEEE Computer Society, 2016, pp.
87-92.

Edet, S. Recurrent Neural Networks in Forecasting S&P 500 Index. . 2017,
Available at SSRN: https://ssrn.com/abstract=3001046 or
http://dx.doi.org/10.2139/ssrn.3001046.

Iqbal, Z, Ilyas, R., Shahzad, W., Mahmood, Z., Anjum, J. Efficient
Machine Learning Techniques for Stock Market Prediction. . 2019,
**3**(1), pp. 855-867.

Wanjawa, B. and Muchemi, L. \[E-print\]. ANN Model to Predict Stock
Prices at Stock Exchange Markets. . arXiv:1602.06561. 2014.

Sirignano, J. and Cont, R. Universal Features of Price Formation in
Financial Markets: Perspectives From Deep Learning. . 2018, Available at
SSRN: https://ssrn.com/abstract=3141294 or
http://dx.doi.org/10.2139/ssrn.3141294.

Raybaut, P. *Spyder* (3.3.1). \[Software\]. 2018. \[Accessed 10th
January 2019\].

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
Duchesnay, E. Scikit-learn: Machine Learning in Python. . 2011,
**12**(1), pp. 2825-2830.

Chollet, F. *Keras* (2.2.0). \[Software\]. 2018. \[Accessed 10th January
2019\].

Yahoo! Finance. Koninklijke KPN N.V. (KPN.AS). 30th Jan. *Yahoo!
Finance*. 2019. \[Online\]. \[Accessed 5th Feb 2019\]. Available from:
https://finance.yahoo.com/quote/KPN.AS

Halevy, A., Norvig, P., Pereira, F. The Unreasonable Effectiveness of
Data. . 2009, **24**(2), pp. 8-12.

Maas, A., Le, Q.V., O'Neil, T.M., Vinyals, O., Nguyen, P., Ng, A.Y.
Recurrent Neural Networks for Noise Reduction in Robust ASR. In: .
Published in INTERSPEECH 2012. Available at:
https://www.isca-speech.org/.


# A1 Trading Strategies

This section will discuss 3 strategies that could be taken when trading
the KPN stock from $15$th February $2017$ to $16/01/2019$ using trend
prediction.

1.  The first strategy is the benchmark strategy, this is the buy and
    hold strategy.

2.  The second strategy is traders who are after quick profit, this
    strategy works by:

    -   If an uptrend is predicted by the model: buy the stock and sell
        it the next day

    -   If a downtrend is predicted by the model: sell the stock and buy
        it the next day

    -   The total profit is then calculated after doing this for all 467
        days.

3.  The third strategy is for more of a part-time investor, this
    strategy works by:

    1.  If the model predicts an uptrend buy the stock, if it predicts a
        downtrend do nothing

    2.  If holding the stock and the model predicts and uptrend hold
        onto it, when a downtrend is predicted sell the stock

    3.  Repeat previous steps until final day and sell the stock on the
        final.

The model used is the classification RNN which achieved the highest hit
rate in Fold 5.

#### Profits

The buy and hold (first) strategy made a loss of €0.509, this is because
the stock price decreased over the time period. Because the stock was
bought on the first day for €3.327 and sold for €2.818. The second
strategy ahcieved a net profit of €4.575 from just buying and selling
one stock a day depending on the model prediction. If investment was
increased in the second strategy extraordinary profits could be made.
The final strategy which is for an investor who does not have as much
money to risk if similar to the buy and hold strategy but instead uses
the model to time purchases and sales inbetween the start and end date.
This strategy can earn the investor €0.624 from just one stock, this
strategy is less risky and so if the investor invested more money into
this strategy they could also make high returns.
