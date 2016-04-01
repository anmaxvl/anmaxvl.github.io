---
Author: Maksim An
title: TensorFlow and Recurrent Neural Networks Contd.
category: Coding
tags: python tensorflow rnn prediction
summary: TensorFlow for internet traffic prediction
date: 2016-03-24 22:00 +0900
layout: post
---

This is a continuation of my previous post, however this time I tried to apply TF's RNN on some real data.
The data set I am going to be using is [internet traffic data][traffic-data] from [datamarket.com][data-market]. The numbers are somewhat big, so we will need to do some preprocessing. I did a simple log of the data and normalized it between 0 and 5. The data is in CSV format with first column being the timestamp in the form `%Y-%m-%d %H-%M-%S` and the second column is the traffic data in bits. There shouldn't be a problem reading the data, so I will skip it.

Log+normalization

{% highlight python %}
def normalize(self, max_norm):
    #lets log our data
    logged_data = np.log(self.values)
    #and normalize between 0 and max_norm
    min_val = np.min(logged_data)
    max_val = np.max(logged_data)
    normalized = max_norm*(logged - min_val)/(max_val-min_val)
    self._values = normalized
{% endhighlight %}

Lets see how it looks:

![Normalized data](/assets/2016-03-25/normalized-data.png)

The full code for preprocessing can be found [here][data-loaders].

Lets start with defining a base class for our model. You can find it [here][base-class] and it is basically a copy of google's [ptb model][ptb-model].

Lets move all of our initializations in [sine wave][github-sine] model into `__init__`:

{% highlight python %}
#class name is TrafficRNN(SequenceRNN)
def __init__(self, is_training, config):
    #Everything is the same here, except for the fact that we want our
    #tensors and ops to be instance variables
    self._seq_input = tf.placeholder(tf.float32, [batch_size, seq_width])
    self._seq_target = ...
    self._early_stop = ...

    #Lets add more layers
    cell = rnn_cell.LSTMCell(num_hidden, seq_width, initializer = initializer)
    if num_layers > 1:
        cell = rnn_cell.MultiRNNCell([cell]*num_layers)

    ...
    #similarly as placeholders make sure that all ops and tensors we will use
    #are instance variables as well.
    self._initial_state = ...

    ...

#Lets add an output property as well
@property
def output(self):
    return self._output
{% endhighlight %}

Everything was pretty much the same as in the previous post, with the only difference that now we are not using global variables, but object properties instead. For example:

{% highlight python %}
m = TrafficRNN(is_training=True, config)
session.run([m.train_op, m.output],
            feed_dict={m.seq_input:_input, m.seq_target:_target})
{% endhighlight %}

Cool, now lets continue. I am going to use the data with 5 min interval measurements. So I want my model to predict the internet traffic in some future time based on the value(s) in the past. The question is how far into the future? And how much of the information from the past are we going to use? This is going to affect the parameters of our model. This is how my configuration file looks like:

{% highlight python %}
class TrafficRNNConfig(object):
    start = 0           #The index of the first data entry we are going to use
    window_size = 12    #Number of values in the "past". For 5 min interval data
                        #this is going to be one hour interval
    n_steps = 4032      #Number of samples in the training set
    use_1st_diffs = True    #Use first order differences or not
    use_2nd_diffs = False   #Use second order differences or not
    lag = 48            #How far into the future do we want to predict. For 5
                        #min data this is 4 hours
    batch_size = 48     #Batch size

    max_epoch = 300
    num_hidden = 150
    num_layers = 1

{% endhighlight %}

[traffic-data]: https://datamarket.com/data/list/?q=cat:ecd%20provider:tsdl
[data-market]: https://datamarket.com
[data-loaders]: https://github.com/anmaxvl/machine-learning/blob/master/data_loaders.py
[base-class]: https://github.com/anmaxvl/machine-learning/blob/master/sequence_rnn.py
[ptb-model]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
[github-sine]: https://github.com/anmaxvl/machine-learning/blob/master/sine_wave.py
