---
author: Maksim An
title:  TensorFlow and Recurrent Neural Networks Contd.
category: TensorFlow
tags: python tensorflow rnn prediction
summary: TensorFlow for internet traffic prediction
date:   2016-06-28 22:00 +0900
layout: post
---

*__TLDR__:* Get some data, manipulate with it and with your model configs, get some results which look kind of *OK-ish* and be happy with it.

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

![Normalized data](/assets/2016-06-28/normalized-data.png)

The full code for preprocessing can be found [here][data-loaders].

Lets start with defining a base class for our model. You can find it [here][base-class] and it is basically a copy of google's [ptb model][ptb-model].

Lets move all of our initializations done in [sine wave][github-sine] model into `__init__`:

{% highlight python %}
# class name is TrafficRNN(SequenceRNN)
def __init__(self, is_training, config):
    seq_width = config.seq_width
    n_steps = config.batch_size
    num_hidden = config.num_hidden
    num_layers = config.num_layers

    self._seq_input = tf.placeholder(tf.float32, [n_steps, seq_width])
    self._seq_target = tf.placeholder(tf.float32, [n_steps, 1])
    self._early_stop = tf.placeholder(tf.int32)

    inputs = [tf.reshape(data, (1, seq_width))
                for data in tf.split(0, n_steps, self.seq_input)]
    initializer = tf.random_uniform_initializer(-.1, .1)

    cell = rnn_cell.LSTMCell(num_hidden, seq_width, initializer=initializer)
    if num_layers > 1:
        cell = rnn_cell.MultiRNNCell([cell]*num_layers)

    self._initial_state = cell.zero_state(1, tf.float32)
    outputs, states = rnn(cell, inputs,
                          initial_state=self._initial_state,
                          sequence_length=self._early_stop)
    self._final_state = states
    outputs = tf.reshape(tf.concat(1, outputs), [-1, num_hidden])

    W = tf.get_variable('W', [num_hidden, 1])
    b = tf.get_variable('b', [1])
    _output = tf.matmul(outputs, W) + b
    self._output = _output
    error = tf.pow(
                tf.reduce_sum(tf.pow(tf.sub(_output, self._seq_target), 2)), .5)
    self._error = error
    if not is_training:
        return

    self._lr = tf.Variable(0., trainable='False', name='lr')
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._error, tvars),
                                      config.max_grad_norm)

    optimizer = tf.train.AdamOptimizer(self._lr)

    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

# Lets add an output property as well
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

Cool, now lets continue. I am going to use the data with 5 min interval measurements. So I want my model to predict the internet traffic in some future time based on the value(s) in the past. The question is how far into the future? And how much of the information from the past are we going to use? This is going to affect the parameters of our model. This is how my configuration looks like:

{% highlight python %}
class TrafficRNNConfig(object):
    start = 0               # The index of the first data entry we are going
                            # to use
    window_size = 12        # Number of values in the "past". For 5-min interval
                            # data this is going to be one hour interval
    n_steps = 4032          # Number of samples in the training set
    use_1st_diffs = True    # Use first order differences or not
    use_2nd_diffs = False   # Use second order differences or not
    lag = 48                # How far into the future do we want to predict.
                            # For 5 min data this is 4 hours
    batch_size = 48         # Batch size

    max_epoch = 300         # Number of epochs during training
    num_hidden = 150        # Number of hidden nodes
    num_layers = 1          # Number of layers

{% endhighlight %}

We have our model and its configuration, now we need to train it. Lets make a function that will do that for us:

{% highlight python %}
# Runs the model on some data (training or testing)
def run_epoch(session, m, data, eval_op, config):
    """
    session:    TensorFlow session
    m:          model
    data:       input data
    eval_op:    train op or nothing
    config:     model config
    """
    state = m.initial_state.eval()

    seq_input = data['seq_input']
    seq_target = data['seq_target']
    early_stop = data['early_stop']

    epoch_error = 0.
    rnn_outs = np.array([])
    for i in range(config.n_steps/config.batch_size):
        # Make batches
        _seq_input = seq_input[i*config.batch_size:(i+1)*config.batch_size][:]
        _seq_target = seq_target[i*config.batch_size:(i+1)*config.batch_size][:]
        _early_stop = early_stop
        feed = {m.seq_input:_seq_input, m.seq_target:_seq_target,
                m.early_stop:_early_stop, m.initial_state:state}
        # Feed data to the model
        step_error, state, step_outs, _ = session.run([m.error, m.final_state,
                                                            m.output, eval_op],
                                                      feed_dict=feed)
        # Accumulate total error
        epoch_error += step_error
        rnn_outs = np.append(rnn_outs, step_outs)

    return epoch_error, rnn_outs
{% endhighlight %}

As the function name states: it runs one epoch on data, if the eval_op is a train_op it also performs trainnig of the model, otherwise it just feeds the data to the model and gets the outputs.

Now a main function where we do all initialization, model training and evaluation:

{% highlight python %}
def main(unused_args):
    tdLoader = TrafficDataLoader(
                    'internet-data/data/internet-traffic-11-cities-5min.csv',
                    max_norm=5.)
    tmConfig = TrafficRNNConfig()
    batch_size = tmConfig.batch_size
    seq_input, seq_target = tdLoader.get_rnn_input(tmConfig)
    data = dict(seq_input=seq_input, seq_target=seq_target,
                early_stop=tmConfig.batch_size)

    is_training = True
    with tf.Session() as session:
        model = TrafficRNN(is_training=True, config=tmConfig)
        tf.initialize_all_variables().run()
        if is_training:
            lr_value = 1e-3
            for epoch in range(tmConfig.max_epoch):
                # Skipped learning rate choice here
                # ...
                model.assign_lr(session, lr_value)
                net_outs_all = np.array([])
                error, net_outs_all = run_epoch(session, model, data,
                                                model.train_op, tmConfig)
                error, net_outs_all = run_epoch(session, model, data,
                                                tf.no_op(), tmConfig)
        else:
            # is_training == False => restore model from backup
            saved_vars = 'your-model-save-file-here'
            saver.restore(session, saved_vars)

        train_error, train_outs_all = run_epoch(session, model, data,
                                                tf.no_op(), tmConfig)
        testDataConfig = TestConfig() # See my repo for this. Sorry :|
        test_seq_input, test_seq_target = tdLoader.get_rnn_input(testDataConfig)
        test_data = dict(seq_input=test_seq_input, seq_target=test_seq_target,
                         early_stop=testDataConfig.batch_size)
        test_outs_all = np.array([])
        test_error, test_outs_all = run_epoch(session, model, test_data,
                                              tf.no_op(), testDataConfig)
{% endhighlight %}

Now the `__main__` function looks like this:

{% highlight python %}
if __name__=='__main__':
    tf.app.run()
{% endhighlight %}

On my PC it ran for about 30mins or so... The result of training looks like this:
![Training Done](/assets/2016-06-28/training-data.png)

Training data + test data:
![Training and Testing Data](/assets/2016-06-28/training-testing-data.png)

Now if we make a small "window" of possible error, we can get something like this for our prediction:
![Prediction](/assets/2016-06-28/prediction.png)

So as you can see it "kind of works", we could catch the seasonality in data, sometimes even the traffic usage drops on weekends, but it's not perfect after all. Maybe we can keep on tuning our model or use a more complex one (not sure which though).

The whole code can be found [here][github-traffic]. Feel free to use it! And really hope that my post was helpfull! I have another use case in my mind and will try to make something when I have free time. Good luck!

*__NOTE:__* This post was in draft since March and due to lack of time I couldn't finish it then.

[traffic-data]: https://datamarket.com/data/list/?q=cat:ecd%20provider:tsdl
[data-market]: https://datamarket.com
[data-loaders]: https://github.com/anmaxvl/machine-learning/blob/master/data_loaders.py
[base-class]: https://github.com/anmaxvl/machine-learning/blob/master/sequence_rnn.py
[ptb-model]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
[github-sine]: https://github.com/anmaxvl/machine-learning/blob/master/sine_wave.py
[github-traffic]: https://github.com/anmaxvl/machine-learning
