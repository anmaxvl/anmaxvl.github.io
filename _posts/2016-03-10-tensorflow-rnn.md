---
author: Maksim An
title:	TensorFlow and Recurrent Neural Networks
category: TensorFlow
tags: python tensorflow rnn
summary: My attempt in using TensorFlow to predict sequences.
date:	2016-03-10 13:34:00 +0900
layout:	post
---

Google open sourced their library for numerical computations in the end of the last year: [TensorFlow][tensorflow-official]. It comes with a nice python API and documentation. I know that there are other libraries which may be simpler or even perform better (in terms of speed), but it's Google, you know, and I am one of those `#google#only#does#great#things` boys (jk). It just happened that it was open sourced right at the time I got interested in machine learning again.

To start things off, I decided to look into examples and tutorials Google provided. They work and all, but guess what? it is actually not that simple to jump straight into developing something on your own (at least that was my case). So I decided to start from simple things and going to share my experience with you.

[This][reddit-post] post on reddit is a good place to start once you figured out TensorFlow's basics. Make sure to play around with variables, tensors and sessions first. So what we have in there? The code generates a random sequence, splits it into batches and feeds a LSTM RNN. If you are not sure what are those, make sure to read [colah's blog][colah-blog], he gives a great explanation and logic behind LSTMs.

Feeding a random sequence might not be very interesting, so lets generate one:

{% highlight python  %}
import numpy as np
import matplotlib.pylab as plt

def gen_seq():
	x = np.arange(0., np.pi*12, 0.03)
	bell = np.exp(-(np.sin(x-np.pi/2)-2*np.pi)**2/9.)
	y = 100*np.sin(8*x)*bell
	y = np.reshape(y, (len(y), 1))

	plt.plot(x, y)
	plt.show()

	return x, y
{% endhighlight%}

And we will get something like this:
![Sequence plot](/assets/sequence.png)

Good, we have a sequence now, and we want to feed it to the network. Following the reddit example, lets build our own TensorFlow computation graph.

{% highlight python %}
import tensorflow as tf
from tensorflow.models.rnn.rnn import *

x, y = gen_seq()
#number of hidden units in RNN
num_hidden = 10
#sequence width
seq_width = 10
#we are going to train on the first half of the sequence
n_steps = len(x)/2

#we are not going to use batches
#our inputs
seq_input = tf.placeholder(tf.float32, [n_steps, seq_width])
#targets, which will be used later during training
seq_target = tf.placeholder(tf.float32, [n_steps, 1])
#early stop to pass a sequence length if needed to save computation time
early_stop = tf.placeholder(tf.int32)

#tensorflow's rnn need a list of tensors instead of a single tensor
inputs = [tf.reshape(i, (1,seq_width)) for i in tf.split(0, n_steps, seq_input)]

initializer = tf.random_uniform_initializer(-.1, .1)
#LSTM cell
cell = rnn_cell.LSTMCell(num_hidden, seq_width, initializer=initializer)
initial_state = cell.zero_state(1, tf.float32)

#feeding inputs to rnn
outputs, states = rnn(cell, inputs, initial_state=initial_state,
						sequence_length=early_stop)

{% endhighlight %}

Our goal is to make a prediction what is the value going to be in `lag` steps, based on `seq_width` number of previous steps.

{% highlight python %}
def gen_inputs(y, n_steps, offset, seq_width=10, lag=60):
	seq_input = []
	seq_target = []

	for i in range(offset, offset+n_steps):
		window=[]
		for j in range(seq_width):
			if i+j<len(y):
				window.append(y[i+j])
			else:
				window.append(0)
		seq_input.append(window)
		if i+lag+seq_width < len(y):
			seq_target.append(y[i+lag+seq_width])
		else:
			seq_target.append(0)

	seq_input = np.reshape(np.array(seq_input), (-1, seq_width))
	seq_target = np.reshape(np.array(seq_target), (-1, 1))

	return seq_input, seq_target
{% endhighlight %}

Lets generate inputs and targets and feed them to our rnn:

{% highlight python %}
train_input, train_target = gen_inputs(y, n_steps, offset=0, seq_width=10, lag=60)

#init op
init = tf.initialize_all_variables()

session = tf.Session()
#actual initialization
session.run(init)
#feed dictionary
feed = {seq_input:train_input, seq_target:train_target, early_stop:n_steps}

net_outs = session.run(output, feed_dict=feed)

plt.plot(x[:n_steps], train_target, 'b-', x[:n_steps], net_outs, 'r-')
plt.show()
{% endhighlight %}

And we get the figure below, with blue line being the target sequence and red line being the network output:

![Sequence and initial output](/assets/sequence-and-network-output.png)

Now let's add some train ops.

{% highlight python %}
#output layer
W = tf.get_variable('W', [num_hidden, 1])
b = tf.get_variable('b', [1])

#network outputs is a list of tensors, but to make matrix multiplications we need a single tensor
outputs = tf.reshape(tf.concat(1, outputs), [-1, num_hidden])

output = tf.matmul(outputs, W) + b

#error function
error = tf.pow(tf.reduce_sum(tf.pow(tf.sub(output, seq_target), 2)), .5)

#learning rate
lr = tf.Variable(0., trainable=False, name='lr')

#optimizer
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(error, tvars), 5.)
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))
{% endhighlight %}

Now lets run training for 100 epochs:

{% highlight python %}
#we will save our model after training is complete, so we can restore it later on
saver = tf.train.Saver()

for i in range(100):
	new_lr = 1e-2
	if i > 25:
		new_lr = 5e-3
	elif i > 50:
		new_lr = 1e-4
	elif i > 75:
		new_lr = 1e-5

	session.run(tf.assign(lr, new_lr))

	err, net_outs, _ = session.run([error, output, train_op], feed_dict=feed)
	print('Epoch %d: %1.5f') % (i, err)

saver.save(session, 'sine-wave-rnn-'+str(num_hidden)+'-'+str(seq_width), global_step=0)

#now lets feed training data to the network
train_outs = session.run(output, feed_dict=feed)
plt.plot(x[:n_steps], train_target, 'b-', x[:n_steps], train_outs, 'g--')

#get the rest of the data
test_input, test_target = get_input(y, n_steps, offset=n_steps, seq_width=10,lag=60)
test_fed={seq_input:test_input, seq_target:test_target, early_stop:n_steps}
test_outs = session.run(output, feed_dict=feed)

plt.plot(x[n_steps:2*n_steps], test_outs, 'r--')
plt.plot(x[n_steps:2*n_steps], test_target, 'b-')
plt.show()
{% endhighlight %}

And here is what we get after training:
![Training complete](/assets/training-complete.png)

Full code is available on my [github][my-github]. This implementation is not optimal of course, but I wanted to make it straightforward. Some sort of batch processing can be done in order to reduce memory consumption etc. But I will leave it for now and may be (if I am not too lazy) fix this later. Hope you will have fun with it :)

Cheers!

[tensorflow-official]: https://www.tensorflow.org/
[reddit-post]: https://www.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/
[colah-blog]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[my-github]: https://github.com/anmaxvl/machine-learning/blob/master/sine_wave.py
