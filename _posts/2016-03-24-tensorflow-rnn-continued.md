---
Author: Maksim An
title: TensorFlow and Recurrent Neural Networks Contd.
category: Coding
tags: python tensorflow rnn prediction
summary: TensorFlow for internet traffic prediction
date: 2016-03-24 22:00 +0900
layout: post
---

This is a continuation of my previous post, but now I will try to apply TF's RNN to some real data. The data set I am going to be using is [internet traffic data][traffic-data] from [datamarket.com][data-market]. The numbers are somewhat big, so we will need to do some preprocessing. I did a simple log of the data and normalized it between `0` and `5`.

Lets start with defining a base class for our model. You can find it [here][base-class].

[traffic-data]: https://
[data-market]: https://
[base-class]: https://github.com/anmaxvl/machine-learning/sequence_rnn.py
