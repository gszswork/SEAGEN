# SEAGEN
Self-Attention Aware Graph Evolution Network

Reposiory for Paper ["Fake News Detection Through Temporally Evolving User Interactions"](https://link.springer.com/chapter/10.1007/978-3-031-33383-5_11) in PAKDD-2023.

## Paper Abstract

Detecting fake news on social media is an increasingly important problem, because of the rapid dissemination and detrimental impact of fake news. Graph-based methods that encode news propagation paths into tree structures have been shown to be effective. Existing studies based on such methods represent the propagation of news through static graphs or coarse-grained graph snapshots. They do not capture the full dynamics of graph evolution and hence the temporal news propagation patterns. To address this issue and model dynamic news propagation at a finer-grained level, we propose a temporal graph-based model. We join this model with a neural Hawkes process model to exploit the distinctive self-exciting patterns of true news and fake news on social media. This creates a highly effective fake news detection model that we named SEAGEN. Experimental results on real datasets show that SEAGEN achieves an accuracy of fake news detection of over 93% with an advantage of over 2.5% compared to other state-of-the-art models.

## Dataset

The dataset used in the paper are Twitter and FakeNewsNet. Twitter dataset is the fusion of publicly released dataset Twitter15 and Twitter16 by (Ma et al. 2016). The original data from Twitter15 and Twitter16 contains 4 labels (true-rumour, false-rumour, non-rumour and unverified). Only true-rumour and false-rumour data samples are collected and form the Twitter dataset, in which true-rumour is defined as true news, and false-rumour is defined as false/fake news. 

FakeNewsNet dataset is released by 

## To Reproduce the Experiment


## Reference 

[1] Jing Ma, Wei Gao, Kam-Fai Wong. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. ACL 2017.
[2] 
