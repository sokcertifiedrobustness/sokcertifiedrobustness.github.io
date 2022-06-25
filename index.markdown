---
layout: page
---

![Taxonomy of of certifiably robust approaches against \\(\ell_p\\) adversary. Full details available in SoK](/assets/taxonomy.svg)

This is the accompying website of [SoK: Certified Robustness for Deep Neural Networks](https://arxiv.org/abs/2009.04131) to appear at [IEEE SP 2023](https://www.ieee-security.org/TC/SP2023/).

<div
        style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.0em .0em .0em .0em;padding:.2em .6em;">
        <pre style="margin: 0; line-height: 125%"><span style="color: #555555; font-weight: bold">@inproceedings</span>{li2023sok,
    title<span style="color: #333333">=</span>{SoK: Certified Robustness for Deep Neural Networks},
    author<span style="color: #333333">=</span>{Linyi Li <span style="color: #000000; font-weight: bold">and</span> Tao Xie <span style="color: #000000; font-weight: bold">and</span> Bo Li},
    booktitle<span style="color: #333333">=</span>{44th {IEEE} Symposium on Security and Privacy, {SP} 2023, San Francisco, CA, USA, 22-26 May 2023},
    publisher<span style="color: #333333">=</span>{IEEE},
    year<span style="color: #333333">=</span>{2023}
}</pre>
</div>

This website provides [**full benchmark results**](/benchmark) and [**state-of-the-art leaderboard**](/leaderboard) on the certified robustness for deep neural networks.

**Benchmark**: In the benchmark page, we provide **full comparison results** along with **experimental setups** of representative certifiably robust approaches, including both verification approaches for DNNs and verification + training approaches for smoothed DNNs.
These results were run with our open-sourced toolbox.


**Leaderboard**: In the leaderboard page, we collect the highest certified accuracies achieved on three common datasets (MNIST, CIFAR-10, and ImageNet) against \\(\ell_1\\), \\(\ell_2\\), and \\(\ell_\infty\\) adversaries reported by existing papers. These high certified robust accuracies are achieved by leveraging both robust training and verification approaches.

On both pages, the main evaluation metric is
\\\[\text{certified accuracy} = \\dfrac{\\text{\# samples verified to be robust}}{\\text{number of all evaluated samples}}.\\\]

Benchmark and Leaderboard are created for different purposes:  
- Benchmark aims to provide fair comparisons for verification approaches (and smoothing distributions/certification for smoothed DNNs), where we drew several findings as detailed in the SoK paper.

- Leaderboard mainly reflects the progresses achieved in certified robustness, which provides a guidance for users who want to pick an approach to achieve high certified robustness for their datasets. Since the progresses are made by both robust training and verification approaches, the leaderboard does not make direct comparison for either robust training or verification approaches.


##### FAQ

- Interested in empirical robustness, say, state-of-the-art robust ML models against existing attacks?

In this site we only consider certified robustness, which cannot be broken by future adaptive attackers. If you would like to know state-of-the-art empirically robust ML models, we recommmend you to browse [RobustBench](https://robustbench.github.io/) and [robustml.org](https://www.robust-ml.org/).

- Interested in robustness against other realistic adversaries beyond \\(\ell_p\\) adversaries?

The certifiably robust approaches for other adversaries are usually inspired by the approaches against \\(\ell_p\\) adversaries. An overview of them is available in the SoK.
