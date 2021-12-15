---
layout: page
---

![Taxonomy of of certifiably robust approaches against \\(\ell_p\\) adversary. Full details available in SoK](/assets/taxonomy.svg)

This is an anonymized website that provides [**full benchmark results**](/benchmark) and [**state-of-the-art leaderboard**](/leaderboard) on the certified robustness for deep neural networks.

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
