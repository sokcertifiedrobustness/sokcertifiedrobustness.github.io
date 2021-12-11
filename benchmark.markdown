---
layout: page
title: Benchmark
permalink: /benchmark/
---


<div class="list-group" style="margin-bottom:50px">
  <a href="#deter" class="list-group-item list-group-item-action" aria-current="true">
    Deterministic Verification Approaches (for DNNs) 
  </a>
  <a href="#prob" class="list-group-item list-group-item-action">
    Probabilistic Verification Approaches (for smoothed DNNs)
  </a>
</div>

<hr>

### <a name='deter'> Deterministic </a> Verification Approaches (for DNNs)


<div class="accordion" id="accordionPanelsStayOpenExample">
  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingOne">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseOne" aria-expanded="false" aria-controls="panelsStayOpen-collapseOne">
        Evaluated Approaches
      </button>
    </h2>
    <div id="panelsStayOpen-collapseOne" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingOne">
      <div class="accordion-body">

      We evaluate the following 17 approaches.
      
      <br><br>

{{
  "
| Category                                                     | Name                                                         | Implementation                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Complete > Solver-Based                                      | [Bounded MILP](https://openreview.net/forum?id=HyGIdiRqtm)   | Reimplementation                                             |
| Complete > Branch-and-Bound                                  | [AI2](https://ieeexplore.ieee.org/document/8418593)          | From [ERAN](https://github.com/eth-sri/eran)                 |
| Incomplete > Linear Relaxation > Linear Programming          | LP-Full[[1](https://arxiv.org/abs/1902.08722),[2](https://arxiv.org/abs/1804.09699v1)] | From [CNN-Cert](https://github.com/IBM/CNN-Cert)             |
| Incomplete > Linear Relaxation > Linear Inequality > Interval | [IBP](https://openaccess.thecvf.com/content_ICCV_2019/html/Gowal_Scalable_Verified_Training_for_Provably_Robust_Image_Classification_ICCV_2019_paper.html) | Reimplementation                                             |
| Incomplete > Linear Relaxation > Linear Inequality > Polyhedra | [Fast-Lin](https://arxiv.org/abs/1804.09699v1)               | From [CNN-Cert](https://github.com/IBM/CNN-Cert)             |
| Incomplete > Linear Relaxation > Linear Inequality > Polyhedra | [CROWN](https://arxiv.org/abs/1811.00866)                    | From [CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP)   |
| Incomplete > Linear Relaxation > Linear Inequality > Polyhedra | [CNN-Cert](https://arxiv.org/abs/1811.12395)                 | From [CNN-Cert](https://github.com/IBM/CNN-Cert)             |
| Incomplete > Linear Relaxation > Linear Inequality > Polyhedra | [CROWN-IBP](https://arxiv.org/abs/1906.06316)                | From [CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP)   |
| Incomplete > Linear Relaxation > Linear Inequality > Polyhedra | [DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf) | From [ERAN](https://github.com/eth-sri/eran)                 |
| Incomplete > Linear Relaxation > Linear Inequality > Polyhedra | [RefineZono](https://openreview.net/forum?id=HJgeEh09KQ)     | From [ERAN](https://github.com/eth-sri/eran)                 |
| Incomplete > Linear Relaxation > Linear Inequality > Duality | WK[[1](https://arxiv.org/abs/1711.00851),[2](https://arxiv.org/abs/1805.12514)] | From [convex_adversarial](https://github.com/locuslab/convex_adversarial/) |
| Incomplete > Linear Relaxation > Multi-Neuron Relaxation     | [K-ReLU](https://files.sri.inf.ethz.ch/website/papers/neurips19_krelu.pdf) | From [ERAN](https://github.com/eth-sri/eran)                 |
| Incomplete > SDP                                             | [SDPVerify](https://arxiv.org/abs/1811.01057)                | Reimplementation                                             |
| Incomplete > SDP                                             | [LMIVerify](https://arxiv.org/abs/1903.01287)                | Reimplementation                                             |
| Incomplete > Lipschitz > General Lipschitz                   | [Op-Norm](https://arxiv.org/abs/1312.6199)                   | From [RecurJac-and-CROWN](https://github.com/huanzhang12/RecurJac-and-CROWN) |
| Incomplete > Lipschitz > General Lipschitz                   | [FastLip](https://arxiv.org/abs/1804.09699v1)                | From [RecurJac-and-CROWN](https://github.com/huanzhang12/RecurJac-and-CROWN) |
| Incomplete > Lipschitz > General Lipschitz                   | [RecurJac](https://arxiv.org/abs/1810.11783)                 | From [RecurJac-and-CROWN](https://github.com/huanzhang12/RecurJac-and-CROWN) |
" | markdownify | liquify
}}

      </div>
    </div>
  </div>

  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingTwo">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseTwo" aria-expanded="false" aria-controls="panelsStayOpen-collapseTwo">
        Neural Network Models
      </button>
    </h2>
    <div id="panelsStayOpen-collapseTwo" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingTwo">
      <div class="accordion-body">

{{"

#### Architectures

We choose 7 network architectures for evaluation, including 3 fully-connected neural networks (`FCNNa`, `FCNNb`, `FCNNc`) and 4 convolutional neural networks (`CNNa`, `CNNb`, `CNNc`, `CNNd`).
These architectures are picked from the literature and adapted for each evaluated dataset MNIST and CIFAR-10.
        
|                  | FCNNa                                                        | FCNNb                                                        | FCNNc                                                        | CNNa                                                         | CNNb                                                         | CNNc                                                  | CNNd                                                         |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| Size on MNIST    | 50 neurons<br />16,330 params                                | 310 neurons<br />99,710 params                               | 7,178 neurons<br />7,111,690 params                          | 4,814 neurons<br />166,406 params                            | 24,042 neurons<br />833,786 params                           | 48,074 neurons<br />1,974,762 params                  | 176,138 neurons<br />13,314,634 params                       |
| Size on CIFAR-10 | 50 neurons<br />62,090 params                                | 310 neurons<br />328,510 params                              | 7,178 neurons<br />9,454,602 params                          | 6,254 neurons<br />214,918 params                            | 31,242 neurons<br />1,079,834 params                         | 62,474 neurons<br />2,466,858 params                  | 229,898 neurons<br />17,247,946 params                       |
| Structure        | Flatten -> 3 FCs                                             | Flatten -> 4 FCs                                             | Flatten -> 8 FCs                                             | 2 Convs -> Flatten -> 2 FCs                                  | 4 Convs -> Flatten -> 2 FCs                                  | 4 Convs -> Flatten -> 3 FCs                           | 5 Convs -> Flatten -> 2 FCs                                  |
| Source           | \\(2 \times [20]\\) from [[1](https://arxiv.org/abs/1804.09699v4),[2](https://arxiv.org/abs/1810.11783)] | \\(3 \times [100]\\) enlarged from [[1](https://arxiv.org/abs/1804.09699v4),[2](https://arxiv.org/abs/1810.11783)] | \\(7 \times [1024]\\) from [[1](https://arxiv.org/abs/1804.09699v4),[3](https://arxiv.org/abs/1811.00866)] | Conv-Small in [[4](https://arxiv.org/abs/1711.00851), [5](https://arxiv.org/abs/1805.12514)] | Half-sized Conv-Large in [[5](https://arxiv.org/abs/1805.12514)] | Conv-Large in [[5](https://arxiv.org/abs/1805.12514)] | Double-sized Conv-Large in [[5](https://arxiv.org/abs/1805.12514)] |


#### Training Methods
"  | markdownify | liquify
}}

<ul class="nav nav-pills mb-3" id="myTab" role="tablist">
  <li class="nav-item" role="presentation">
    <button class="nav-link active" id="training-mnist-tab" data-bs-toggle="pill" data-bs-target="#training-mnist" type="button" role="tab" aria-controls="home" aria-selected="true">MNIST</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="training-cifar10-tab" data-bs-toggle="pill" data-bs-target="#training-cifar10" type="button" role="tab" aria-controls="profile" aria-selected="false">CIFAR-10</button>
  </li>
</ul>

<div class="tab-content" id="myTabContent" style="background-color:#f8f9fa">
  <div class="tab-pane fade show active" id="training-mnist" role="tabpanel" aria-labelledby="training-mnist-tab">
{{    "

For each of the 7 architectures, the model is trained by 5 approaches, yielding 5 concrete models:

- ``clean``: standard training with cross-entropy loss

- ``adv1``: [PGD adversarial training](https://arxiv.org/abs/1706.06083) with \\(\epsilon=0.1\\) under \\(\ell_\infty\\) adversary

- ``adv3``: [PGD adversarial training](https://arxiv.org/abs/1706.06083) with \\(\epsilon=0.3\\) under \\(\ell_\infty\\) adversary 


- ``cadv1``: [CROWN-IBP training](https://arxiv.org/abs/1906.06316) with \\(\epsilon=0.1\\) under \\(\ell_\infty\\) adversary

- ``cadv3``: [CROWN-IBP training](https://arxiv.org/abs/1906.06316) with \\(\epsilon=0.3\\) under \\(\ell_\infty\\) adversary 

" | markdownify | liquify }}
  </div>
  <div class="tab-pane fade" id="training-cifar10" role="tabpanel" aria-labelledby="training-cifar10-tab">
{{    "
For each of the 7 architectures, the model is trained by 5 approaches, yielding 5 concrete models:

- ``clean``: standard training with cross-entropy loss

- ``adv2``: [PGD adversarial training](https://arxiv.org/abs/1706.06083) with \\(\epsilon=2/255\\) under \\(\ell_\infty\\) adversary

- ``adv8``: [PGD adversarial training](https://arxiv.org/abs/1706.06083) with \\(\epsilon=8/255\\) under \\(\ell_\infty\\) adversary 


- ``cadv2``: [CROWN-IBP training](https://arxiv.org/abs/1906.06316) with \\(\epsilon=2/255\\) under \\(\ell_\infty\\) adversary

- ``cadv8``: [CROWN-IBP training](https://arxiv.org/abs/1906.06316) with \\(\epsilon=8/255\\) under \\(\ell_\infty\\) adversary 


" | markdownify | liquify }}
  </div>
</div>


{{ "

We choose these training
configurations to reflect three common types of models on which
verification approaches are used: vanilla (undefended) models, empirical defense models and
certification-oriented trained models. All models are trained to
reach their expected robustness.

#### Clean and Empirical Robust Accuracies
  We report the clean accuracies and empirical robust accuracies of all these models.

  **Clean accuracy**: accuracy on the whole original test set with no perturbation.

  **Empirical robust accuracy**: accuracy under 100-step PGD attack on the whole original test set, where the attack is under \\(\ell_\infty\\) norm and perturbation radius \\(\epsilon\\) is bounded by corresponding training radius (e.g., \\(0.1\\) for ``adv1`` and ``cadv1`` and \\(0.3\\) for ``adv3`` and ``cadv3``). The step size is \\(\epsilon/50\\) following the literature. Due to low robustness, we do not report empirical robust accuracy for ``clean`` models. 
" | markdownify | liquify}}
        
<div class="accordion accordion-flush" id="clean-empirical-acc">
  <div class="accordion-item">
    <h2 class="accordion-header" id="clean-empirical-acc-mnist-heading">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#clean-empirical-acc-mnist-collapse" aria-expanded="false" aria-controls="clean-empirical-acc-mnist-collapse" style='background-color:#f8f9fa'>
        MNIST Models
      </button>
    </h2>
    <div id="clean-empirical-acc-mnist-collapse" class="accordion-collapse collapse" aria-labelledby="clean-empirical-acc-mnist-heading" data-bs-parent="#clean-empirical-acc">
      <br>
      {{
        "

##### Clean Accuracy

|           | ``FCNNa`` | ``FCNNb`` | ``FCNNc`` | ``CNNa`` | ``CNNb`` | ``CNNc`` | ``CNNd`` |
| --------- | --------- | --------- | --------- | -------- | -------- | -------- | -------- |
| ``clean`` | 93.63%    | 96.12%    | 95.05%    | 98.48%   | 98.85%   | 98.85%   | 99.20%   |
| ``adv1``  | 93.36%    | 97.12%    | 98.00%    | 99.01%   | 99.33%   | 99.24%   | 99.37%   |
| ``cadv1`` | 88.35%    | 95.23%    | 96.89%    | 98.52%   | 98.87%   | 98.83%   | 99.13%   |
| ``adv3``  | 76.77%    | 89.96%    | 83.27%    | 98.20%   | 99.01%   | 99.16%   | 99.38%   |
| ``cadv3`` | 45.79%    | 76.21%    | 35.87%    | 96.26%   | 98.14%   | 98.11%   | 98.58%   |

##### Empirical Robust Accuracy

|                             | ``FCNNa`` | ``FCNNb`` | ``FCNNc`` | ``CNNa`` | ``CNNb`` | ``CNNc`` | ``CNNd`` |
| --------------------------- | --------- | --------- | --------- | -------- | -------- | -------- | -------- |
| ``clean``                   | /         | /         | /         | /        | /        | /        | /        |
| ``adv1``(\\(\epsilon=0.1\\))  | 78.39%    | 85.18%    | 87.40%    | 95.48%   | 96.09%   | 96.35%   | 97.64%   |
| ``cadv1``(\\(\epsilon=0.1\\)) | 80.99%    | 90.67%    | 93.43%    | 97.18%   | 98.12%   | 98.41%   | 98.50%   |
| ``adv3``(\\(\epsilon=0.3\\))  | 31.67%    | 33.30%    | 33.31%    | 86.06%   | 92.84%   | 93.47%   | 95.09%   |
| ``cadv3``(\\(\epsilon=0.3\\))  | 41.35%    | 69.90%    | 34.08%    | 93.06%   | 95.88%   | 96.10%   | 96.72%   |

      " | markdownify | liquify}}
    </div>
  </div>
  <div class="accordion-item">
    <h2 class="accordion-header" id="clean-empirical-acc-cifar10-heading">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#clean-empirical-acc-cifar10-collapse" aria-expanded="false" aria-controls="clean-empirical-acc-cifar10-collapse" style='background-color:#f8f9fa'>
        CIFAR-10 Models
      </button>
    </h2>
    <div id="clean-empirical-acc-cifar10-collapse" class="accordion-collapse collapse" aria-labelledby="clean-empirical-acc-cifar10-heading" data-bs-parent="#clean-empirical-acc">
      <br>
      {{
        "

##### Clean Accuracy

|           | ``FCNNa`` | ``FCNNb`` | ``FCNNc`` | ``CNNa`` | ``CNNb`` | ``CNNc`` | ``CNNd`` |
| --------- | --------- | --------- | --------- | -------- | -------- | -------- | -------- |
| ``clean`` | 38.46%    | 41.76%    | 46.37%    | 59.45%   | 65.65%   | 58.60%   | 83.53%   |
| ``adv2``  | 41.17%    | 44.31%    | 36.19%    | 60.40%   | 68.66%   | 65.59%   | 83.65%   |
| ``cadv8`` | 38.97%    | 44.18%    | 46.34%    | 54.78%   | 58.81%   | 59.46%   | 60.74%   |
| ``adv2``  | 34.13%    | 37.68%    | 25.21%    | 49.42%   | 55.15%   | 54.01%   | 72.25%   |
| ``cadv8`` | 30.59%    | 32.05%    | 30.30%    | 40.51%   | 40.05%   | 40.11%   | 40.61%   |

##### Empirical Robust Accuracy

|                                 | ``FCNNa`` | ``FCNNb`` | ``FCNNc`` | ``CNNa`` | ``CNNb`` | ``CNNc`` | ``CNNd`` |
| ------------------------------- | --------- | --------- | --------- | -------- | -------- | -------- | -------- |
| ``clean``                       | /         | /         | /         | /        | /        | /        | /        |
| ``adv2``(\\(\epsilon=2/255\\))  | 39.49%    | 42.99%    | 35.87%    | 57.39%   | 64.01%   | 62.05%   | 76.41%   |
| ``cadv2``(\\(\epsilon=2/255\\)) | 38.71%    | 43.83%    | 45.42%    | 54.17%   | 58.76%   | 59.25%   | 59.88%   |
| ``adv8``(\\(\epsilon=8/255\\))  | 31.36%    | 34.01%    | 25.87%    | 41.87%   | 45.35%   | 45.91%   | 53.86%   |
| ``cadv8``(\\(\epsilon=8/255\\)) | 30.36%    | 31.40%    | 29.59%    | 39.93%   | 39.36%   | 39.50%   | 39.94%   |

      " | markdownify | liquify}}
    </div>
  </div>
</div>



      </div>
    </div>
  </div>
  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingFive">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseFive" aria-expanded="false" aria-controls="panelsStayOpen-collapseFive">
        Datasets and Evaluation Protocols
      </button>
    </h2>
    <div id="panelsStayOpen-collapseFive" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingFive">
      <div class="accordion-body">
{{"

#### Dataset
  We evaluate on two datasets: MNIST and CIFAR-10 (in SoK paper we only present results on CIFAR-10).
  We do not consider larger datasets like TinyImageNet, because most verification approaches cannot handle models for that scale, or can only provide low robust accuracy (e.g., \\(20\%\\) against \\(\ell_\infty\\) adversary with \\(1/255\\) attack radius).

  MNIST is a set of \\(28 \times 28\\) gray-scale images.

  CIFAR-10 is a set of \\(3 \times 32 \times 32\\) images.

#### Adversary

We focus on \\(\ell_\infty\\) adversary since this type is supported by most number of verification approaches to the best of our knowledge.
On MNIST, we evaluate under attack radii \\(0.02\\), \\(0.1\\), and \\(0.3\\).
On CIFAR-10, we evaluate under attack radii \\(0.5/255\\), \\(2/255\\), and \\(8/255\\).

#### Metrics

On each dataset, we uniformly sample 100 test set samples as the fixed set for evaluation.
We evaluate by two metrics: **certified accuracy** and **average certified robustness radius**(in SoK we only report certified accuracy due to space limit).

- **Certified Accuracy**: As described on the frontpage, 
``certified accuracy = # samples verified to be robust / number of all evaluated samples`` under given \\((\ell_\infty, \epsilon)\\)-adversary.
We also report the robust accuracy under empirical attack (PGD attack), which gives an upper bound of robust accuracy, so
we can estimate the gap between certified robust accuracy and accuracy of existing attack.
The **time limit** is ``60 s`` per instance, and we count timeout instances as \`\`non-robust\`\`.

- **Average Certified Robustness Radius**: 
We also evaluate the verification approaches by measuring their average certified
robustness radius. The average certified robustness radius (\\(\bar r\\)) stands for the average \\(\ell_\infty\\) radius the verification approach can
verify on the given subset of test set samples. We use the same uniformly sampled sets as in certified accuracy evaluation. To
determine the best certified radius of each verification approach, we conduct a binary search process due to the monotonicity.
Specifically, we do binary search on interval \\([0, 0.5]\\) because the largest possible radius is \\(0.5\\) for \\([0, 1]\\) bounded MNIST and CIFAR-10 inputs. If
current radius \\(mid\\) is verified to be robust, we update current best by \\(mid\\) and let \\(l \gets mid\\), if current radius mid cannot be
verified, we let \\(r \gets mid\\), until we reach the precision \\(0.01\\) on MNIST or \\(0.001\\) on CIFAR-10, or time is up. For the evaluation
of average certified robustness radius, since it involves multiple evaluations because of binary search, we set the running **time limit**
to ``120 s`` per input and record the highest certified radius it has been verified before the time is used up. The average certified
robustness radius is evaluated on the same subset of test set samples as used for robust accuracy evaluation.
We also report the smallest radius of adversarial samples found by empirical attack (PGD attack), which gives an upper
bound of certified robustness radius for us to estimate the gap.

#### Experimental Environment

For MNIST experiments, we run the evaluation on 24-core Intel Xeon E5-2650 CPU running at
2.20 GHz with single NVIDIA GeForce GTX 1080 Ti GPU. For CIFAR-10 experiments, we run the evaluation on 24-core
Intel Xeon Platinum 8259CL CPU running at 2.50 GHz with single NVIDIA Tesla T4 GPU. The CIFAR-10 experiments use
slightly faster running environment due to its larger scale.

 "
  | markdownify | liquify
}}
      </div>
    </div>
  </div>
  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingThree">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseThree" aria-expanded="false" aria-controls="panelsStayOpen-collapseThree">
        Full Results
      </button>
    </h2>
    <div id="panelsStayOpen-collapseThree" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingThree">
      <div class="accordion-body">

<!-- Main START -->
  {{"Choose the dataset and model training methods:" | markdownify | liquify}}

<ul class="nav nav-pills mb-3" id="myTab" role="tablist" style="background-color:#f8f9fa; margin-left: 0px;">
  <li class="nav-item" role="presentation">
    <button class="nav-link active" id="main-mnist-002-tab" data-bs-toggle="pill" data-bs-target="#main-mnist-002" type="button" role="tab" aria-controls="home" aria-selected="true">MNIST <br> clean</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="main-mnist-01-tab" data-bs-toggle="pill" data-bs-target="#main-mnist-01" type="button" role="tab" aria-controls="profile" aria-selected="false">MNIST <br> adv1 and cadv1</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="main-mnist-03-tab" data-bs-toggle="pill" data-bs-target="#main-mnist-03" type="button" role="tab" aria-controls="profile" aria-selected="false">MNIST <br> adv3 and cadv3</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="main-cifar-05-tab" data-bs-toggle="pill" data-bs-target="#main-cifar-05" type="button" role="tab" aria-controls="home" aria-selected="true">CIFAR-10 <br> clean</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="main-cifar-2-tab" data-bs-toggle="pill" data-bs-target="#main-cifar-2" type="button" role="tab" aria-controls="profile" aria-selected="false">CIFAR-10 <br> adv2 and cadv2</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="main-cifar-8-tab" data-bs-toggle="pill" data-bs-target="#main-cifar-8" type="button" role="tab" aria-controls="profile" aria-selected="false">CIFAR-10 <br> adv8 and cadv8</button>
  </li>
</ul>

<div class="tab-content" id="mainTabContent">
  <div class="tab-pane fade show active" id="main-mnist-002" role="tabpanel" aria-labelledby="main-mnist-002-tab">
    <nav>
      <div class="nav nav-tabs" id="nav-main-mnist-002" role="tablist">
        <button class="nav-link active" id="nav-main-mnist-002-acc-tab" data-bs-toggle="tab" data-bs-target="#nav-main-mnist-002-acc" type="button" role="tab" aria-controls="nav-home" aria-selected="true">Certified Accuracy under \(\epsilon=0.02\)</button>
        <button class="nav-link" id="nav-main-mnist-002-rad-tab" data-bs-toggle="tab" data-bs-target="#nav-main-mnist-002-rad" type="button" role="tab" aria-controls="nav-profile" aria-selected="false">Average Certified Robustness Radius</button>
      </div>
    </nav>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane fade show active" id="nav-main-mnist-002-acc" role="tabpanel" aria-labelledby="nav-main-mnist-002-acc-tab">
        <!-- MNIST, \(\epsilon=0.02\), certified accuracy -->

        {{"
**Bolded** numbers mark the highest robust accuracies among verification approaches.
*0%* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.

|                    | FCNNa   | FCNNb   | FCNNc | CNNa    | CNNb    | CNNc   | CNNd   |
| ------------------ | ------- | ------- | ----- | ------- | ------- | ------ | ------ |
| Bounded MILP       | **80%** | 84%     | *0%*  | 44%     | *0%*    | *0%*   | *0%*   |
| AI2                | **80%** | **86%** | *0%*  | **89%** | **90%** | *0%*   | *0%*   |
| LP-Full            | 70%     | 68%     | *0%*  | *0%*    | *0%*    | *0%*   | *0%*   |
| IBP                | 3%      | *0%*    | *0%*  | *0%*    | *0%*    | *0%*   | *0%*   |
| Fast-Lin           | 68%     | 56%     | *0%*  | 72%     | 28%     | 5%     | *0%*   |
| CROWN              | 79%     | 85%     | *0%*  | **89%** | 90%     | *0%*   | *0%*   |
| CNN-Cert           | 70%     | 67%     | *0%*  | 73%     | 44%     | *0%*   | *0%*   |
| CROWN-IBP          | 77%     | *0%*    | *0%*  | 13%     | *0%*    | *0%*   | *0%*   |
| DeepPoly           | 79%     | 85%     | *0%*  | **89%** | **90%** | *0%*   | *0%*   |
| RefineZono         | 79%     | *0%*    | *0%*  | *0%*    | *0%*    | *0%*   | *0%*   |
| WK                 | 68%     | 56%     | *0%*  | 72%     | 28%     | **5%** | *0%*   |
| K-ReLU             | 79%     | 85%     | *0%*  | **89%** | *0%*    | *0%*   | *0%*   |
| SDPVerify          | *0%*    | *0%*    | *0%*  | *0%*    | *0%*    | *0%*   | *0%*   |
| LMIVerify          | *0%*    | *0%*    | *0%*  | *0%*    | *0%*    | *0%*   | *0%*   |
| Op-Norm            | *0%*    | *0%*    | *0%*  | *0%*    | *0%*    | *0%*   | *0%*   |
| FastLip            | 60%     | 4%      | *0%*  | 6%      | *0%*    | *0%*   | *0%*   |
| RecurJac           | 70%     | 52%     | *0%*  | 33%     | *0%*    | *0%*   | *0%*   |
| *PGD Upper Bound*  | *80%*   | *86%*   | *85%* | *92%*   | *95%*   | *93%*  | *97%*  |
| *Clean Acc.*       | *89%*   | *92%*   | *95%* | *99%*   | *100%*  | *100%* | *100%* |



**Average running time for single-instance robustness verification** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``60 s'' per instance.)



|                    | FCNNa   | FCNNb   | FCNNc   | CNNa    | CNNb    | CNNc    | CNNd    |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Bounded MILP       | 0.37    | 12.67   | *60.00* | 54.28   | *60.00* | *60.00* | *60.00* |
| AI2                | 0.09    | 1.09    | *60.00* | 7.92    | 48.89   | *60.00* | *60.00* |
| LP-Full            | 0.41    | 15.22   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| IBP                | 0.00    | 0.00    | 0.01    | 0.00    | 0.01    | 0.01    | 0.01    |
| Fast-Lin           | 0.03    | 0.03    | 1.71    | 0.35    | 4.70    | 15.95   | *60.00* |
| CROWN              | 0.01    | 0.01    | 0.03    | 0.02    | 0.13    | *60.00* | *60.00* |
| CNN-Cert           | 0.12    | 0.43    | *60.00* | 1.13    | 28.40   | 59.67   | *60.00* |
| CROWN-IBP          | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    |
| DeepPoly           | 0.03    | 0.25    | *60.00* | 2.45    | 49.04   | *60.00* | *60.00* |
| RefineZono         | 3.40    | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| WK                 | 0.01    | 0.01    | 0.04    | 0.01    | 0.03    | 0.08    | *60.00* |
| K-ReLU             | 12.49   | 26.05   | *60.00* | 30.49   | *60.00* | *60.00* | *60.00* |
| SDPVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| LMIVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| Op-Norm            | 0.00    | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | *60.00* |
| FastLip            | 0.20    | 0.20    | 1.68    | 5.81    | 25.47   | 54.26   | *60.00* |
| RecurJac           | 0.81    | 2.25    | 43.29   | 50.35   | *60.00* | *60.00* | *60.00* |
| *PGD Attack*       | 0.26    | 0.30    | 0.42    | 0.33    | 0.43    | 0.46    | 0.51    |
| *Normal Inference* | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    |



        "
  | markdownify | liquify
}}



      </div>
      <div class="tab-pane fade" id="nav-main-mnist-002-rad" role="tabpanel" aria-labelledby="nav-main-mnist-002-rad-tab">
        <!-- MNIST, \(\epsilon=0.02\), certified radius -->

        {{"
**Bolded** numbers mark the highest certified radii among verification approaches.
*0.000* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.

|                   | FCNNa     | FCNNb     | FCNNc     | CNNa      | CNNb      | CNNc      | CNNd    |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- | ------- |
| Bounded MILP      | **0.054** | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| AI2               | **0.054** | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| LP-Full           | 0.027     | 0.019     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| IBP               | 0.006     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| Fast-Lin          | 0.026     | 0.018     | *0.000*   | 0.021     | 0.014     | **0.009** | *0.000* |
| CROWN             | 0.045     | **0.035** | **0.005** | **0.039** | **0.030** | *0.000*   | *0.000* |
| CNN-Cert          | 0.027     | 0.020     | *0.000*   | 0.022     | 0.015     | *0.000*   | *0.000* |
| CROWN-IBP         | 0.033     | 0.006     | *0.000*   | 0.011     | *0.000*   | *0.000*   | *0.000* |
| DeepPoly          | 0.045     | **0.035** | *0.000*   | **0.039** | *0.000*   | *0.000*   | *0.000* |
| RefineZono        | 0.045     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| WK                | 0.026     | 0.018     | *0.000*   | 0.021     | 0.014     | **0.009** | *0.000* |
| K-ReLU            | 0.046     | 0.001     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| SDPVerify         | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| LMIVerify         | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| Op-Norm           | 0.003     | 0.001     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* |
| FastLip           | 0.020     | 0.008     | *0.000*   | 0.009     | *0.000*   | *0.000*   | *0.000* |
| RecurJac          | 0.030     | 0.021     | *0.000*   | 0.009     | *0.000*   | *0.000*   | *0.000* |
| *PGD Upper Bound* | *0.057*   | *0.060*   | *0.046*   | *0.073*   | *0.087*   | *0.090*   | *0.103* |


**Average running time for robustness radius computation** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``120 s'' per instance.)


|              | FCNNa    | FCNNb    | FCNNc    | CNNa     | CNNb     | CNNc     | CNNd     |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Bounded MILP | 3.14     | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| AI2          | 3.95     | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LP-Full      | 2.63     | 113.84   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| IBP          | 0.02     | 0.02     | 0.03     | 0.02     | 0.03     | 0.04     | 0.05     |
| Fast-Lin     | 0.04     | 0.17     | 12.30    | 2.04     | 32.25    | 103.94   | *120.00* |
| CROWN        | 0.03     | 0.05     | 0.17     | 0.09     | 0.80     | *120.00* | *120.00* |
| CNN-Cert     | 0.11     | 1.19     | *120.00* | 3.23     | 75.62    | *120.00* | *120.00* |
| CROWN-IBP    | 0.03     | 0.03     | 0.07     | 0.04     | 0.06     | 0.08     | 0.08     |
| DeepPoly     | 0.17     | 1.50     | *120.00* | 13.42    | *120.00* | *120.00* | *120.00* |
| RefineZono   | 32.98    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| WK           | 0.04     | 0.06     | 0.22     | 0.08     | 0.32     | 1.06     | *120.00* |
| K-ReLU       | 70.72    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| SDPVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LMIVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| Op-Norm      | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.01     | *120.00* |
| FastLip      | 0.15     | 0.57     | 9.95     | 19.23    | 111.50   | *120.00* | *120.00* |
| RecurJac     | 6.81     | 29.75    | *120.00* | 114.36   | *120.00* | *120.00* | *120.00* |
| *PGD Attack* | 1.61     | 1.81     | 2.59     | 2.03     | 2.68     | 2.85     | 3.13     |

        "
  | markdownify | liquify
}}
      </div>
    </div>
  </div>

  <div class="tab-pane fade" id="main-mnist-01" role="tabpanel" aria-labelledby="main-mnist-01-tab">
    <nav>
      <div class="nav nav-tabs" id="nav-main-mnist-01" role="tablist">
        <button class="nav-link active" id="nav-main-mnist-01-acc-tab" data-bs-toggle="tab" data-bs-target="#nav-main-mnist-01-acc" type="button" role="tab" aria-controls="nav-home" aria-selected="true">Certified Accuracy under \(\epsilon=0.1\)</button>
        <button class="nav-link" id="nav-main-mnist-01-rad-tab" data-bs-toggle="tab" data-bs-target="#nav-main-mnist-01-rad" type="button" role="tab" aria-controls="nav-profile" aria-selected="false">Average Certified Robustness Radius</button>
      </div>
    </nav>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane fade show active" id="nav-main-mnist-01-acc" role="tabpanel" aria-labelledby="nav-main-mnist-01-acc-tab">

        <!-- MNIST, \(\epsilon=0.1\), certified accuracy -->

        {{"
**Bolded** numbers mark the highest robust accuracies among verification approaches.
*0%* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.

|                   | FCNNa   | FCNNa   | FCNNb  | FCNNb   | FCNNc | FCNNc   | CNNa    | CNNa    | CNNb   | CNNb    | CNNc   | CNNc    | CNNd   | CNNd    |
| -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- |
|  | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 |
| Bounded MILP | **70%** | **68%** | *0%* | **85%** | *0%* | 67% | *0%* | **95%** | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| AI2 | **70%** | **68%** | 1% | **85%** | *0%* | 52% | 6% | **95%** | 1% | 93% | *0%* | *0%* | *0%* | *0%* |
| LP-Full | 8% | 46% | *0%* | 57% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| IBP | *0%* | **68%** | *0%* | **85%** | *0%* | **80%** | *0%* | **95%** | *0%* | 91% | *0%* | **89%** | *0%* | 90% |
| Fast-Lin | 5% | 35% | *0%* | 36% | *0%* | *0%* | *0%* | 90% | *0%* | 89% | *0%* | 86% | *0%* | *0%* |
| CROWN | 47% | 64% | **4%** | 72% | *0%* | 52% | **32%** | 93% | **2%** | 93% | *0%* | *0%* | *0%* | *0%* |
| CNN-Cert | 5% | 36% | *0%* | 44% | *0%* | *0%* | *0%* | 92% | *0%* | 93% | *0%* | *0%* | *0%* | *0%* |
| CROWN-IBP | 2% | 67% | *0%* | 76% | *0%* | 73% | *0%* | 93% | *0%* | **94%** | *0%* | 88% | *0%* | **94%** |
| DeepPoly | 47% | 64% | **4%** | 72% | *0%* | 52% | **32%** | 93% | **2%** | 93% | *0%* | *0%* | *0%* | *0%* |
| RefineZono | 49% | 63% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| WK | 5% | 35% | *0%* | 36% | *0%* | *0%* | *0%* | 90% | *0%* | 89% | *0%* | 86% | *0%* | 1% |
| K-ReLU | 51% | **68%** | **4%** | **85%** | *0%* | *0%* | 3% | **95%** | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| SDPVerify | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| LMIVerify | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| Op-Norm | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| FastLip | *0%* | 30% | *0%* | 31% | *0%* | 2% | *0%* | 86% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| RecurJac | 2% | 30% | *0%* | 31% | *0%* | *0%* | *0%* | 85% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| *PGD Upper Bound* | *71%* | *69%* | *80%* | *85%* | *82%* | *89%* | *94%* | *95%* | *95%* | *97%* | *95%* | *96%* | *97%* | *99%* |
| *Clean Acc.* | *92%* | *88%* | *97%* | *97%* | *98%* | *96%* | *100%* | *99%* | *100%* | *99%* | *100%* | *100%* | *100%* | *100%* |


**Average running time for single-instance robustness verification** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``60 s'' per instance.)


|                  | FCNNa   | FCNNa   | FCNNb   | FCNNb   | FCNNc   | FCNNc   | CNNa    | CNNa    | CNNb    | CNNb    | CNNc    | CNNc    | CNNd    | CNNd    |
| ---------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
|                  | adv1    | cadv1   | adv1    | cadv1   | adv1    | cadv1   | adv1    | cadv1   | adv1    | cadv1   | adv1    | cadv1   | adv1    | cadv1   |
| Bounded MILP     | 0.73    | 0.28    | *60.00* | 1.32    | *60.00* | 53.35   | *60.00* | 6.86    | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| AI2             | 0.52    | 0.23    | 59.38   | 1.12    | *60.00* | 31.50   | 56.49   | 2.61    | 59.80   | 25.03   | *60.00* | *60.00* | *60.00* | *60.00* |
| LP-Full          | 0.43    | 0.35    | 23.59   | 9.57    | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| IBP              | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | 0.01    | 0.00    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    |
| Fast-Lin         | 0.01    | 0.01    | 0.03    | 0.02    | 2.06    | 1.71    | 0.39    | 0.41    | 5.63    | 4.80    | 17.31   | 19.13   | *60.00* | *60.00* |
| CROWN            | 0.01    | 0.01    | 0.01    | 0.01    | 0.03    | 0.03    | 0.02    | 0.02    | 0.13    | 0.13    | *60.00* | *60.00* | *60.00* | *60.00* |
| CNN-Cert         | 0.02    | 0.03    | 0.05    | 0.31    | *60.00* | *60.00* | 0.21    | 1.35    | 4.18    | 36.00   | *60.00* | *60.00* | *60.00* | *60.00* |
| CROWN-IBP        | 0.00    | 0.00    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    |
| DeepPoly         | 0.03    | 0.02    | 0.24    | 0.15    | *60.00* | 7.97    | 1.61    | 1.34    | 41.59   | 23.53   | *60.00* | *60.00* | *60.00* | *60.00* |
| RefineZono       | 7.27    | 14.51   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| WK               | 0.01    | 0.01    | 0.01    | 0.01    | 0.04    | 0.04    | 0.01    | 0.01    | 0.07    | 0.03    | 0.21    | 0.09    | *60.00* | 59.41   |
| K-ReLU           | 12.87   | 11.92   | 34.56   | 38.43   | *60.00* | *60.00* | 59.35   | 34.48   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| SDPVerify        | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| LMIVerify        | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| Op-Norm          | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | *60.00* | *60.00* |
| FastLip          | 0.02    | 0.03    | 0.05    | 0.12    | 1.17    | 6.04    | 2.52    | 5.51    | 20.05   | *60.00* | 46.69   | *60.00* | *60.00* | *60.00* |
| RecurJac         | 0.36    | 0.35    | 0.88    | 1.77    | 24.96   | 57.75   | 28.61   | 49.91   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| *PGD Attack*       | 0.26    | 0.30    | 0.30    | 0.33    | 0.42    | 0.48    | 0.33    | 0.37    | 0.44    | 0.48    | 0.47    | 0.50    | 0.52    | 0.54    |
| *Normal Inference* | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    |



        "
  | markdownify | liquify
}}

      </div>
      <div class="tab-pane fade" id="nav-main-mnist-01-rad" role="tabpanel" aria-labelledby="nav-main-mnist-01-rad-tab">
      {{
        "

**Bolded** numbers mark the highest certified radii among verification approaches.
*0.000* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.

|                   | FCNNa     | FCNNa     | FCNNb     | FCNNb     | FCNNc     | FCNNc     | CNNa      | CNNa      | CNNb      | CNNb      | CNNc      | CNNc      | CNNd    | CNNd      |
| ---- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- |
|      | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 | adv1 | cadv1 |
| Bounded MILP | **0.135** | **0.131** | *0.000* | **0.150** | *0.000* | *0.000* | *0.000* | 0.122 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| AI2 | **0.135** | **0.131** | *0.000* | **0.150** | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| LP-Full | 0.063 | 0.097 | 0.040 | 0.102 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| IBP | 0.011 | 0.128 | *0.000* | 0.142 | *0.000* | **0.143** | 0.005 | **0.154** | *0.000* | 0.174 | *0.000* | 0.175 | *0.000* | 0.168 |
| Fast-Lin | 0.060 | 0.083 | 0.036 | 0.080 | 0.007 | 0.056 | 0.061 | 0.142 | 0.051 | 0.149 | **0.041** | 0.138 | *0.000* | *0.000* |
| CROWN | 0.095 | 0.120 | **0.064** | 0.118 | **0.019** | 0.096 | **0.086** | 0.152 | **0.075** | 0.168 | *0.000* | *0.000* | *0.000* | *0.000* |
| CNN-Cert | 0.062 | 0.084 | 0.039 | 0.085 | *0.000* | *0.000* | 0.066 | 0.150 | 0.056 | 0.161 | *0.000* | *0.000* | *0.000* | *0.000* |
| CROWN-IBP | 0.055 | 0.123 | 0.009 | 0.135 | *0.000* | 0.133 | 0.030 | **0.154** | 0.001 | **0.176** | *0.000* | **0.179** | *0.000* | **0.174** |
| DeepPoly | 0.095 | 0.120 | **0.064** | 0.118 | *0.000* | 0.096 | **0.086** | 0.152 | 0.008 | 0.160 | *0.000* | *0.000* | *0.000* | *0.000* |
| RefineZono | 0.097 | 0.120 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| WK | 0.060 | 0.083 | 0.036 | 0.080 | 0.007 | 0.056 | 0.061 | 0.142 | 0.051 | 0.149 | **0.041** | 0.139 | *0.000* | *0.000* |
| K-ReLU | 0.100 | 0.129 | 0.014 | 0.097 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| SDPVerify | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| LMIVerify | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| Op-Norm | 0.002 | 0.002 | 0.001 | 0.002 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| FastLip | 0.041 | 0.078 | 0.016 | 0.074 | *0.000* | 0.052 | 0.032 | 0.131 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| RecurJac | 0.060 | 0.081 | 0.035 | 0.078 | *0.000* | *0.000* | 0.032 | 0.107 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| *PGD Upper Bound* | *0.145* | *0.157* | *0.155* | *0.190* | *0.151* | *0.213* | *0.163* | *0.194* | *0.168* | *0.233* | *0.169* | *0.278* | *0.181* | *0.320* |

**Average running time for robustness radius computation** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``120 s'' per instance.)


|              | FCNNa    | FCNNa    | FCNNb    | FCNNb    | FCNNc    | FCNNc    | CNNa     | CNNa     | CNNb     | CNNb     | CNNc     | CNNc     | CNNd     | CNNd     |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
|              | adv1     | cadv1    | adv1     | cadv1    | adv1     | cadv1    | adv1     | cadv1    | adv1     | cadv1    | adv1     | cadv1    | adv1     | cadv1    |
| Bounded MILP | 4.97     | 2.39     | *120.00* | 14.06    | *120.00* | *120.00* | *120.00* | 118.88   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| AI2          | 5.28     | 6.44     | *120.00* | 39.72    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LP-Full      | 2.77     | 2.33     | 107.16   | 73.58    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| IBP          | 0.02     | 0.02     | 0.02     | 0.03     | 0.03     | 0.04     | 0.02     | 0.03     | 0.03     | 0.04     | 0.04     | 0.05     | 0.05     | 0.06     |
| Fast-Lin     | 0.03     | 0.04     | 0.16     | 0.17     | 11.18    | 9.15     | 2.11     | 2.62     | 32.73    | 28.51    | 96.23    | 119.36   | *120.00* | *120.00* |
| CROWN        | 0.03     | 0.03     | 0.05     | 0.05     | 0.17     | 0.18     | 0.09     | 0.09     | 0.80     | 0.80     | *120.00* | *120.00* | *120.00* | *120.00* |
| CNN-Cert     | 0.16     | 0.16     | 1.48     | 1.70     | *120.00* | *120.00* | 4.09     | 4.69     | 94.97    | 115.68   | *120.00* | *120.00* | *120.00* | *120.00* |
| CROWN-IBP    | 0.03     | 0.03     | 0.03     | 0.03     | 0.10     | 0.07     | 0.04     | 0.04     | 0.06     | 0.07     | 0.08     | 0.08     | 0.08     | 0.08     |
| DeepPoly     | 0.16     | 0.12     | 1.50     | 0.99     | *120.00* | 65.62    | 9.55     | 8.65     | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| RefineZono   | 56.38    | 100.87   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| WK           | 0.04     | 0.04     | 0.06     | 0.06     | 0.21     | 0.20     | 0.08     | 0.08     | 0.33     | 0.23     | 1.03     | 0.77     | *120.00* | *120.00* |
| K-ReLU       | 65.10    | 69.00    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| SDPVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LMIVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| Op-Norm      | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.01     | 0.01     | *120.00* | *120.00* |
| FastLip      | 0.16     | 0.16     | 0.67     | 0.69     | 8.59     | 46.72    | 20.61    | 25.84    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| RecurJac     | 5.87     | 5.62     | 29.77    | 28.10    | *120.00* | *120.00* | 119.21   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| *PGD Attack* | 1.70     | 2.04     | 1.91     | 2.25     | 2.67     | 2.96     | 2.11     | 2.33     | 2.74     | 2.97     | 2.85     | 3.21     | 3.16     | 3.40     |
      "
    | markdownify | liquify
}}

      </div>
    </div>
  </div>

  <div class="tab-pane fade" id="main-mnist-03" role="tabpanel" aria-labelledby="main-mnist-03-tab">
    <nav>
      <div class="nav nav-tabs" id="nav-main-mnist-03" role="tablist">
        <button class="nav-link active" id="nav-main-mnist-03-acc-tab" data-bs-toggle="tab" data-bs-target="#nav-main-mnist-03-acc" type="button" role="tab" aria-controls="nav-home" aria-selected="true">Certified Accuracy under \(\epsilon=0.3\)</button>
        <button class="nav-link" id="nav-main-mnist-03-rad-tab" data-bs-toggle="tab" data-bs-target="#nav-main-mnist-03-rad" type="button" role="tab" aria-controls="nav-profile" aria-selected="false">Average Certified Robustness Radius</button>
      </div>
    </nav>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane fade show active" id="nav-main-mnist-03-acc" role="tabpanel" aria-labelledby="nav-main-mnist-03-acc-tab">
        {{"
        
**Bolded** numbers mark the highest robust accuracies among verification approaches.
*0%* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.

|                   | FCNNa  | FCNNa   | FCNNb | FCNNb   | FCNNc | FCNNc   | CNNa  | CNNa    | CNNb   | CNNb    | CNNc  | CNNc    | CNNd   | CNNd    |
| -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- |
|  | adv3 | cadv3 | adv3 | cadv3 | adv3 | cadv3 | adv3 | cadv3 | adv3 | cadv3 | adv3 | cadv3 | adv3 | cadv3 |
| Bounded MILP | **6%** | **25%** | *0%* | **54%** | *0%* | 7% | *0%* | **88%** | *0%* | 84% | *0%* | *0%* | *0%* | *0%* |
| AI2 | **6%** | **25%** | *0%* | **54%** | *0%* | 16% | *0%* | 73% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| LP-Full | *0%* | 15% | *0%* | 8% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| IBP | *0%* | **25%** | *0%* | **54%** | *0%* | **22%** | *0%* | **88%** | *0%* | **85%** | *0%* | **87%** | *0%* | **89%** |
| Fast-Lin | *0%* | 4% | *0%* | 1% | *0%* | *0%* | *0%* | 6% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| CROWN | *0%* | 23% | *0%* | 27% | *0%* | 16% | *0%* | 40% | *0%* | 7% | *0%* | *0%* | *0%* | *0%* |
| CNN-Cert | *0%* | 13% | *0%* | 5% | *0%* | *0%* | *0%* | 39% | *0%* | 1% | *0%* | *0%* | *0%* | *0%* |
| CROWN-IBP | *0%* | 23% | *0%* | 45% | *0%* | 20% | *0%* | 76% | *0%* | 71% | *0%* | 75% | *0%* | 65% |
| DeepPoly | *0%* | 23% | *0%* | 27% | *0%* | 16% | *0%* | 40% | *0%* | 7% | *0%* | *0%* | *0%* | *0%* |
| RefineZono | *0%* | 21% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| WK | *0%* | 4% | *0%* | 1% | *0%* | *0%* | *0%* | 6% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| K-ReLU | *0%* | **25%** | *0%* | **54%** | *0%* | *0%* | *0%* | 57% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| SDPVerify | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| LMIVerify | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| Op-Norm | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| FastLip | *0%* | 1% | *0%* | 2% | *0%* | *0%* | *0%* | 2% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| RecurJac | *0%* | 1% | *0%* | 2% | *0%* | *0%* | *0%* | 2% | *0%* | *0%* | *0%* | *0%* | *0%* | *0%* |
| *PGD Upper Bound* | *19%* | *26%* | *26%* | *64%* | *26%* | *26%* | *81%* | *90%* | *91%* | *94%* | *90%* | *93%* | *92%* | *94%* |
| *Clean Acc.* | *75%* | *37%* | *88%* | *76%* | *78%* | *28%* | *97%* | *95%* | *100%* | *98%* | *99%* | *98%* | *100%* | *98%* |


**Average running time for single-instance robustness verification** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``60 s'' per instance.)


|                    | FCNNa   | FCNNa   | FCNNb   | FCNNb   | FCNNc   | FCNNc   | CNNa    | CNNa    | CNNb    | CNNb    | CNNc    | CNNc    | CNNd    | CNNd    |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
|                    | adv3    | cadv3   | adv3    | cadv3   | adv3    | cadv3   | adv3    | cadv3   | adv3    | cadv3   | adv3    | cadv3   | adv3    | cadv3   |
| Bounded MILP       | 0.63    | 0.30    | *60.00* | 1.39    | *60.00* | 52.52   | *60.00* | 5.61    | *60.00* | 50.35   | *60.00* | *60.00* | *60.00* | *60.00* |
| AI2                | 0.72    | 0.29    | *60.00* | 1.72    | *60.00* | 23.66   | *60.00* | 27.46   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| LP-Full            | 0.49    | 0.39    | 22.97   | 12.75   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| IBP                | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | 0.01    | 0.00    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    |
| Fast-Lin           | 0.01    | 0.01    | 0.03    | 0.03    | 2.14    | 1.96    | 0.34    | 0.44    | 5.66    | 5.28    | 18.63   | 20.04   | *60.00* | *60.00* |
| CROWN              | 0.01    | 0.01    | 0.01    | 0.01    | 0.03    | 0.03    | 0.02    | 0.02    | 0.13    | 0.13    | *60.00* | *60.00* | *60.00* | *60.00* |
| CNN-Cert           | 0.01    | 0.02    | 0.06    | 0.12    | *60.00* | *60.00* | 0.17    | 0.89    | 4.20    | 5.91    | *60.00* | *60.00* | *60.00* | *60.00* |
| CROWN-IBP          | 0.01    | 0.01    | 0.01    | 0.01    | 0.02    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.02    | 0.01    |
| DeepPoly           | 0.02    | 0.02    | 0.29    | 0.15    | *60.00* | 5.14    | 1.75    | 1.28    | 40.27   | 27.57   | *60.00* | *60.00* | *60.00* | *60.00* |
| RefineZono         | 12.08   | 18.52   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| WK                 | 0.01    | 0.01    | 0.01    | 0.01    | 0.04    | 0.04    | 0.01    | 0.01    | 0.07    | 0.06    | 0.38    | 0.19    | *60.00* | *60.00* |
| K-ReLU             | 12.37   | 12.78   | 45.15   | 44.37   | *60.00* | *60.00* | *60.00* | 48.59   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| SDPVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| LMIVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| Op-Norm            | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | 0.01    | *60.00* | *60.00* |
| FastLip            | 0.01    | 0.03    | 0.02    | 0.07    | 1.87    | 1.91    | 2.04    | 3.11    | 8.27    | 50.14   | 46.81   | *60.00* | *60.00* | *60.00* |
| RecurJac           | 0.14    | 0.31    | 0.36    | 1.28    | 10.81   | 29.23   | 12.47   | 38.94   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| *PGD Attack*       | 0.26    | 0.32    | 0.30    | 0.34    | 0.43    | 0.49    | 0.33    | 0.38    | 0.45    | 0.48    | 0.47    | 0.51    | 0.52    | 0.54    |
| *Normal Inference* | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    |


        "
      | markdownify | liquify
}}
      </div>
      <div class="tab-pane fade" id="nav-main-mnist-03-rad" role="tabpanel" aria-labelledby="nav-main-mnist-03-rad-tab">
        {{"

**Bolded** numbers mark the highest certified radii among verification approaches.
*0.000* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.

|                   | FCNNa     | FCNNa     | FCNNb     | FCNNb     | FCNNc     | FCNNc     | CNNa      | CNNa      | CNNb      | CNNb      | CNNc      | CNNc      | CNNd    | CNNd      |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------- | --------- |
|                   | adv3      | cadv3     | adv3      | cadv3     | adv3      | cadv3     | adv3      | cadv3     | adv3      | cadv3     | adv3      | cadv3     | adv3    | cadv3     |
| Bounded MILP      | **0.157** | **0.256** | *0.000*   | **0.278** | *0.000*   | 0.230     | *0.000*   | 0.255     | *0.000*   | 0.233     | *0.000*   | 0.018     | *0.000* | *0.000*   |
| AI2               | **0.157** | **0.256** | *0.000*   | **0.278** | *0.000*   | 0.232     | *0.000*   | 0.240     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| LP-Full           | 0.079     | 0.229     | 0.049     | 0.217     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| IBP               | 0.008     | **0.256** | *0.000*   | 0.275     | *0.000*   | **0.295** | 0.018     | **0.294** | *0.000*   | **0.296** | *0.000*   | **0.292** | *0.000* | **0.299** |
| Fast-Lin          | 0.074     | 0.202     | 0.043     | 0.151     | 0.008     | 0.033     | 0.117     | 0.221     | 0.106     | 0.174     | 0.083     | 0.143     | *0.000* | *0.000*   |
| CROWN             | 0.119     | 0.239     | **0.076** | 0.224     | **0.021** | 0.228     | **0.174** | 0.254     | **0.141** | 0.204     | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| CNN-Cert          | 0.078     | 0.210     | 0.048     | 0.189     | *0.000*   | *0.000*   | 0.130     | 0.253     | 0.120     | 0.179     | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| CROWN-IBP         | 0.069     | 0.239     | 0.005     | 0.256     | *0.000*   | 0.288     | 0.074     | 0.283     | *0.000*   | 0.272     | *0.000*   | 0.277     | *0.000* | 0.272     |
| DeepPoly          | 0.119     | 0.239     | **0.076** | 0.224     | *0.000*   | 0.228     | **0.174** | 0.254     | 0.105     | 0.188     | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| RefineZono        | 0.120     | 0.237     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| WK                | 0.074     | 0.202     | 0.043     | 0.151     | 0.008     | 0.033     | 0.117     | 0.221     | 0.106     | 0.174     | **0.085** | 0.143     | *0.000* | *0.000*   |
| K-ReLU            | 0.125     | **0.256** | 0.002     | 0.209     | *0.000*   | *0.000*   | *0.000*   | 0.227     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| SDPVerify         | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| LMIVerify         | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| Op-Norm           | 0.001     | *0.000*   | *0.000*   | 0.001     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| FastLip           | 0.053     | 0.189     | 0.015     | 0.153     | *0.000*   | 0.043     | 0.067     | 0.199     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| RecurJac          | 0.072     | 0.193     | 0.039     | 0.157     | *0.000*   | *0.000*   | 0.058     | 0.146     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   |
| *PGD Upper Bound* | *0.199*   | *0.309*   | *0.216*   | *0.331*   | *0.231*   | *0.427*   | *0.308*   | *0.302*   | *0.313*   | *0.318*   | *0.328*   | *0.316*   | *0.366* | *0.325*   |


**Average running time for robustness radius computation** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``120 s'' per instance.)


|              | FCNNa    | FCNNa    | FCNNb    | FCNNb    | FCNNc    | FCNNc    | CNNa     | CNNa     | CNNb     | CNNb     | CNNc     | CNNc     | CNNd     | CNNd     |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
|              | adv3     | cadv3    | adv3     | cadv3    | adv3     | cadv3    | adv3     | cadv3    | adv3     | cadv3    | adv3     | cadv3    | adv3     | cadv3    |
| Bounded MILP | 4.75     | 2.53     | *120.00* | 7.77     | *120.00* | *120.00* | *120.00* | 116.93   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| AI2          | 4.96     | 3.74     | *120.00* | 12.73    | *120.00* | 117.14   | *120.00* | 118.62   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LP-Full      | 2.51     | 2.39     | 104.91   | 63.49    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| IBP          | 0.02     | 0.02     | 0.02     | 0.03     | 0.03     | 0.04     | 0.02     | 0.03     | 0.03     | 0.04     | 0.04     | 0.05     | 0.05     | 0.06     |
| Fast-Lin     | 0.03     | 0.04     | 0.17     | 0.15     | 11.21    | 8.92     | 1.78     | 2.40     | 29.16    | 28.74    | 115.24   | 106.29   | *120.00* | *120.00* |
| CROWN        | 0.03     | 0.03     | 0.05     | 0.05     | 0.17     | 0.18     | 0.09     | 0.09     | 0.80     | 0.80     | *120.00* | *120.00* | *120.00* | *120.00* |
| CNN-Cert     | 0.16     | 0.16     | 1.41     | 1.95     | *120.00* | *120.00* | 4.71     | 5.14     | 112.83   | 117.19   | *120.00* | *120.00* | *120.00* | *120.00* |
| CROWN-IBP    | 0.03     | 0.03     | 0.03     | 0.03     | 0.07     | 0.08     | 0.04     | 0.04     | 0.10     | 0.07     | 0.08     | 0.08     | 0.08     | 0.08     |
| DeepPoly     | 0.16     | 0.13     | 1.76     | 0.94     | *120.00* | 32.34    | 10.85    | 8.12     | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| RefineZono   | 79.13    | 21.92    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| WK           | 0.04     | 0.04     | 0.06     | 0.06     | 0.21     | 0.18     | 0.07     | 0.08     | 0.18     | 0.21     | 0.57     | 0.78     | *120.00* | *120.00* |
| K-ReLU       | 64.68    | 63.76    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| SDPVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LMIVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| Op-Norm      | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.01     | 0.00     | 0.01     | 0.01     | *120.00* | *120.00* |
| FastLip      | 0.16     | 0.19     | 0.62     | 0.74     | 9.15     | 38.31    | 31.29    | 25.23    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| RecurJac     | 5.37     | 5.92     | 29.40    | 30.22    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| *PGD Attack* | 1.81     | 2.38     | 2.00     | 2.23     | 2.79     | 3.08     | 2.10     | 2.24     | 2.71     | 2.82     | 2.92     | 3.10     | 3.25     | 3.32     |



        "
      | markdownify | liquify
}}
      </div>
    </div>
  </div>

  <div class="tab-pane fade" id="main-cifar-05" role="tabpanel" aria-labelledby="main-cifar-05-tab">
    <nav>
      <div class="nav nav-tabs" id="nav-main-cifar-05" role="tablist">
        <button class="nav-link active" id="nav-main-cifar-05-acc-tab" data-bs-toggle="tab" data-bs-target="#nav-main-cifar-05-acc" type="button" role="tab" aria-controls="nav-home" aria-selected="true">Certified Accuracy under \(\epsilon=0.5/255\)</button>
        <button class="nav-link" id="nav-main-cifar-05-rad-tab" data-bs-toggle="tab" data-bs-target="#nav-main-cifar-05-rad" type="button" role="tab" aria-controls="nav-profile" aria-selected="false">Average Certified Robustness Radius</button>
      </div>
    </nav>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane fade show active" id="nav-main-cifar-05-acc" role="tabpanel" aria-labelledby="nav-main-cifar-05-acc-tab">
        <!-- CIFAR-10, \(\epsilon=0.5/255\), certified accuracy -->
        {{ "
        
**Bolded** numbers mark the highest robust accuracies among verification approaches.
*0%* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.
        

|                   | FCNNa   | FCNNb   | FCNNc   | CNNa    | CNNb    | CNNc    | CNNd   |
| ----------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------ |
| Bounded MILP      | **40%** | **43%** | *0%*    | 23%     | *0%*    | *0%*    | *0%*   |
| AI2               | **40%** | **43%** | *0%*    | **51%** | 9%      | *0%*    | *0%*   |
| LP-Full           | **40%** | **43%** | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   |
| IBP               | 29%     | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   |
| Fast-Lin          | **40%** | 42%     | 49%     | **51%** | 49%     | **45%** | *0%*   |
| CROWN             | **40%** | **43%** | **51%** | **51%** | **52%** | *0%*    | *0%*   |
| CNN-Cert          | **40%** | **43%** | *0%*    | **51%** | **52%** | *0%*    | *0%*   |
| CROWN-IBP         | **40%** | 24%     | *0%*    | 13%     | *0%*    | *0%*    | *0%*   |
| DeepPoly          | **40%** | **43%** | *0%*    | **51%** | **52%** | *0%*    | *0%*   |
| RefineZono        | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   |
| WK                | **40%** | 42%     | 49%     | **51%** | 49%     | **45%** | **3%** |
| K-ReLU            | **40%** | **43%** | *0%*    | 48%     | *0%*    | *0%*    | *0%*   |
| SDPVerify         | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   |
| LMIVerify         | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   |
| Op-Norm           | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   |
| FastLip           | 40%     | 40%     | *0%*    | 36%     | *0%*    | *0%*    | *0%*   |
| RecurJac          | **40%** | 42%     | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   |
| *PGD Upper Bound* | *40%*   | *43%*   | *53%*   | *51%*   | *61%*   | *49%*   | *76%*  |
| *Clean Acc.*      | *41%*   | *47%*   | *54%*   | *57%*   | *68%*   | *54%*   | *85%*  |



**Average running time for single-instance robustness verification** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``60 s'' per instance.)


|                    | FCNNa   | FCNNb   | FCNNc   | CNNa    | CNNb    | CNNc    | CNNd    |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Bounded MILP       | 0.93    | 10.70   | *60.00* | 50.60   | *60.00* | *60.00* | *60.00* |
| AI2                | 0.07    | 2.09    | *60.00* | 7.69    | 57.57   | *60.00* | *60.00* |
| LP-Full            | 0.71    | 33.68   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| IBP                | 0.00    | 0.00    | 0.01    | 0.00    | 0.00    | 0.01    | 0.01    |
| Fast-Lin           | 0.03    | 0.02    | 1.03    | 1.49    | 11.07   | 32.62   | *60.00* |
| CROWN              | 0.00    | 0.01    | 0.08    | 0.03    | 0.32    | *60.00* | *60.00* |
| CNN-Cert           | 0.19    | 0.71    | *60.00* | 1.43    | 35.36   | 59.69   | *60.00* |
| CROWN-IBP          | 0.00    | 0.00    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    |
| DeepPoly           | 0.05    | 0.53    | *60.00* | 2.55    | 41.52   | *60.00* | *60.00* |
| RefineZono         | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| WK                 | 0.00    | 0.01    | 0.05    | 0.01    | 0.06    | 0.10    | 54.42   |
| K-ReLU             | 5.83    | 20.91   | *60.00* | 31.63   | *60.00* | *60.00* | *60.00* |
| SDPVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| LMIVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| Op-Norm            | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | 0.03    | *60.00* |
| FastLip            | 0.09    | 0.43    | 1.35    | 21.28   | 49.81   | *60.00* | *60.00* |
| RecurJac           | 1.56    | 5.67    | *60.00* | 57.99   | *60.00* | *60.00* | *60.00* |
| *PGD Attack*       | 0.23    | 0.27    | 0.35    | 0.30    | 0.35    | 0.39    | 0.54    |
| *Normal Inference* | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    |

        "
      | markdownify | liquify
}}
      </div>
      <div class="tab-pane fade" id="nav-main-cifar-05-rad" role="tabpanel" aria-labelledby="nav-main-cifar-05-rad-tab">
        <!-- CIFAR-10, \(\epsilon=0.5/255\), average certified robustness radius -->
        
        {{ "
Numbers are multiplied by \\(255\\) for simplicity.
**Bolded** numbers mark the highest certified radii among verification approaches.
*0.000* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.
        
|      | FCNNa | FCNNb | FCNNc | CNNa | CNNb | CNNc | CNNd |
| ---- | ------------------ | ------------------ | ------------------ | ----------------- | ----------------- | ----------------- | ------------------ |
| Bounded MILP | **5.053** | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| AI2 | 5.047 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| LP-Full | 3.966 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| IBP | 1.026 | 0.011 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| Fast-Lin | 3.723 | 2.596 | 0.812 | 1.503 | 0.637 | *0.000* | *0.000* |
| CROWN | 3.984 | **2.888** | **1.065** | **1.669** | **0.765** | *0.000* | *0.000* |
| CNN-Cert | 3.893 | 2.819 | *0.000* | 1.625 | 0.699 | *0.000* | *0.000* |
| CROWN-IBP | 2.187 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| DeepPoly | 3.984 | **2.888** | *0.000* | **1.669** | *0.000* | *0.000* | *0.000* |
| RefineZono | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| WK | 2.818 | 1.738 | *0.000* | 0.734 | *0.000* | *0.000* | *0.000* |
| K-ReLU | 4.203 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| SDPVerify | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| LMIVerify | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| Op-Norm | 0.183 | 0.071 | *0.000* | 0.002 | 0.001 | *0.000* | *0.000* |
| FastLip | 2.988 | 1.256 | *0.000* | 0.599 | *0.000* | *0.000* | *0.000* |
| RecurJac | 3.596 | 2.400 | *0.000* | *0.000* | *0.000* | *0.000* | *0.000* |
| *PGD Upper Bound* | *5.679* | *5.521* | *6.037* | *3.482* | *2.560* | *4.187* | *1.819* |





**Average running time for robustness radius computation** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``120 s'' per instance.)

|      | FCNNa | FCNNb | FCNNc | CNNa | CNNb | CNNc | CNNd |
| ---- | ------------------ | ------------------ | ------------------ | ----------------- | ----------------- | ----------------- | ------------------ |
| Bounded MILP | 23.76 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| AI2 | 40.75 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LP-Full | 8.96 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| IBP | 0.02 | 0.02 | 0.04 | 0.02 | 0.04 | 0.05 | 0.08 |
| Fast-Lin | 0.05 | 0.22 | 12.19 | 13.14 | 112.11 | *120.00* | *120.00* |
| CROWN | 0.03 | 0.05 | 0.43 | 0.20 | 2.96 | *120.00* | *120.00* |
| CNN-Cert | 0.24 | 3.05 | *120.00* | 5.03 | 102.79 | *120.00* | *120.00* |
| CROWN-IBP | 0.02 | 0.02 | 0.04 | 0.03 | 0.04 | 0.05 | 0.06 |
| DeepPoly | 0.40 | 5.26 | *120.00* | 24.44 | *120.00* | *120.00* | *120.00* |
| RefineZono | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| WK | 0.03 | 0.05 | 0.25 | 0.11 | 0.70 | *120.00* | *120.00* |
| K-ReLU | 100.18 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| SDPVerify | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LMIVerify | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| Op-Norm | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.03 | *120.00* |
| FastLip | 0.41 | 1.98 | 13.48 | 77.41 | *120.00* | *120.00* | *120.00* |
| RecurJac | 5.72 | 26.99 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| *PGD Attack* | 2.54 | 2.87 | 3.68 | 2.89 | 3.48 | 3.69 | 4.81 |



        "
      | markdownify | liquify
}}
      </div>
    </div>
  </div>

  <div class="tab-pane fade" id="main-cifar-2" role="tabpanel" aria-labelledby="main-cifar-2-tab">
    <nav>
      <div class="nav nav-tabs" id="nav-main-cifar-2" role="tablist">
        <button class="nav-link active" id="nav-main-cifar-2-acc-tab" data-bs-toggle="tab" data-bs-target="#nav-main-cifar-2-acc" type="button" role="tab" aria-controls="nav-home" aria-selected="true">Certified Accuracy under \(\epsilon=2/255\)</button>
        <button class="nav-link" id="nav-main-cifar-2-rad-tab" data-bs-toggle="tab" data-bs-target="#nav-main-cifar-2-rad" type="button" role="tab" aria-controls="nav-profile" aria-selected="false">Average Certified Robustness Radius</button>
      </div>
    </nav>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane fade show active" id="nav-main-cifar-2-acc" role="tabpanel" aria-labelledby="nav-main-cifar-2-acc-tab">
        <!-- CIFAR-10, \(\epsilon=2/255\), certified accuracy -->
         {{ "
        
**Bolded** numbers mark the highest robust accuracies among verification approaches.
*0%* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.
        
|                   | FCNNa   | FCNNa   | FCNNb   | FCNNb   | FCNNc  | FCNNc   | CNNa    | CNNa    | CNNb    | CNNb    | CNNc    | CNNc    | CNNd  | CNNd    |
| ----------------- | ------- | ------- | ------- | ------- | ------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ----- | ------- |
|                   | adv2    | cadv2   | adv2    | cadv2   | adv2   | cadv2   | adv2    | cadv2   | adv2    | cadv2   | adv2    | cadv2   | adv2  | cadv2   |
| Bounded MILP      | **35%** | **33%** | 30%     | **37%** | *0%*   | *0%*    | *0%*    | **47%** | *0%*    | 49%     | *0%*    | *0%*    | *0%*  | *0%*    |
| AI2               | **35%** | **33%** | **39%** | **37%** | *0%*   | 20%     | **34%** | **47%** | **19%** | 45%     | *0%*    | *0%*    | *0%*  | *0%*    |
| LP-Full           | **35%** | **33%** | 36%     | **37%** | *0%*   | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| IBP               | 6%      | **33%** | *0%*    | 35%     | *0%*   | **32%** | *0%*    | 45%     | *0%*    | **49%** | *0%*    | **51%** | *0%*  | **51%** |
| Fast-Lin          | **35%** | **33%** | 35%     | 34%     | *0%*   | 30%     | 33%     | 45%     | 14%     | 43%     | **13%** | 46%     | *0%*  | *0%*    |
| CROWN             | **35%** | **33%** | 36%     | 35%     | **1%** | **32%** | **34%** | 46%     | **19%** | 45%     | *0%*    | *0%*    | *0%*  | *0%*    |
| CNN-Cert          | **35%** | **33%** | 35%     | 35%     | *0%*   | *0%*    | **34%** | 46%     | **19%** | 45%     | *0%*    | *0%*    | *0%*  | *0%*    |
| CROWN-IBP         | 30%     | **33%** | *0%*    | 34%     | *0%*   | **32%** | *0%*    | 46%     | *0%*    | 46%     | *0%*    | 50%     | *0%*  | 47%     |
| DeepPoly          | **35%** | **33%** | 36%     | 35%     | *0%*   | 17%     | **34%** | 46%     | **19%** | 45%     | *0%*    | *0%*    | *0%*  | *0%*    |
| RefineZono        | *0%*    | **33%** | *0%*    | *0%*    | *0%*   | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| WK                | **35%** | **33%** | 35%     | 34%     | *0%*   | 30%     | 33%     | 45%     | 14%     | 43%     | **13%** | 46%     | *0%*  | 23%     |
| K-ReLU            | **35%** | **33%** | 35%     | **37%** | *0%*   | *0%*    | *0%*    | 44%     | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| SDPVerify         | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| LMIVerify         | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| Op-Norm           | *0%*    | *0%*    | *0%*    | *0%*    | *0%*   | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| FastLip           | 33%     | 32%     | 15%     | 32%     | *0%*   | 25%     | 3%      | 40%     | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| RecurJac          | 34%     | 32%     | 33%     | 32%     | *0%*   | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*    | *0%*  | *0%*    |
| *PGD Upper Bound* | *36%*   | *35%*   | *41%*   | *39%*   | *26%*  | *38%*   | *43%*   | *49%*   | *52%*   | *50%*   | *49%*   | *53%*   | *62%* | *54%*   |
| *Clean Acc.*      | *44%*   | *40%*   | *48%*   | *43%*   | *35%*  | *46%*   | *54%*   | *55%*   | *66%*   | *60%*   | *59%*   | *63%*   | *81%* | *62%*   |




**Average running time for single-instance robustness verification** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``60 s'' per instance.)

|                    | FCNNa   | FCNNa   | FCNNb   | FCNNb   | FCNNc   | FCNNc   | CNNa    | CNNa    | CNNb    | CNNb    | CNNc    | CNNc    | CNNd    | CNNd    |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
|                    | adv2    | cadv2   | adv2    | cadv2   | adv2    | cadv2   | adv2    | cadv2   | adv2    | cadv2   | adv2    | cadv2   | adv2    | cadv2   |
| Bounded MILP       | 1.24    | 1.05    | 32.64   | 4.87    | *60.00* | *60.00* | *60.00* | 4.70    | *60.00* | 30.27   | *60.00* | 59.55   | *60.00* | *60.00* |
| AI2                | 1.99    | 1.06    | 5.16    | 7.01    | *60.00* | 45.69   | 23.65   | 4.81    | 54.19   | 27.01   | *60.00* | *60.00* | *60.00* | *60.00* |
| LP-Full            | 0.74    | 0.77    | 34.64   | 26.08   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| IBP                | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.01    | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    |
| Fast-Lin           | 0.01    | 0.03    | 0.02    | 0.05    | 1.36    | 1.36    | 1.48    | 1.45    | 11.77   | 10.92   | 33.87   | 34.73   | *60.00* | *60.00* |
| CROWN              | 0.00    | 0.00    | 0.01    | 0.01    | 0.07    | 0.04    | 0.03    | 0.03    | 0.33    | 0.47    | *60.00* | *60.00* | *60.00* | *60.00* |
| CNN-Cert           | 0.05    | 0.17    | 0.67    | 0.70    | 10.71   | 55.88   | 1.18    | 1.26    | 18.84   | 33.65   | 42.77   | 58.12   | *60.00* | *60.00* |
| CROWN-IBP          | 0.00    | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    | 0.01    |
| DeepPoly           | 0.05    | 0.04    | 0.56    | 0.45    | *60.00* | 41.60   | 2.49    | 1.10    | 43.68   | 17.67   | *60.00* | *60.00* | *60.00* | *60.00* |
| RefineZono         | *60.00* | 26.05   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| WK                 | 0.01    | 0.01    | 0.01    | 0.01    | 0.06    | 0.03    | 0.02    | 0.01    | 0.08    | 0.10    | 0.15    | 0.09    | *60.00* | 0.40    |
| K-ReLU             | 7.08    | 7.33    | 34.73   | 25.81   | *60.00* | *60.00* | *60.00* | 26.78   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| SDPVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| LMIVerify          | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| Op-Norm            | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.01    | 0.01    | 0.00    | 0.01    | 0.01    | 0.01    | 0.03    | *60.00* | *60.00* |
| FastLip            | 0.06    | 0.09    | 0.30    | 0.30    | 1.30    | 11.06   | 11.90   | 11.24   | 37.44   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| RecurJac           | 0.90    | 1.43    | 4.83    | 5.14    | *60.00* | *60.00* | 59.51   | 59.17   | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| *PGD Attack*       | 0.24    | 0.27    | 0.28    | 0.30    | 0.37    | 0.40    | 0.30    | 0.37    | 0.36    | 0.45    | 0.37    | 0.46    | 0.55    | 0.64    |
| *Normal Inference* | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    | 0.00    |



        "
      | markdownify | liquify
}}
      </div>
      <div class="tab-pane fade" id="nav-main-cifar-2-rad" role="tabpanel" aria-labelledby="nav-main-cifar-2-rad-tab">
        <!-- CIFAR-10, \(\epsilon=2/255\), average certified robustness radius -->
        
        {{ "
Numbers are multiplied by \\(255\\) for simplicity.
**Bolded** numbers mark the highest certified radii among verification approaches.
*0.000* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.
        
|                   | FCNNa     | FCNNa     | FCNNb     | FCNNb     | FCNNc     | FCNNc     | CNNa      | CNNa      | CNNb      | CNNb      | CNNc    | CNNc      | CNNd    | CNNd      |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------- | --------- | ------- | --------- |
|                   | adv2      | cadv2     | adv2      | cadv2     | adv2      | cadv2     | adv2      | cadv2     | adv2      | cadv2     | adv2    | cadv2     | adv2    | cadv2     |
| Bounded MILP      | **6.429** | **9.301** | *0.000*   | 5.229     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| AI2               | 6.033     | **9.301** | *0.000*   | 0.185     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| LP-Full           | 4.930     | 8.747     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| IBP               | 0.826     | 8.324     | 0.016     | **5.027** | *0.000*   | 3.508     | 0.028     | **5.958** | *0.000*   | **4.611** | *0.000* | **4.411** | *0.000* | **4.233** |
| Fast-Lin          | 4.714     | 7.103     | 3.211     | 4.036     | 0.655     | 3.080     | 2.269     | 4.079     | 1.294     | 2.486     | *0.000* | *0.000*   | *0.000* | *0.000*   |
| CROWN             | 5.009     | 7.408     | **3.683** | 4.720     | **0.847** | **4.071** | **2.601** | 4.682     | **1.551** | 3.030     | *0.000* | *0.000*   | *0.000* | *0.000*   |
| CNN-Cert          | 4.856     | 7.321     | 3.575     | 4.911     | *0.000*   | *0.000*   | 2.532     | 4.578     | 1.339     | 2.702     | *0.000* | *0.000*   | *0.000* | *0.000*   |
| CROWN-IBP         | 2.717     | 7.863     | *0.000*   | 5.004     | *0.000*   | 3.936     | *0.000*   | 5.415     | *0.000*   | 3.889     | *0.000* | 3.909     | *0.000* | 3.551     |
| DeepPoly          | 5.009     | 7.408     | **3.683** | 4.720     | *0.000*   | 0.173     | **2.601** | 4.682     | *0.000*   | 0.498     | *0.000* | *0.000*   | *0.000* | *0.000*   |
| RefineZono        | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| WK                | 3.939     | 7.103     | 2.407     | 4.036     | *0.000*   | 3.080     | 1.476     | 4.079     | 0.423     | 2.486     | *0.000* | *0.000*   | *0.000* | *0.000*   |
| K-ReLU            | 5.184     | 8.871     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| SDPVerify         | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| LMIVerify         | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| Op-Norm           | 0.174     | 0.011     | 0.068     | *0.000*   | *0.000*   | *0.000*   | 0.002     | *0.000*   | 0.001     | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| FastLip           | 3.696     | 6.319     | 1.359     | 3.399     | *0.000*   | 2.404     | 0.742     | 3.523     | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| RecurJac          | 4.550     | 6.375     | 2.765     | 3.509     | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000*   | *0.000* | *0.000*   | *0.000* | *0.000*   |
| *PGD Upper Bound* | *7.856*   | *13.485*  | *8.581*   | *10.864*  | *9.292*   | *9.181*   | *7.185*   | *11.432*  | *6.629*   | *10.301*  | *7.901* | *9.578*   | *5.472* | *9.981*   |





**Average running time for robustness radius computation** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``120 s'' per instance.)


|              | FCNNa    | FCNNa    | FCNNb    | FCNNb    | FCNNc    | FCNNc    | CNNa     | CNNa     | CNNb     | CNNb     | CNNc     | CNNc     | CNNd     | CNNd     |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
|              | adv2     | cadv2    | adv2     | cadv2    | adv2     | cadv2    | adv2     | cadv2    | adv2     | cadv2    | adv2     | cadv2    | adv2     | cadv2    |
| Bounded MILP | 26.48    | 16.01    | *120.00* | 117.08   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| AI2          | 59.12    | 48.56    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LP-Full      | 8.89     | 8.65     | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| IBP          | 0.02     | 0.02     | 0.02     | 0.03     | 0.04     | 0.05     | 0.03     | 0.04     | 0.04     | 0.05     | 0.04     | 0.05     | 0.08     | 0.10     |
| Fast-Lin     | 0.05     | 0.08     | 0.22     | 0.27     | 12.10    | 14.38    | 13.27    | 13.08    | 108.53   | 96.77    | *120.00* | *120.00* | *120.00* | *120.00* |
| CROWN        | 0.03     | 0.03     | 0.05     | 0.05     | 0.43     | 0.41     | 0.20     | 0.20     | 3.00     | 4.26     | *120.00* | *120.00* | *120.00* | *120.00* |
| CNN-Cert     | 0.25     | 0.34     | 3.17     | 2.66     | *120.00* | *120.00* | 5.60     | 5.75     | 113.73   | 116.81   | *120.00* | *120.00* | *120.00* | *120.00* |
| CROWN-IBP    | 0.02     | 0.03     | 0.02     | 0.03     | 0.05     | 0.06     | 0.03     | 0.04     | 0.04     | 0.10     | 0.05     | 0.07     | 0.06     | 0.08     |
| DeepPoly     | 0.41     | 0.36     | 5.07     | 3.91     | *120.00* | *120.00* | 22.75    | 11.02    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| RefineZono   | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| WK           | 0.03     | 0.04     | 0.05     | 0.07     | 0.25     | 0.28     | 0.11     | 0.13     | 0.79     | 1.35     | *120.00* | *120.00* | *120.00* | *120.00* |
| K-ReLU       | 91.47    | 93.79    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| SDPVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LMIVerify    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| Op-Norm      | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     | 0.01     | 0.01     | 0.03     | 0.03     | *120.00* | *120.00* |
| FastLip      | 0.42     | 0.40     | 1.88     | 1.74     | 16.78    | 65.67    | 99.47    | 65.69    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| RecurJac     | 5.83     | 6.41     | 27.72    | 28.83    | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| *PGD Attack* | 2.79     | 3.64     | 3.04     | 3.60     | 4.15     | 4.26     | 3.19     | 4.13     | 3.79     | 4.57     | 4.01     | 4.73     | 5.05     | 6.63     |



        "
      | markdownify | liquify
}}
      </div>
    </div>
  </div>

  <div class="tab-pane fade" id="main-cifar-8" role="tabpanel" aria-labelledby="main-cifar-8-tab">
    <nav>
      <div class="nav nav-tabs" id="nav-main-cifar-8" role="tablist">
        <button class="nav-link active" id="nav-main-cifar-8-acc-tab" data-bs-toggle="tab" data-bs-target="#nav-main-cifar-8-acc" type="button" role="tab" aria-controls="nav-home" aria-selected="true">Certified Accuracy under \(\epsilon=8/255\)</button>
        <button class="nav-link" id="nav-main-cifar-8-rad-tab" data-bs-toggle="tab" data-bs-target="#nav-main-cifar-8-rad" type="button" role="tab" aria-controls="nav-profile" aria-selected="false">Average Certified Robustness Radius</button>
      </div>
    </nav>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane fade show active" id="nav-main-cifar-8-acc" role="tabpanel" aria-labelledby="nav-main-cifar-8-acc-tab">
        <!-- CIFAR-10, \(\epsilon=8/255\), certified accuracy -->
        
         {{ "
        
**Bolded** numbers mark the highest robust accuracies among verification approaches.
*0%* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.
        
|                   | FCNNa   | FCNNa   | FCNNb  | FCNNb   | FCNNc | FCNNc   | CNNa   | CNNa    | CNNb   | CNNb    | CNNc  | CNNc    | CNNd  | CNNd    |
| ----------------- | ------- | ------- | ------ | ------- | ----- | ------- | ------ | ------- | ------ | ------- | ----- | ------- | ----- | ------- |
|                   | adv8    | cadv8   | adv8   | cadv8   | adv8  | cadv8   | adv8   | cadv8   | adv8   | cadv8   | adv8  | cadv8   | adv8  | cadv8   |
| Bounded MILP      | **19%** | **27%** | 1%     | **25%** | *0%*  | *0%*    | *0%*   | **34%** | *0%*   | **36%** | *0%*  | *0%*    | *0%*  | *0%*    |
| AI2               | **19%** | **27%** | **7%** | 23%     | *0%*  | 22%     | **8%** | **34%** | *0%*   | 20%     | *0%*  | 14%     | *0%*  | *0%*    |
| LP-Full           | 15%     | **27%** | 6%     | **25%** | *0%*  | *0%*    | *0%*   | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| IBP               | *0%*    | **27%** | *0%*   | **25%** | *0%*  | **30%** | *0%*   | **34%** | *0%*   | 35%     | *0%*  | **38%** | *0%*  | **28%** |
| Fast-Lin          | 15%     | 25%     | 4%     | 18%     | *0%*  | 19%     | 3%     | 26%     | *0%*   | 15%     | *0%*  | 7%      | *0%*  | *0%*    |
| CROWN             | 15%     | **27%** | 6%     | 20%     | *0%*  | 22%     | **8%** | 33%     | **1%** | 20%     | *0%*  | *0%*    | *0%*  | *0%*    |
| CNN-Cert          | 15%     | **27%** | 5%     | 20%     | *0%*  | *0%*    | 7%     | 33%     | *0%*   | 20%     | *0%*  | *0%*    | *0%*  | *0%*    |
| CROWN-IBP         | 9%      | **27%** | *0%*   | 22%     | *0%*  | 28%     | *0%*   | **34%** | *0%*   | 31%     | *0%*  | 32%     | *0%*  | 25%     |
| DeepPoly          | 15%     | **27%** | 6%     | 20%     | *0%*  | 22%     | **8%** | 33%     | **1%** | 20%     | *0%*  | 7%      | *0%*  | *0%*    |
| RefineZono        | *0%*    | **27%** | *0%*   | *0%*    | *0%*  | *0%*    | *0%*   | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| WK                | 15%     | 25%     | 4%     | 18%     | *0%*  | 19%     | 3%     | 26%     | *0%*   | 15%     | *0%*  | 7%      | *0%*  | 5%      |
| K-ReLU            | 15%     | **27%** | 2%     | 23%     | *0%*  | *0%*    | *0%*   | 32%     | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| SDPVerify         | *0%*    | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*   | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| LMIVerify         | *0%*    | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*   | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| Op-Norm           | *0%*    | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*   | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| FastLip           | 12%     | **27%** | *0%*   | 17%     | *0%*  | 17%     | *0%*   | 24%     | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| RecurJac          | 14%     | **27%** | 2%     | 17%     | *0%*  | *0%*    | *0%*   | *0%*    | *0%*   | *0%*    | *0%*  | *0%*    | *0%*  | *0%*    |
| *PGD Upper Bound* | *22%*   | *28%*   | *23%*  | *26%*   | *19%* | *34%*   | *34%*  | *34%*   | *33%*  | *39%*   | *36%* | *40%*   | *41%* | *31%*   |
| *Clean Acc.*      | *33%*   | *31%*   | *37%*  | *30%*   | *26%* | *39%*   | *44%*  | *46%*   | *53%*  | *48%*   | *52%* | *46%*   | *66%* | *46%*   |




**Average running time for single-instance robustness verification** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``60 s'' per instance.)

|                    | FCNNa   | FCNNa   | FCNNb   | FCNNb   | FCNNc   | FCNNc   | CNNa    | CNNa    | CNNb    | CNNb    | CNNc    | CNNc    | CNNd    | CNNd    |
| ---- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- |
|      | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 |
| Bounded MILP | 3.34 | 0.95 | 59.44 | 3.85 | *60.00* | 58.94 | *60.00* | 4.55 | *60.00* | 27.30 | *60.00* | *60.00* | *60.00* | *60.00* |
| AI2 | 3.68 | 1.72 | 47.08 | 10.74 | *60.00* | 36.69 | 49.39 | 4.29 | *60.00* | 37.84 | *60.00* | 57.95 | *60.00* | *60.00* |
| LP-Full | 1.02 | 0.82 | 51.35 | 20.84 | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| IBP | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Fast-Lin | 0.01 | 0.01 | 0.03 | 0.02 | 1.44 | 1.58 | 1.49 | 1.24 | 11.77 | 10.14 | 38.27 | 34.86 | *60.00* | *60.00* |
| CROWN | 0.00 | 0.00 | 0.01 | 0.01 | 0.07 | 0.04 | 0.03 | 0.03 | 0.33 | 0.46 | *60.00* | *60.00* | *60.00* | *60.00* |
| CNN-Cert | 0.04 | 0.05 | 0.25 | 0.50 | 10.94 | 42.49 | 0.48 | 1.03 | 4.11 | 21.88 | 14.79 | 41.58 | *60.00* | *60.00* |
| CROWN-IBP | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| DeepPoly | 0.05 | 0.04 | 0.57 | 0.29 | *60.00* | 25.26 | 1.95 | 0.97 | 38.23 | 15.16 | *60.00* | 57.25 | *60.00* | *60.00* |
| RefineZono | *60.00* | 7.77 | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| WK | 0.01 | 0.00 | 0.01 | 0.01 | 0.06 | 0.03 | 0.02 | 0.01 | 0.09 | 0.11 | 0.40 | 0.10 | *60.00* | 5.71 |
| K-ReLU | 10.14 | 7.66 | 57.61 | 34.26 | *60.00* | *60.00* | *60.00* | 32.65 | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| SDPVerify | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| LMIVerify | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| Op-Norm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.03 | *60.00* | *60.00* |
| FastLip | 0.05 | 0.06 | 0.12 | 0.19 | 1.44 | 9.25 | 8.40 | 8.36 | 28.66 | 59.01 | *60.00* | *60.00* | *60.00* | *60.00* |
| RecurJac | 0.74 | 0.90 | 2.81 | 4.04 | 48.90 | 59.14 | 57.64 | 58.93 | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* | *60.00* |
| *PGD Attack* | 0.23 | 0.27 | 0.27 | 0.32 | 0.38 | 0.39 | 0.29 | 0.37 | 0.38 | 0.43 | 0.39 | 0.47 | 0.55 | 0.65 |
| *Normal Inference* | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |



        "
      | markdownify | liquify
}}
      </div>
      <div class="tab-pane fade" id="nav-main-cifar-8-rad" role="tabpanel" aria-labelledby="nav-main-cifar-8-rad-tab">
        <!-- CIFAR-10, \(\epsilon=8/255\), average certified robustness radius -->
        
        {{ "
        
Numbers are multiplied by \\(255\\) for simplicity.
**Bolded** numbers mark the highest certified radii among verification approaches.
*0.000* cells imply that corresponding verification approaches are either too loose to certify robustness or too slow to handle large models.
        
|                   | FCNNa      | FCNNa      | FCNNb     | FCNNb      | FCNNc     | FCNNc      | CNNa      | CNNa       | CNNb      | CNNb       | CNNc     | CNNc       | CNNd     | CNNd       |
| ----------------- | ---------- | ---------- | --------- | ---------- | --------- | ---------- | --------- | ---------- | --------- | ---------- | -------- | ---------- | -------- | ---------- |
|                   | adv8       | cadv8      | adv8      | cadv8      | adv8      | cadv8      | adv8      | cadv8      | adv8      | cadv8      | adv8     | cadv8      | adv8     | cadv8      |
| Bounded MILP      | **12.134** | **22.428** | *0.000*   | **23.259** | *0.000*   | *0.000*    | *0.000*   | 18.953     | *0.000*   | 1.328      | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| AI2               | 10.972     | 22.364     | *0.000*   | 13.613     | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| LP-Full           | 8.542      | 21.593     | *0.000*   | 10.625     | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| IBP               | 1.517      | 22.276     | 0.074     | 22.271     | *0.000*   | **15.880** | 0.277     | **19.115** | *0.000*   | **18.770** | *0.000*  | **18.920** | *0.000*  | **15.553** |
| Fast-Lin          | 7.825      | 17.946     | 4.637     | 11.422     | 0.881     | 9.259      | 4.471     | 9.171      | 2.396     | 6.469      | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| CROWN             | 8.882      | 20.870     | **5.398** | 15.780     | **1.226** | 11.615     | **5.156** | 11.617     | **2.890** | 8.093      | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| CNN-Cert          | 8.444      | 20.163     | 5.169     | 14.900     | *0.000*   | 2.452      | 4.952     | 11.601     | 2.523     | 7.019      | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| CROWN-IBP         | 5.614      | 21.014     | 0.108     | 20.603     | *0.000*   | 14.852     | 0.589     | 15.012     | *0.000*   | 14.604     | *0.000*  | 13.934     | *0.000*  | 11.710     |
| DeepPoly          | 8.882      | 20.870     | **5.398** | 15.780     | *0.000*   | 6.411      | **5.156** | 11.617     | *0.000*   | 7.294      | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| RefineZono        | *0.000*    | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| WK                | 7.003      | 17.946     | 3.877     | 11.422     | *0.000*   | 9.259      | 3.577     | 9.171      | 1.541     | 6.469      | *0.000*  | 5.137      | *0.000*  | *0.000*    |
| K-ReLU            | 9.304      | 22.300     | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| SDPVerify         | *0.000*    | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| LMIVerify         | *0.000*    | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| Op-Norm           | 0.117      | 0.020      | 0.052     | *0.000*    | *0.000*   | *0.000*    | 0.002     | *0.000*    | 0.001     | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| FastLip           | 6.309      | 18.002     | 1.992     | 10.019     | *0.000*   | 8.799      | 2.173     | 8.147      | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| RecurJac          | 7.539      | 18.002     | 3.971     | 10.027     | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*   | *0.000*    | *0.000*  | *0.000*    | *0.000*  | *0.000*    |
| *PGD Upper Bound* | *15.500*   | *35.723*   | *15.783*  | *35.278*   | *19.558*  | *25.796*   | *15.219*  | *31.399*   | *13.128*  | *34.687*   | *14.913* | *37.088*   | *12.040* | *28.578*   |





**Average running time for robustness radius computation** in seconds per correctly-predicted instance (We stop the execution when time exceeds ``120 s'' per instance.)

|              | FCNNa    | FCNNa    | FCNNb    | FCNNb    | FCNNc    | FCNNc    | CNNa     | CNNa     | CNNb     | CNNb     | CNNc     | CNNc     | CNNd     | CNNd     |
| ---- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- | ------------- | -------------- |
|      | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 | adv8 | cadv8 |
| Bounded MILP | 36.26 | 19.76 | *120.00* | 89.66 | *120.00* | *120.00* | *120.00* | 106.38 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| AI2 | 89.11 | 68.36 | *120.00* | 117.86 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LP-Full | 9.56 | 8.43 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| IBP | 0.02 | 0.03 | 0.03 | 0.03 | 0.04 | 0.05 | 0.03 | 0.03 | 0.04 | 0.05 | 0.05 | 0.06 | 0.08 | 0.09 |
| Fast-Lin | 0.05 | 0.06 | 0.23 | 0.23 | 12.23 | 12.47 | 12.97 | 13.05 | 106.36 | 92.09 | *120.00* | *120.00* | *120.00* | *120.00* |
| CROWN | 0.03 | 0.03 | 0.05 | 0.05 | 0.43 | 0.41 | 0.20 | 0.20 | 3.00 | 4.15 | *120.00* | *120.00* | *120.00* | *120.00* |
| CNN-Cert | 0.28 | 0.27 | 3.56 | 3.52 | *120.00* | *120.00* | 6.38 | 6.27 | 112.92 | 118.42 | *120.00* | *120.00* | *120.00* | *120.00* |
| CROWN-IBP | 0.02 | 0.03 | 0.02 | 0.03 | 0.05 | 0.06 | 0.03 | 0.04 | 0.04 | 0.09 | 0.05 | 0.06 | 0.06 | 0.08 |
| DeepPoly | 0.47 | 0.38 | 5.10 | 2.87 | *120.00* | *120.00* | 18.33 | 9.65 | *120.00* | 119.79 | *120.00* | *120.00* | *120.00* | *120.00* |
| RefineZono | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| WK | 0.03 | 0.04 | 0.05 | 0.07 | 0.24 | 0.27 | 0.10 | 0.13 | 0.74 | 1.19 | *120.00* | 1.61 | *120.00* | *120.00* |
| K-ReLU | 96.06 | 101.99 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| SDPVerify | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| LMIVerify | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| Op-Norm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.03 | 0.03 | *120.00* | *120.00* |
| FastLip | 0.45 | 0.45 | 2.05 | 1.79 | 17.55 | 86.96 | 102.45 | 70.16 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| RecurJac | 6.91 | 7.11 | 30.71 | 34.06 | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* | *120.00* |
| *PGD Attack* | 3.41 | 4.58 | 3.80 | 5.17 | 4.95 | 5.52 | 4.04 | 5.66 | 4.48 | 6.73 | 4.58 | 6.66 | 5.67 | 8.05 |



        "
      | markdownify | liquify
}}
      </div>
    </div>
  </div>

</div>



<!-- Main END -->

      </div>
    </div>
  </div>
  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingFour">
      <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseFour" aria-expanded="true" aria-controls="panelsStayOpen-collapseFour">
        Summary of Findings
      </button>
    </h2>
    <div id="panelsStayOpen-collapseFour" class="accordion-collapse collapse show" aria-labelledby="panelsStayOpen-headingFour">
      <div class="accordion-body">
        {{"

+ On relatively small models (e.g., ``FCNNa`` and ``FCNNb``), complete verification approaches can
effectively verify robustness, thus they are the best choice.
+ On larger models (e.g., ``CNNb``, ``CNNc``, ``CNNd``), usually
linear relaxation based verification approaches perform the
best, since the complete verification approaches are too slow
and other approaches are too loose, yielding almost 0%
certified accuracy. However, linear relaxation based verification
still cannot handle large DNNs and they are still too loose
compared with the upper bound provided by PGD evaluation.
+ On robustly trained models, if the robust training approach
is CROWN-IBP which is tailored for IBP and CROWN,
these two verification approaches can certify high certified
accuracy (e.g., 25% − 28% on cadv8 for ``CNNd`` on CIFAR-10) while other
approaches fail to verify. Indeed, robust training approaches
can usually boost the certified accuracy but the models must
be verified with their corresponding verification approaches.
Similar observations can also be found in the literature. 
+ SDP approaches usually take too long and thus are less practical.
+ Generally, the tendency on MNIST is the same as that on CIFAR-10, despite their different input sizes.
+ From these additional evaluations of average certified robustness radius, we find that the average radius has better precision than robust accuracy in terms of distinguishing different approaches' performance. For example, on small models such as ``FNNa`` and ``FNNb``, if measured by robust accuracy, the
complete verification approaches and some linear relaxation based approaches have almost the same precision (compare Bounded MILP, AI2 with CROWN, CNN-Cert). However, if measured by average certified robustness radius,
we can observe that complete verification approaches can certify much larger radius since they do not use any relaxations and these models are small enough to be efficiently certified by complete veritification.
+ From the running time statistics, we observe that a main cause of the failing cases (i.e., 0 robust accuracy or certified radius) for these approaches is the excessive verification time. In particular, since the evaluation of average certified robustness
radius requires multiple times of verification for each instance, inefficient verification has relatively poorer performance when measured by averaged certified radius.

        "
      | markdownify | liquify
}}
      </div>
    </div>
  </div>
</div>


<hr>

### <a name='prob'>Probabilistic </a> Verification Approaches (for smoothed DNNs)


<div class="accordion" id="accordionPanelsStayOpenExample">
  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingOne-Prob">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseOne-Prob" aria-expanded="false" aria-controls="panelsStayOpen-collapseOne-Prob">
          Evaluated Approaches
        </button>
    </h2>
    <div id="panelsStayOpen-collapseOne-Prob" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingOne-Prob">
      <div class="accordion-body">
      {{"
When evaluating certified robustness for smoothed DNNs, we evaluate the impact from the following three aspects:

- **Verification Approaches**: we evaluate 4 robustness verification approaches.

| Verification                                                 | Evaluated Adversary Type                      |
| ------------------------------------------------------------ | --------------------------------------------- |
| Differential Privacy Based [[1](http://mathias.lecuyer.me/assets/assets/pixeldp_sp19.pdf)] | \\(\ell_1\\), \\(\ell_2\\)                    |
| Neyman-Pearson [[2](https://arxiv.org/abs/1902.02918), [3](https://arxiv.org/abs/2002.08118), [4](https://arxiv.org/abs/2002.09169), [5](https://openreview.net/forum?id=H1lQIgrFDS)] | \\(\ell_1\\), \\(\ell_2\\), \\(\ell_\infty\\) |
| \\(f\\)-Divergence [[6](https://openreview.net/forum?id=SJlKrkSFPH)] | \\(\ell_2\\)                                  |
| Renyi Divergence [[7](https://arxiv.org/abs/1809.03113v6)]   | \\(\ell_1\\)                                  |



- **Robust Training Approaches**: we evaluate 5 robust training approaches.

| Verification                                                 | Evaluated Adversary Type                      |
| ------------------------------------------------------------ | --------------------------------------------- |
| Data Augmentation [[2](https://arxiv.org/abs/1902.02918), [3](https://arxiv.org/abs/2002.08118)] | \\(\ell_1\\), \\(\ell_2\\), \\(\ell_\infty\\) |
| Adversarial Training [[8](https://arxiv.org/abs/1906.04584)] | \\(\ell_2\\)                                  |
| Adversarial + Pretraining [[8](https://arxiv.org/abs/1906.04584), [9](https://arxiv.org/abs/1905.13736)] | \\(\ell_2\\)                                  |
| MACER [[10](https://openreview.net/forum?id=rJx1Na4Fwr)]     | \\(\ell_2\\)                                  |
| ADRE [[11](https://arxiv.org/abs/2002.07246)]                | \\(\ell_2\\)                                  |



- **Smoothing Distributions**: we compare Gaussian, Laplace, and Uniform distribution.

" | markdownify | liquify
}}
      </div>
    </div>
  </div>

  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingFive-Prob">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseFive-Prob" aria-expanded="false" aria-controls="panelsStayOpen-collapseFive-Prob">
        Datasets, Models, and Evaluation Protocols
      </button>
    </h2>
    <div id="panelsStayOpen-collapseFive-Prob" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingFive-Prob">
      <div class="accordion-body">
        {{"

#### Datasets

we evaluate on CIFAR-10 and ImageNet. CIFAR-10 is a set of \\(3 \times 32 \times 32\\) images, and ImageNet is a set of images rescaled to \\(3 \times 224 \times 224\\).

#### Model Architectures
Following common practice in literature [[2](https://arxiv.org/abs/1902.02918), [3](https://arxiv.org/abs/2002.08118)],
we use ResNet-110 and Wide ResNet 40-2 on CIFAR-10; and
ResNet-50 on ImageNet.

#### Evaluation Protocol
Following common practice, on both datasets, \\(n = 1, 000\\) samples are
used for selecting the top label; \\(N = 10^5\\) samples are used
for certification. The failure probability is set to \\(1 − \alpha = .001\\).
For both datasets, we evaluate on a
subset of 500 samples uniformly drawn from the test set.
We report and compare the **certified accuracy** at given attack radius \\(\epsilon\\) under \\(\ell_p\\) adversaries for \\(p = 1, 2, \infty\\).
To evaluate the \\(\ell_\infty\\) robustness, we use the following norm conversion rule: 
given input dimension \\(d\\), 
the model with certified robust radius \\(\epsilon\\) in \\(\ell_2\\) norm at point \\(x_0\\) is also
certified robust with radius \\(\epsilon/\sqrt d\\) in \\(\ell_\infty\\) norm. 
The above conversion gives \\(\ell_\infty\\) robustness certification from existing L2-
based certification, which is empirically shown to achieve the
highest certified robustness for probabilistic approaches under
\\(\ell_\infty\\) adversary.

We tune the hyper-parameters to achieve the
best performance for each approach. 

" | markdownify | liquify}}
      </div>
    </div>
  </div>

  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingThree-Prob">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseThree-Prob" aria-expanded="false" aria-controls="panelsStayOpen-collapseThree-Prob">
        Full Results
      </button>
    </h2>
    <div id="panelsStayOpen-collapseThree-Prob" class="accordion-collapse collapse" aria-labelledby="panelsStayOpen-headingThree-Prob">
      <div class="accordion-body">

<!-- Main START -->
  {{"Choose the dataset:" | markdownify | liquify}}

<ul class="nav nav-pills mb-3" id="myTab" role="tablist" style="background-color:#f8f9fa; margin-left: 0px;">
  <li class="nav-item" role="presentation">
    <button class="nav-link active" id="prob-cifar-tab" data-bs-toggle="pill" data-bs-target="#prob-cifar" type="button" role="tab" aria-controls="home" aria-selected="true">CIFAR-10</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="prob-imagenet-tab" data-bs-toggle="pill" data-bs-target="#prob-imagenet" type="button" role="tab" aria-controls="profile" aria-selected="false">ImageNet</button>
  </li>
</ul>

<div class="tab-content" id="mainTabContent">
  <div class="tab-pane fade show active" id="prob-cifar" role="tabpanel" aria-labelledby="prob-cifar-tab">
    
<!-- CIFAR-10 prob main table -->

<table>
  <thead>
    <tr>
      <th>Adversary</th>
      <th>Model Architecture</th>
      <th>Verification</th>
      <th>Training</th>
      <th>Smoothing Distribution</th><th colspan="6">Certified Accuracy under Attack Radius \(\epsilon\)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><strong>\(\epsilon\)=</strong></td><td><strong>0.25</strong></td><td><strong>0.50</strong></td><td><strong>0.75</strong></td><td><strong>1.00</strong></td><td><strong>1.25</strong></td><td><strong>1.50</strong></td>
    </tr>
    <tr class='oddrow'>
      <td>\(\ell_2\)</td><td>Wide ResNet 40-2</td><td>Differential Privacy</td><td>Data Augmentation</td><td>Gaussian</td><td>34.2%</td><td>14.8%</td><td>6.8%</td><td>2.2%</td><td>0.0%</td><td>0.0%</td>
    </tr>
    <tr class='oddrow'>
      <td>\(\ell_2\)</td><td>Wide ResNet 40-2</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Gaussian</td><td><strong>68.8%</strong></td><td><strong>46.8%</strong></td><td><strong>36.0%</strong></td><td><strong>25.4%</strong></td><td><strong>19.8%</strong></td><td><strong>15.6%</strong></td>
    </tr>
    <tr class='oddrow'>
      <td>\(\ell_2\)</td><td>Wide ResNet 40-2</td><td>\(f\)-Divergence</td><td>Data Augmentation</td><td>Gaussian</td><td>62.2%</td><td>41.8%</td><td>27.2%</td><td>19.2%</td><td>14.2%</td><td>11.4%</td>
    </tr>
    <tr class='evenrow'>
      <td>\(\ell_2\)</td><td>ResNet-110</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Gaussian</td><td>61.2%</td><td>43.2%</td><td>32.0%</td><td>22.4%</td><td>17.2%</td><td>14.0%</td>
    </tr>
    <tr class='evenrow'>
      <td>\(\ell_2\)</td><td>ResNet-110</td><td>Neyman-Pearson</td><td>Adversarial Training</td><td>Gaussian</td><td>73.0%</td><td>57.8%</td><td>48.2%</td><td><strong>37.2%</strong></td><td>33.6%</td><td><strong>28.2%</strong></td>
    </tr>
    <tr class='evenrow'>
      <td>\(\ell_2\)</td><td>ResNet-110</td><td>Neyman-Pearson</td><td>Adversarial + Pretraining</td><td>Gaussian</td><td><strong>81.8%</strong></td><td><strong>62.6%</strong></td><td><strong>52.4%</strong></td><td>37.2%</td><td><strong>34.0%</strong></td><td>30.2%</td>
    </tr>
    <tr class='evenrow'>
      <td>\(\ell_2\)</td><td>ResNet-110</td><td>Neyman-Pearson</td><td>MACER</td><td>Gaussian</td><td>68.8%</td><td>52.6%</td><td>40.4%</td><td>33.0%</td><td>27.8%</td><td>25.0%</td>
    </tr>
    <tr class='evenrow'>
      <td>\(\ell_2\)</td><td>ResNet-110</td><td>Neyman-Pearson</td><td>ADRE</td><td>Gaussian</td><td>68.0%</td><td>50.2%</td><td>37.8%</td><td>30.2%</td><td>23.0%</td><td>17.0%</td>
    </tr>
    <tr>
      <td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><strong>\(\epsilon\)=</strong></td><td><strong>0.5</strong></td><td><strong>1.0</strong></td><td><strong>1.5</strong></td><td><strong>2.0</strong></td><td><strong>3.0</strong></td><td><strong>4.0</strong></td>
    </tr>
    <tr class='oddrow'>
      <td>\(\ell_1\)</td><td>Wide ResNet 40-2</td><td>Differential Privacy Based</td><td>Data Augmentation</td><td>Laplace</td><td>43.0%</td><td>20.8%</td><td>12.2%</td><td>7.2%</td><td>1.4%</td><td>0.0%</td>
    </tr>
    <tr class='oddrow'>
      <td>\(\ell_1\)</td><td>Wide ResNet 40-2</td><td>Renyi Divergence</td><td>Data Augmentation</td><td>Laplace</td><td>58.2%</td><td>39.4%</td><td>27.0%</td><td>16.8%</td><td>9.2%</td><td>4.0%</td>
    </tr>
    <tr class='oddrow'>
      <td>\(\ell_1\)</td><td>Wide ResNet 40-2</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Laplace</td><td>58.4%</td><td>39.6%</td><td>27.0%</td><td>17.2%</td><td>9.2%</td><td>4.2%</td>
    </tr>
    <tr class='oddrow'>
      <td>\(\ell_1\)</td><td>Wide ResNet 40-2</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Uniform</td><td><strong>69.2%</strong></td><td><strong>56.6%</strong></td><td><strong>48.0%</strong></td><td><strong>39.4%</strong></td><td><strong>26.0%</strong></td><td><strong>20.4%</strong></td>
    </tr>
    <tr>
      <td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><strong>\(\epsilon\)=</strong></td><td><strong>1/255</strong></td><td><strong>2/255</strong></td><td><strong>4/255</strong></td><td><strong>8/255</strong></td><td>&nbsp;</td><td>&nbsp;</td>
    </tr>
    <tr class='evenrow'>
      <td>\(\ell_\infty\)</td><td>Wide ResNet 40-2</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Gaussian</td><td>71.4%</td><td>52.0%</td><td>29.0%</td><td>12.8%</td><td>&nbsp;</td><td>&nbsp;</td>
    </tr>
    <tr class='evenrow'>
      <td>\(\ell_\infty\)</td><td>Wide ResNet 40-2</td><td>Neyman-Pearson</td><td>Adversarial Training</td><td>Gaussian</td><td><strong>83.2%</strong></td><td><strong>65.0%</strong></td><td><strong>49.6%</strong></td><td><strong>25.4%</strong></td><td>&nbsp;</td><td>&nbsp;</td>
    </tr>
  </tbody>
</table>

  </div>

  <div class="tab-pane fade" id="prob-imagenet" role="tabpanel" aria-labelledby="prob-imagenet-tab">

<!-- ImageNet prob main table -->
<table>
  <thead>
    <tr><th>Adversary</th><th>Model Architecture</th><th>Verification</th><th>Training</th><th>Smoothing Distribution</th><th colspan="6">Certified Accuracy under Attack Radius \(\epsilon\)</th></tr>
  </thead>
  <tbody>
    <tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><strong>\(\epsilon\)=</strong></td><td><strong>0.5</strong></td><td><strong>1.0</strong></td><td><strong>1.5</strong></td><td><strong>2.0</strong></td><td><strong>2.5</strong></td><td><strong>3.0</strong></td></tr>
    <tr class='oddrow'><td>\(\ell_2\)</td><td>ResNet-50</td><td>Differential Privacy Based</td><td>Data Augmentation</td><td>Gaussian</td><td>26.0%</td><td>12.2%</td><td>4.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr>
    <tr class='oddrow'><td>\(\ell_2\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Gaussian</td><td><strong>49.2%</strong></td><td><strong>37.4%</strong></td><td><strong>29.0%</strong></td><td><strong>19.2%</strong></td><td><strong>14.8%</strong></td><td><strong>12.0%</strong></td></tr>
    <tr class='oddrow'><td>\(\ell_2\)</td><td>ResNet-50</td><td>\(f\)-Divergence</td><td>Data Augmentation</td><td>Gaussian</td><td>43.4%</td><td>30.4%</td><td>13.6%</td><td>3.2%</td><td>0.0%</td><td>0.0%</td></tr>
    <tr class='evenrow'><td>\(\ell_2\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Gaussian</td><td>49.2%</td><td>37.4%</td><td>29.0%</td><td>19.2%</td><td>14.8%</td><td>12.0%</td></tr>
    <tr class='evenrow'><td>\(\ell_2\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>Adversarial Training</td><td>Gaussian</td><td>56.4%</td><td><strong>44.8%</strong></td><td><strong>38.2%</strong></td><td><strong>28.0%</strong></td><td><strong>25.6%</strong></td><td><strong>20.0%</strong></td></tr>
    <tr class='evenrow'><td>\(\ell_2\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>MACER</td><td>Gaussian</td><td><strong>57.0%</strong></td><td>43.2%</td><td>31.4%</td><td>24.8%</td><td>17.6%</td><td>14.0%</td></tr>
    <tr class='evenrow'><td>\(\ell_2\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>ADRE</td><td>Gaussian</td><td><strong>57.0%</strong></td><td>41.8%</td><td>30.0%</td><td>23.6%</td><td>17.8%</td><td>14.2%</td></tr>
    <tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><strong>\(\epsilon\)=</strong></td><td><strong>0.5</strong></td><td><strong>1.0</strong></td><td><strong>1.5</strong></td><td><strong>2.0</strong></td><td><strong>3.0</strong></td><td><strong>4.0</strong></td></tr>
    <tr class='oddrow'><td>\(\ell_1\)</td><td>ResNet-50</td><td>Differential Privacy Based</td><td>Data Augmentation</td><td>Laplace</td><td>39.0%</td><td>26.2%</td><td>17.8%</td><td>13.0%</td><td>6.8%</td><td>0.0%</td></tr>
    <tr class='oddrow'><td>\(\ell_1\)</td><td>ResNet-50</td><td>Renyi Divergence</td><td>Data Augmentation</td><td>Laplace</td><td>48.2%</td><td>40.4%</td><td>31.0%</td><td>25.8%</td><td>19.0%</td><td>13.6%</td></tr>
    <tr class='oddrow'><td>\(\ell_1\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Laplace</td><td>49.0%</td><td>40.8%</td><td>31.2%</td><td>26.0%</td><td>19.0%</td><td>13.6%</td></tr>
    <tr class='oddrow'><td>\(\ell_1\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Uniform</td><td><strong>55.2%</strong></td><td><strong>49.0%</strong></td><td><strong>45.6%</strong></td><td><strong>42.0%</strong></td><td><strong>32.8%</strong></td><td><strong>24.8%</strong></td></tr>
    <tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td><strong>\(\epsilon\)=</strong></td><td><strong>1/255</strong></td><td><strong>2/255</strong></td><td><strong>4/255</strong></td><td><strong>8/255</strong></td><td>&nbsp;</td><td>&nbsp;</td></tr>
    <tr class='evenrow'><td>\(\ell_\infty\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>Data Augmentation</td><td>Gaussian</td><td>28.2%</td><td>11.4%</td><td>0.0%</td><td>0.0%</td><td>&nbsp;</td><td>&nbsp;</td></tr>
    <tr class='evenrow'><td>\(\ell_\infty\)</td><td>ResNet-50</td><td>Neyman-Pearson</td><td>Adversarial Training</td><td>Gaussian</td><td><strong>38.2%</strong></td><td><strong>20.4%</strong></td><td><strong>4.6%</strong></td><td>0.0%</td><td>&nbsp;</td><td>&nbsp;</td></tr>
  </tbody>
</table>

  </div>
</div>

      The neighboring rows in the same background color (light yellow or light blue) form a variable-controlled comparable group, where the highest certified accuracies within each group are <strong>bolded</strong>.

      </div>
    </div>
  </div>

  <div class="accordion-item">
    <h2 class="accordion-header" id="panelsStayOpen-headingFour-Prob">
      <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#panelsStayOpen-collapseFour-Prob" aria-expanded="true" aria-controls="panelsStayOpen-collapseFour-Prob">
        Summary of Findings
      </button>
    </h2>
    <div id="panelsStayOpen-collapseFour-Prob" class="accordion-collapse collapse show" aria-labelledby="panelsStayOpen-headingFour-Prob">
      <div class="accordion-body">

{{
"
+ For both \\(\ell_1\\) and \\(\ell_2\\) adversaries,
Neyman-Pearson based verification achieves the tightest results
compared to others. 
+ The robust training approaches effectively enhance the models’ certified robustness. Among these
existing robust training approaches, adversarial training usually
achieves the best performance in terms of certified accuracy, and pretraining can further improve the performance.
+ The choice of smoothing distribution can greatly affect the certified accuracy of smoothed
models. One finding is that, under \\(\ell_1\\) adversary, the superior
result is achieved by uniform smoothing distribution. 
+ For probabilistic verification approaches, certifying robustness
under \\(\ell_\infty\\) adversary is challenging, and would become more
challenging when the data dimension increases (note that ImageNet data has higher dimension than CIFAR-10 data, thus yielding relatively lower certified accuracy).
" | markdownify | liquify}}


      </div>
    </div>
  </div>
</div>


