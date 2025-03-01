# Task 3: Exploration of Data Selection and Fine-Tuning Methods

In this task, you can explore alternative methods for data selection and investigate several fine-tuning techniques to adapt a pre-trained model for a specific task. The goal is to improve the model performance on our target dataset. You can use the [`task3.py`](../scripts/Task3.py) file for your implementation.

1. Data Selection Strategies
The first step in fine-tuning a model is to carefully select the training data. While the previous tasks focused on influence-based data selection, here you will experiment with other selection strategies. Pick one data selection method by yourself. Log your findings about the selected data subsets:
    - How much data is used in each strategy?
    - Compare the performance of models trained with each selection method.

2. Fine-tuning Strategies
In this section, you will implement and compare some parameter-efficient fine-tuning approaches:

    - [bitfit](https://arxiv.org/abs/2106.10199)
    - [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)
    - iA3 (see Section 3.3. of this [paper](https://arxiv.org/abs/2205.05638)) (Implicit Adapter)