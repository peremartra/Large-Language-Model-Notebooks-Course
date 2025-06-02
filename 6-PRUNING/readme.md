**disclaimer: The pruning section was created after the first edition of the book was published. They are not included in the book’s original content but are intended to supplement and expand on the topics covered.**

<table>
  <tr>
    <td  width="130">
      <a href="https://amzn.to/4eanT1g">
        <img src="https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/img/Large_Language_Models_Projects_Book.jpg" height="160" width="104">
      </a>
    </td>
    <td>
      <p>
        This is the unofficial repository for the book: 
        <a href="https://amzn.to/4eanT1g"> <b>Large Language Models:</b> Apply and Implement Strategies for Large Language Models</a> (Apress).
        The book is based on the content of this repository, but the notebooks are being updated, and I am incorporating new examples and chapters.
        If you are looking for the official repository for the book, with the original notebooks, you should visit the 
        <a href="https://github.com/Apress/Large-Language-Models-Projects">Apress repository</a>, where you can find all the notebooks in their original format as they appear in the book. Buy it at: <a href="https://amzn.to/3Bq2zqs">[Amazon]</a> <a href="https://link.springer.com/book/10.1007/979-8-8688-0515-8">[Springer]</a>
      </p>
    </td>
  </tr>
</table>

# Pruning Techniques for Large Language Models
**This section is still under construction. The goal is to build a curriculum that will take us from the most simple pruning techniques to creating a model using the same techniques employed by leading companies in the field, such as Microsoft, Google, Nvidia, or OpenAI, to build their models.**

📢 A portion of the structured pruning work in this section was presented at the [STAC Summit 2025 in London](https://www.stacresearch.com/spring2025LON), an international event focused on AI in finance. The talk explored pruning strategies for LLMs models, with a focus on performance-efficiency trade-offs and carbon footprint reduction.

Pruning is a crucial optimization technique in machine learning, particularly for large language models (LLMs). It involves reducing the size and complexity of a model by eliminating less important components—such as neurons, layers, or weights, while maintaining most of the model’s performance. Pruning helps to make models more efficient, reducing their computational and memory requirements, which is especially important when deploying models on resource-constrained environments like mobile devices or edge servers.

One of the great advantages of pruning compared to other techniques like quantization is that when selecting parts of the model to remove, you can choose those that contribute less to the model’s output, depending on the intended use.

## Structured Width Pruning: Eliminating Less Important Neurons from Feedforward Layers.
In this type of pruning, the neurons that contribute least to the model's output are identified and removed. It is crucial to know or have access to the model's structure, as the pruning process is not applied to all layers. Depending on the need, specific layers are selected for pruning.

### Prune a distilGPT2 model using l1 norm to determine less important neurons. 
In the first notebook, the pruning process will be applied to the feedforward layers of a distilGPT2 model. This means the model will have reduced weights in those specific layers. The neurons to prune are selected based on their importance scores, which we compute using the L1 norm of their weights. It is a simple aproach, for this first example, that can be used when you want to create a Pruned Model that mimics the Base model in all the areas. 

By altering the model's structure, a new configuration file must be created to ensure it works correctly with the `transformers` library.

| [Notebook: Pruning a distilGPT2 model.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_1_pruning_structured_l1_diltilgpt2.ipynb) |
| --- |

### Prune a Llama3.2 model. 
In this first notebook, we attempt to replicate the pruning process used with the distilGPT2 model but applied to a Llama model. By not taking the model's characteristics into account, the pruning process results in a completely unusable model. This notebook serves as an exercise to understand how crucial it is to know the structure of the models that will undergo pruning.
| [Notebook: Pruning a Llama3.2 model INCORRECT APROACH.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_2_pruning_structured_llama3.2-1b_KO.ipynb) |
| --- |


The second notebook addresses the issues encountered when applying the same pruning process to the Llama model as was used for distilGPT2.

The correct approach is to treat the MLP layers of the model as pairs rather than individual layers and to calculate neuron importance by considering both layers together. Additionally, we switch to using the maximum absolute weight to decide which neurons remain in the pruned layers.

With these two simple changes, which adapt the pruning process to the structure of Llama models, we achieve a smaller model that retains a significant portion of the original model's knowledge.
| [Pruning Llama3 Article](https://medium.com/towards-data-science/how-to-prune-llama-3-2-and-similar-large-language-models-cf18e9a2afb6?sk=af4c5e40e967437325050f019b3ae606) | [Notebook: Pruning a Llama3.2 model CORRECT APROACH.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb) |
| --- | --- | 

## Structured Depth Pruning. Eliminating complete blocks from large language models. 
### Depth pruning in a Llama-3.2 model. 
In this notebook, we will look at an example of depth pruning, which involves removing entire layers from the model.
The first thing to note is that removing entire layers from a transformer model usually has a significant impact on the model's performance. This is a much more drastic architectural change compared to the simple removal of neurons from the MLP layers, as seen in the previous example.
| [Notebook: Depth pruning a Llama Model.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_5_pruning_depth_st_llama3.2-1b_OK.ipynb) |
| --- |

## Attention Bypass
### Pruning Attention Layers. 
This notebook implements the ideas presented in the paper: [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786). 

In this notebook, the attention layers that contribute the least to the model are marked to be bypassed, improving inference efficiency and reducing the model's resource consumption. To identify the layers that contribute the least, a simple activation with a prompt is used, and the cosine similarity between the layer's input and output is measured. The smaller the difference, the less modification the layer introduces.

The layer selection process implemented in the notebook is iterative. That is, the least contributing layer is selected, and the contribution of the remaining layers is recalculated using the same prompt. This process is repeated until the desired number of layers has been deactivated.

Since this type of pruning does not alter the model's structure, it does not reduce the model's size.
| Article: WIP. | [Notebook: Pruning Attention Layers.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_6_pruning_attention_layers.ipynb) |
| --- | --- |

### Adaptive Attention Bypass. 
Adaptive models are those that can dynamically adapt their structure or change the parts they execute, either while creating the response or upon receiving the user's request. This notebook represents one of the first, if not the first, implementations of an adaptive model compatible with the Transformers library.

The resulting model is capable of deciding which attention layers to execute depending on the complexity of the prompt it receives. It is the most complex notebook in the entire repository and is very close to what can be considered pure research. In fact, there is no paper that describes the functioning of the implemented method, so it is considered an original work by the author (Pere Martra).

The model goes through a calibration process in which the importance of each layer is decided, and a configuration file is created. For each prompt received, the complexity is calculated using its length and embedding variance, and the model decides which layers it should use to provide a response to the user.
| Article: WIP. | [Notebook: Adaptive Attention Bypass.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_6b_Adaptive_Inference_Attention_Pruning.ipynb) |
| --- | --- |

# Knowledge distillation. 
Knowledge Distillation involves training a smaller "student" model to mimic a larger, well-trained "teacher" model. The student learns not just from the correct labels but also from the probability distributions (soft targets) that the teacher model produces, effectively transferring the teacher's learned knowledge into a more compact form.

When combined with Pruning, you first create a pruned version of your base model by removing less important connections. During this process, some knowledge is inevitably lost. To recover this lost knowledge, you can apply Knowledge Distillation using the original base model as the teacher and the pruned model as the student, helping to restore some of the lost performance.

Both techniques address the same challenge: reducing model size and computational requirements while maintaining performance, making them crucial for deploying AI in resource-constrained environments like mobile devices.

## Recovering knwoledge from the base model using KD. 
In this notebook, we will use Knowledge distillation to recover some of the knowledge lost during the model pruning process. Llama-3.2-1B will be used as the Teacher model, and the 40% pruned version will be used as the Student model. We will specifically improve the performance on the Lambada benchmark.
| [Notebook: Knowledge Distillation Llama 3.2.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/7_1_knowledge_distillation_Llama.ipynb) |
| --- |

# Bias & Fairness in LLMs. 
This section introduces a preliminary line of work focused on detecting bias in LLMs by visualizing neural activations. While still in its early stages, these analyses pave the way for future fairness-aware pruning strategies, where structural pruning decisions also take into account the impact on different demographic or semantic groups.

### Visualizing Bias in State of the art Transformer Models. 
This notebook introduces techniques for visualizing neural activations in Transformer models, as a first step toward detecting and mitigating bias in language models.
Techniques applied:
  *  Dimensionality reduction with PCA
  *  Visualization using heatmaps
  *  Differential activation analysis between contrastive groups

| [Article: From Biased to Balanced: Visualizing and Fixing Bias in Transformer Models](https://medium.com/data-science-collective/from-biased-to-balanced-visualizing-and-fixing-bias-in-transformer-models-d1a82f35393c?sk=abd12073ee311c3752da3219a5baf20f) | [Notebook: 8_1_transformer_activations_visualization.ipynb](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/8_1_transformer_activations_visualization.ipynb) |
| --- | --- |

Why is it in the Pruning section?
While this work is thematically aligned with fairness and equity evaluation in models, it's temporarily placed here due to its direct connection with upcoming fairness-aware pruning strategies. These strategies explore activation patterns as a criterion for reducing parameters while preserving the representation of different groups.

This notebook will serve as the foundation for the future section 8-Bias, which will include:
* Tools for automated bias analysis
* Fairness experiments in pruned models
* Integration with the WizardSData library for contrastive dataset generation

### References
> Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, Pavlo Molchanov, "Compact Language Models via Pruning and Knowledge Distillation," arXiv preprint arXiv:2407.14679, 2024. Available at: [https://doi.org/10.48550/arXiv.2407.14679](https://doi.org/10.48550/arXiv.2407.14679).

> He, S., Sun, G., Shen, Z., & Li, A. (2024). What matters in transformers? not all attention is needed. arXiv preprint arXiv:2406.15786. https://doi.org/10.48550/arXiv.2406.15786


> Kim, B. K., Kim, G., Kim, T. H., Castells, T., Choi, S., Shin, J., & Song, H. K. (2024). Shortened llama: A simple depth pruning for large language models. arXiv preprint arXiv:2402.02834, 11. https://doi.org/10.48550/arXiv.2402.02834

> Martra, P. (2024, December 26). Exploring GLU Expansion Ratios: Structured Pruning in Llama-3.2 Models. https://doi.org/10.31219/osf.io/qgxea














