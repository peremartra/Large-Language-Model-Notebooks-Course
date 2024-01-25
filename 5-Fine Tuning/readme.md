In this section, we will explore different techniques such as Prompt Fine Tuning or LoRA, and we will use the Hugging Face PEFT library to efficiently fine-tune Large Language Models.
Additionally, we will explore optimization techniques such as model quantization.

### Prompt tuning using PEFT Library from Hugging Face. 
In this notebook, two models are trained using Prompt Tuning from the PEFT library. This technique not only allows us to train by modifying the weights of very few parameters but also enables us to have different specialized models loaded in memory that use the same foundational model.

Prompt tuning is an additive technique, and the weights of the pre-trained model are not modified. The weights that we modify in this case are those of virtual tokens that we add to the prompt.
| [Article](https://medium.com/towards-artificial-intelligence/fine-tuning-models-using-prompt-tuning-with-hugging-faces-peft-library-998ae361ee27) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/Prompt_Tuning_PEFT.ipynb) |
| --- | --- |

### Fine-Tuning with LoRA using PEFT from Hugging Face. 
After a brief explanation of how the fine-tuning technique LoRA works, we will fine-tune a model from the Bloom family to teach it to construct prompts that can be used to instruct large language models.
| [Article](https://medium.com/towards-artificial-intelligence/fine-tuning-models-using-prompt-tuning-with-hugging-faces-peft-library-998ae361ee27) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb) |
| --- | --- |

### Fine-Tuning a 7B Model in a single 16GB GPU using QLoRA.
We are going to see a brief introduction to quantization, used to reduce the size of big Large Language Models. With quantization you can load big models reducing the memory resources needed. It also applies to the fine-tunig process, you can fine-tune the model in a single GPU without consuming all the resources. After the brief explanation we see an example about how is possible to fine-tune a Bloom 7B Model ina a t4 16GB GPU on Google Colab.
| [Article](https://medium.com/towards-artificial-intelligence/qlora-training-a-large-language-model-on-a-16gb-gpu-00ea965667c1) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/QLoRA_Tuning_PEFT.ipynb)
| --- | --- |

# Papers used in the Chapter: 
[The Power of Scale for Parameter-Efficient Prompt Tuning](https://doi.org/10.48550/arXiv.2104.08691). Prompt tuning using PEFT Library from Hugging Face. 

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Fine-Tuning with LoRA using PEFT from Hugging Face.

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). Fine Tuning & Optimizacion Lesson. QLoRA Fine-Tuning Sample.
