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
        <a href="https://github.com/Apress/Large-Language-Models-Projects">Apress repository</a>, where you can find all the notebooks in their original format as they appear in the book.
      </p>
    </td>
  </tr>
</table>

In this section, we will explore different techniques such as Prompt Fine Tuning or LoRA, and we will use the Hugging Face PEFT library to efficiently fine-tune Large Language Models.
Additionally, we will explore optimization techniques such as model quantization.

### Fine-Tuning with LoRA using PEFT from Hugging Face. 
After a brief explanation of how the fine-tuning technique LoRA works, we will fine-tune a model from the Bloom family to teach it to construct prompts that can be used to instruct large language models.
| [Article](https://medium.com/gitconnected/efficient-fine-tuning-with-lora-optimal-training-for-large-language-models-266b63c973ca) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/5_2_LoRA_Tuning.ipynb) |
| --- | --- |

### Fine-Tuning a 7B Model in a single 16GB GPU using QLoRA.
We are going to see a brief introduction to quantization, used to reduce the size of big Large Language Models. With quantization you can load big models reducing the memory resources needed. It also applies to the fine-tunig process, you can fine-tune the model in a single GPU without consuming all the resources. After the brief explanation we see an example about how is possible to fine-tune a Bloom 7B Model ina a t4 16GB GPU on Google Colab.
| [Article](https://medium.com/towards-artificial-intelligence/qlora-training-a-large-language-model-on-a-16gb-gpu-00ea965667c1) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/5_3_QLoRA_Tuning.ipynb)
| --- | --- |

### Prompt tuning using PEFT Library from Hugging Face. 
In this notebook, two models are trained using Prompt Tuning from the PEFT library. This technique not only allows us to train by modifying the weights of very few parameters but also enables us to have different specialized models loaded in memory that use the same foundational model.

Prompt tuning is an additive technique, and the weights of the pre-trained model are not modified. The weights that we modify in this case are those of virtual tokens that we add to the prompt.
| [Article](https://medium.com/towards-artificial-intelligence/fine-tuning-models-using-prompt-tuning-with-hugging-faces-peft-library-998ae361ee27) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/5_4_Prompt_Tuning.ipynb)|
| --- | --- |

# Papers used in the Chapter: 
[The Power of Scale for Parameter-Efficient Prompt Tuning](https://doi.org/10.48550/arXiv.2104.08691). Prompt tuning using PEFT Library from Hugging Face. 

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Fine-Tuning with LoRA using PEFT from Hugging Face.

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). Fine Tuning & Optimizacion Lesson. QLoRA Fine-Tuning Sample.
