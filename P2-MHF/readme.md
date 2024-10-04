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

## [Create and publish an LLM.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P2-MHF/readme.md) 
In this small project we will create a new model alingning a couple of models a [microsoft-phi-3-model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and a [gemma-2b-it](https://huggingface.co/google/gemma-2b-it) with DPO and then publish it to Hugging Face.

Alignment is usually the final step taken when creating a model, after fine-tuning. Many people believe that the true revolution of GPT-3.5 was due to the alignment process that OpenAI carried out: Reinforcement Learning by Human Feedback, or RLFH.

RLHF proved to be a highly efficient technique for controlling the model's responses, and at first, it seemed that it had to be the price to pay for any model that wanted to compete with GPT-3.5.

Recently RLHF has been displaced by a technique that achieves the same result in a much more efficient way: DPO - Direct Preference Optimization.

The implementation of DPO that we'll be using in the notebooks is the one developed by Hugging Face in their TRL library, which stands for Transformer Reinforcement Learning. DPO can be considered a reinforcement learning technique, where the model is rewarded during its training phase based on its responses.

This library greatly simplifies the implementation of DPO. All you have to do is specify the model you want to fine-tune and provide it with a dataset in the necessary format.

The dataset to be used should have three columns:
* Prompt: The prompt used.
* Chosen: The desired response.
* Rejected: An undesired response.

The dataset selected for this example is [argilla/distilabel-capybara-dpo-7k-binarized](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized)

If you'd like to take a look at the models resulting from two hours of training on an A100 GPU with the Argila dataset, you can do so on its Hugging Face page. 
* [martra-open-gemma-2b-it-dpo](https://huggingface.co/oopere/martra-open-gemma-2b-it-dpo)
* [martra-phi-3-mini-dpo](https://huggingface.co/oopere/martra-phi-3-mini-dpo)


| Article | Notebook |
| --- | --- |
| WIP | [Aligning with DPO a phi3-3 model.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P2-MHF/7_2_Aligning_DPO_phi3.ipynb) <br/> [Aligning with DPO a Gemma model.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P2-MHF/Aligning_DPO_open_gemma-2b-it.ipynb)|
