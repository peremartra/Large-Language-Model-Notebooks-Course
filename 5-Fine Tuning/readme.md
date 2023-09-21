In this section, we will explore different techniques such as Prompt Fine Tuning or LoRA, and we will use the Hugging Face PEFT library to efficiently fine-tune Large Language Models.

### Prompt tuning using PEFT Library from Hugging Face. 
In this notebook, two models are trained using Prompt Tuning from the PEFT library. This technique not only allows us to train by modifying the weights of very few parameters but also enables us to have different specialized models loaded in memory that use the same foundational model.

Prompt tuning is an additive technique, and the weights of the pre-trained model are not modified. The weights that we modify in this case are those of virtual tokens that we add to the prompt.
* notebook: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/Prompt_Tuning_PEFT.ipynb

# Papers used in the Lesson: 
[arXiv:2104.08691](https://doi.org/10.48550/arXiv.2104.08691). Prompt Tuning Sample. 
