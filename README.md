# Build with LLMs: Hands-on Projects for Engineers, Researchers and Developers using Large Language Models, GPT, LLaMA, LangChain and Hugging Face

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

*Please note that the course on GitHub does not contain all the information that is in the book.*

**This practical free hands on course about Large Language models and their applications is 👷🏼in permanent development👷🏼. I will be posting the different lessons and samples as I complete them.**

The course provides a hands-on experience using models from OpenAI and the Hugging Face library. We are going to see and use a lot of tools and practice with small projects that will grow as we can apply the new knowledge acquired. 

![Large Language Models Course Path](LLM_course_diagram.jpg)

<h1> The course is divided into three major sections:</h1>

<h2>1- Techniques and Libraries:</h2> 
In this part, we will explore different techniques through small examples that will enable us to build bigger projects in the following section. We will learn how to use the most common libraries in the world of Large Language Models, always with a practical focus, while basing our approach on published papers.

Some of the topics and technologies covered in this section include: Chatbots, Code Generation, OpenAI API, Hugging Face, Vector databases, LangChain, Fine Tuning, PEFT Fine Tuning, Soft Prompt tuning, LoRA, QLoRA, Evaluate Models, Knowledge Distillation.

<h2>2- Projects:</h2> 
We will create projects, explaining design decisions. Each project may have more than one possible implementation, as often there is not just one perfect solution. In this section, we will also delve into LLMOps-related topics, although it is not the primary focus of the course.

<h2>3- Enterprise Solutions:</h2> Large Language Models are not a standalone solution. In large corporate environments, they are just one piece of the puzzle. We will explore how to structure solutions capable of transforming organizations with thousands of employees, and how Large Language Models play a main role in these new solutions.

<h1>How to use the course.</h1>
Under each section you can find different chapters, that are formed by different lessons. The title of the lesson is a link to the lesson page, where you can found all the notebooks and articles of the lesson. 

Each Lesson is conformed by notebooks and articles. The notebooks contain sufficient information for understanding the code within them, the article provides more detailed explanations about the code and the topic covered. 

My advice is to have the article open alongside the notebook and follow along. Many of the articles offer small tips on variations that you can introduce to the notebooks. I recommend following them to enhance clarity of the concepts.

Most of the notebooks are hosted on Colab, while a few are on Kaggle. Kaggle provides more memory in the free version compared to Colab, but I find that copying and sharing notebooks is simpler in Colab, and not everyone has a Kaggle account.

Some of the notebooks require more memory than what the free version of Colab provides. As we are working with large language models, this is a common situation that will recur if you continue working with them. You can run the notebooks in your own environment or opt for the Pro version of Colab.
_____________
<h1>🚀1- Techniques and Libraries.</h1>

Each notebook is supported with a Medium article where the code is explained in detail. 
## [Introduction to Large Language Models with OpenAI.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/1-Introduction%20to%20LLMs%20with%20OpenAI)
In this first section of the course, we will learn to work with the OpenAI API by creating two small projects. We'll delve into OpenAI's roles and how to provide the necessary instructions to the model through the prompt to make it behave as we desire.

The first project is a restaurant chatbot where the model will take customer orders. Building upon this project, we will construct an SQL statement generator. Here, we'll attempt to create a secure prompt that only accepts SQL creation commands and nothing else.

### Create Your First Chatbot Using GPT 3.5, OpenAI, Python and Panel.
We will be utilizing OpenAI GPT-3.5 and Panel to develop a straightforward Chatbot tailored for a fast food restaurant. During the course, we will explore the fundamentals of prompt engineering, including understanding the various OpenAI roles, manipulating temperature settings, and how to avoid Prompt Injections. 
| [article panel](https://medium.com/towards-artificial-intelligence/create-your-first-chatbot-using-gpt-3-5-openai-python-and-panel-7ec180b9d7f2) / [article gradio](https://ai.plainenglish.io/create-a-simple-chatbot-with-openai-and-gradio-202684d18f35?sk=e449515ec7a803ae828418011bbaca52)| [Notebook panel](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_1-First_Chatbot_OpenAI.ipynb) / [notebook gradio](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_1-First_Chatbot_OpenAI_Gradio.ipynb)|
| --- | --- |

### How to Create a Natural Language to SQL Translator Using OpenAI API.
Following the same framework utilized in the previous article to create the ChatBot, we made a few modifications to develop a Natural Language to SQL translator. In this case, the Model needs to be provided with the table structures, and adjustments were made to the prompt to ensure smooth functionality and avoid any potential malfunctions. With these modifications in place, the translator is capable of converting natural language queries into SQL queries. @fmquaglia has created a notebook using DBML to describe the tables that by far is a better aproach than the original.
| [Article](https://pub.towardsai.net/how-to-create-a-natural-language-to-sql-translator-using-openai-api-e1b1f72ac35a) / [Article Gradio](https://medium.com/towards-artificial-intelligence/first-nl2sql-chat-with-openai-and-gradio-b1de0d6541b4)| [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_2-Easy_NL2SQL.ipynb) / [Notebook Gradio](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_2-Easy_NL2SQL_Gradio.ipynb) / [Notebook DBML](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_2b-Easy_NL2SQL.ipynb)
| --- | --- |

### Brief Introduction to Prompt Engineering with OpenAI.
We will explore prompt engineering techniques to improve the results we obtain from Models. Like how to format the answer and obtain a structured response using Few Shot Samples. 
| [Article](https://medium.com/gitconnected/influencing-a-large-language-model-response-with-in-context-learning-b212f0eaa113) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_3-Intro_Prompt_Engineering.ipynb)
| --- | --- |

## [Vector Databases with LLMs.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/2-Vector%20Databases%20with%20LLMs) 
A brief introduction to Vector Databases, a technology that will accompany us in many lessons throughout the course. We will work on an example of Retrieval Augmented Generation using information from various news datasets stored in ChromaDB.

### Influencing Language Models with Personalized Information using a Vector Database. 
If there's one aspect gaining importance in the world of large language models, it's exploring how to leverage proprietary information with them. In this lesson, we explore a possible solution that involves storing information in a vector database, ChromaDB in our case, and using it to create enriched prompts.
|[Article](https://pub.towardsai.net/harness-the-power-of-vector-databases-influencing-language-models-with-personalized-information-ab2f995f09ba?sk=ea2c5286fbff8430e5128b0c3588dbab) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/2_1_Vector_Databases_LLMs.ipynb) |
| --- | --- |

### Semantic Cache for RAG systems 
We enhanced the RAG system by introducing a semantic cache layer capable of determining if a similar question has been asked before. If affirmative, it retrieves information from a cache system created with Faiss instead of accessing the Vector Database. 

The inspiration and base code of the semantic cache present in this notebook exist thanks to the course: https://maven.com/boring-bot/advanced-llm/1/home from Hamza Farooq.

| Article | Notebook |
| --- | ---|
| WIP | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/semantic_cache_chroma_vector_database.ipynb) |


## [LangChain.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/3-LangChain)
LangChain has been one of the libraries in the universe of large language models that has contributed the most to this revolution. 
It allows us to chain calls to Models and other systems, allowing us to build applications based on large language models. In the course, we will use it several times, creating increasingly complex projects.

### Retrieval Augmented Generation (RAG). Use the Data from your DataFrames with LLMs.
In this lesson, we used LangChain to enhance the notebook from the previous lesson, where we used data from two datasets to create an enriched prompt. This time, with the help of LangChain, we built a pipeline that is responsible for retrieving data from the vector database and passing it to the Language Model. The notebook is set up to work with two different datasets and two different models. One of the models is trained for Text Generation, while the other is trained for Text2Text Generation.
| [Article](https://medium.com/towards-artificial-intelligence/query-your-dataframes-with-powerful-large-language-models-using-langchain-abe25782def5) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/3_1_RAG_langchain.ipynb) |
| --- | --- |

### Create a Moderation system using LangChain. 
We will create a comment response system using a two-model pipeline built with LangChain. In this setup, the second model will be responsible for moderating the responses generated by the first model.

One effective way to prevent our system from generating unwanted responses is by using  a second model that has no direct interaction with users to handle response generation. 

This approach can reduce the risk of undesired responses generated by the first model in response to the user's entry. 


I will create separate notebooks for this task. One will involve models from OpenAI, and the others will utilize open-source models provided by Hugging Face. The results obtained in the three notebooks are very different. The system works much better with the OpenAI, and LLAMA2 models. 
| Article | Notebook |
| --- | --- |
| [OpenAI article](https://pub.towardsai.net/create-a-self-moderated-commentary-system-with-langchain-and-openai-406a51ce0c8d?sk=b4903b827e44642f7f7c311cebaef57f) | [OpenAI notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/3_2_OpenAI_Moderation_Chat.ipynb) |
| [Llama2-7B Article](https://levelup.gitconnected.com/create-a-self-moderated-comment-system-with-llama-2-and-langchain-656f482a48be?sk=701ead7afb80e015ea4345943a1aeb1d) | [Llama2-7B Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/3_2_LLAMA2_Moderation_Chat.ipynb) |
| No Article | [GPT-J Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/3_2_GPT_Moderation_System.ipynb) |

### Create a Data Analyst Assistant using a LLM Agent. 
Agents are one of the most powerful tools in the world of Large Language Models. The agent is capable of interpreting the user's request and using the tools and libraries at its disposal until it achieves the expected result.

With LangChain Agents, we are going to create in just a few lines one of the simplest yet incredibly powerful agents. The agent will act as a Data Analyst Assistant and help us in analyzing data contained in any Excel file. It will be able to identify trends, use models, make forecasts. In summary, we are going to create a simple agent that we can use in our daily work to analyze our data.
| [Article](https://pub.towardsai.net/create-your-own-data-analyst-assistant-with-langchain-agents-722f1cdcdd7e) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/3_3_Data_Analyst_Agent.ipynb) |
| --- | --- |

### Create a Medical ChatBot with LangChain and ChromaDB. 
In this example, two technologies seen previously are combined: agents and vector databases. Medical information is stored in ChromaDB, and a LangChain Agent is created, which will fetch it only when necessary to create an enriched prompt that will be sent to the model to answer the user's question.

In other words, a RAG system is created to assist a Medical ChatBot.

**Attention!!! Use it only as an example. Nobody should take the boot's recommendations as those of a real doctor. I disclaim all responsibility for the use that may be given to the ChatBot. I have built it only as an example of different technologies.**
| [Article](https://medium.com/towards-artificial-intelligence/query-your-dataframes-with-powerful-large-language-models-using-langchain-abe25782def5) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/3_4_Medical_Assistant_Agent.ipynb) |
| ------ | ------ |

## [Evaluating LLMs.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/4-Evaluating%20LLMs)
The metrics used to measure the performance of Large Language Models are quite different from the ones we've been using in more traditional models. We're shifting away from metrics like Accuracy, F1 score, or recall, and moving towards metrics like BLEU, ROUGE, or METEOR. 

These metrics are tailored to the specific task assigned to the language model. 

In this section, we'll explore examples of several of these metrics and how to use them to determine whether one model is superior to another for a given task. We'll delve into practical scenarios where these metrics help us make informed decisions about the performance of different models.

### Evaluating translations with BLEU. 
Bleu is one of the first Metrics stablished to evaluate the quality of translations. In the notebook we compare the quality of a translation made by google with other from an Open Source Model from Hugging Face.
| Article WIP | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/4_1_bleu_evaluation.ipynb) |
| --- | --- |

### Evaluating Summarization with ROUGE. 
We will explore the usage of the ROUGE metric to measure the quality of summaries generated by a language model. 
We are going to use two T5 models, one of them being the t5-Base model and the other a t5-base fine-tuned specifically designed for creating summaries.

| [Article](https://medium.com/towards-artificial-intelligence/rouge-metrics-evaluating-summaries-in-large-language-models-d200ee7ca0e6) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/4_1_rouge_evaluations.ipynb) |
| --- | --- |

### Monitor an Agent using LangSmith. 
In this initial example, you can observe how to use LangSmith to monitor the traffic between the various components that make up the Agent. The agent is a RAG system that utilizes a vectorial database to construct an enriched prompt and pass it to the model. LangSmith captures both the use of the Agent's tools and the decisions made by the model, providing information at all times about the sent/received data, consumed tokens, the duration of the query, and all of this in a truly user-friendly environment.
| [Article](https://medium.com/towards-artificial-intelligence/tracing-a-llm-agent-with-langsmith-a81975634555)  | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/4_2_tracing_medical_agent.ipynb) |
| ------ | ------ |

### Evaluating the quality of summaries using Embedding distance with LangSmith. 
Previously in the notebook, Rouge Metrics: Evaluating Summaries, we learned how to use ROUGE to evaluate which summary best approximated the one created by a human. This time, we will use embedding distance and LangSmith to verify which model produces summaries more similar to the reference ones.
| [Article](https://medium.com/towards-artificial-intelligence/evaluating-llm-summaries-using-embedding-distance-with-langsmith-5fb46fdae2a5?sk=24eb18ce187d28547cebd6fd3dd1ddad) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/4_2_Evaluating_summaries_embeddings.ipynb) |
| ------ | ------ |

### Evaluating a RAG solution using Giskard. 
We take the agent that functions as a medical assistant and incorporate Giskard to evaluate if its responses are correct. In this way, not only the model's response is evaluated, but also the information retrieval in the vector database. Giskard is a solution that allows evaluating a complete RAG solution.
| [Article](https://medium.com/towards-artificial-intelligence/evaluating-a-rag-solution-with-giskard-1bc138fa44af?sk=10811fe2953eb511fb1ffefda326f7a2) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/4_3_evaluating_rag_giskard.ipynb)
| ------ | ------ |

### Introduction to the lm-evaluation library from Eluther.ai. 
The lm-eval library by EleutherAI provides easy access to academic benchmarks that have become industry standards. It supports the evaluation of both Open Source models and APIs from providers like OpenAI, and even allows for the evaluation of adapters created using techniques such as LoRA.

In this notebook, I will focus on a small but important feature of the library: evaluating models compatible with Hugging Face's Transformers library.
| Article - WIP| [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/4_4_lm-evaluation-harness.ipynb)
| ------ | ------ |


## [Fine Tuning & Optimization.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/5-Fine%20Tuning) 
In the FineTuning & Optimization section, we will explore different techniques such as Prompt Fine Tuning or LoRA, and we will use the Hugging Face PEFT library to efficiently fine-tune Large Language Models. We will explore techniques like quantization to reduce the weight of the Models. 

### Prompt tuning using PEFT Library from Hugging Face. 
In this notebook, two models are trained using Prompt Tuning from the PEFT library. This technique not only allows us to train by modifying the weights of very few parameters but also enables us to have different specialized models loaded in memory that use the same foundational model.

Prompt tuning is an additive technique, and the weights of the pre-trained model are not modified. The weights that we modify in this case are those of virtual tokens that we add to the prompt.
| [Article](https://medium.com/towards-artificial-intelligence/fine-tuning-models-using-prompt-tuning-with-hugging-faces-peft-library-998ae361ee27) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/5_4_Prompt_Tuning.ipynb) |
| --- | --- |

### Fine-Tuning with LoRA using PEFT from Hugging Face. 
After a brief explanation of how the fine-tuning technique LoRA works, we will fine-tune a model from the Bloom family to teach it to construct prompts that can be used to instruct large language models.
|[Article](https://levelup.gitconnected.com/efficient-fine-tuning-with-lora-optimal-training-for-large-language-models-266b63c973ca?sk=85d7b5d78e64e568faedfe07a35f81bd) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/5_2_LoRA_Tuning.ipynb)
| --- | --- |

### Fine-Tuning a 7B Model in a single 16GB GPU using QLoRA.
We are going to see a brief introduction to quantization, used to reduce the size of big Large Language Models. With quantization, you can load big models reducing the memory resources needed. It also applies to the fine-tuning process, you can fine-tune the model in a single GPU without consuming all the resources. 
After the brief explanation we see an example about how is possible to fine-tune a Bloom 7B Model ina a t4 16GB GPU on Google Colab. 
| [Article](https://medium.com/towards-artificial-intelligence/qlora-training-a-large-language-model-on-a-16gb-gpu-00ea965667c1) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/5_3_QLoRA_Tuning.ipynb) |
| --- | --- |

## [Pruning Techniques for Large Language Models](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/6-PRUNING)
**This section is still under construction. The goal is to build a curriculum that will take us from the most simple pruning techniques to creating a model using the same techniques employed by leading companies in the field, such as Microsoft, Google, Nvidia, or OpenAI, to build their models.**

### Prune a distilGPT2 model using l1 norm to determine less important neurons. 
In the first notebook, the pruning process will be applied to the feedforward layers of a distilGPT2 model. This means the model will have reduced weights in those specific layers. The neurons to prune are selected based on their importance scores, which we compute using the L1 norm of their weights. It is a simple aproach, for this first example, that can be used when you want to create a Pruned Model that mimics the Base model in all the areas. 

By altering the model's structure, a new configuration file must be created to ensure it works correctly with the `transformers` library.

| [Notebook: Pruning a distilGPT2 model.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_1_pruning_structured_l1_diltilgpt2.ipynb) |
| --- |

### Prune a Llama3.2 model. 
In this first notebook, we attempt to replicate the pruning process used with the distilGPT2 model but applied to a Llama model. By not taking the model's characteristics into account, the pruning process results in a completely unusable model. This notebook serves as an exercise to understand how crucial it is to know the structure of the models that will undergo pruning.
| [Notebook: Pruning a Llama3.2 model INCORRECT APROACH.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6_2_pruning_structured_llama3.2-1b_KO.ipynb) |
| --- |

The second notebook addresses the issues encountered when applying the same pruning process to the Llama model as was used for distilGPT2.

The correct approach is to treat the MLP layers of the model as pairs rather than individual layers and to calculate neuron importance by considering both layers together. Additionally, we switch to using the maximum absolute weight to decide which neurons remain in the pruned layers.

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

## Knowledge distillation. 
Knowledge Distillation involves training a smaller "student" model to mimic a larger, well-trained "teacher" model. The student learns not just from the correct labels but also from the probability distributions (soft targets) that the teacher model produces, effectively transferring the teacher's learned knowledge into a more compact form.

When combined with Pruning, you first create a pruned version of your base model by removing less important connections. During this process, some knowledge is inevitably lost. To recover this lost knowledge, you can apply Knowledge Distillation using the original base model as the teacher and the pruned model as the student, helping to restore some of the lost performance.

Both techniques address the same challenge: reducing model size and computational requirements while maintaining performance, making them crucial for deploying AI in resource-constrained environments like mobile devices.

### Recovering knowledge from the base model using KD. 
In this notebook, we will use Knowledge distillation to recover some of the knowledge lost during the model pruning process. Llama-3.2-1B will be used as the Teacher model, and the 40% pruned version will be used as the Student model. We will specifically improve the performance on the Lambada benchmark.
| [Notebook: Knowledge Distillation Llama 3.2.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/7_1_knowledge_distillation_Llama.ipynb) |
| --- |

## Bias & Fairness in LLMs. 
This section introduces a preliminary line of work focused on detecting bias in LLMs by visualizing neural activations. While still in its early stages, these analyses pave the way for future fairness-aware pruning strategies, where structural pruning decisions also take into account the impact on different demographic or semantic groups.

### Visualizing Bias in State of the art Transformer Models. 
This notebook introduces techniques for visualizing neural activations in Transformer models, as a first step toward detecting and mitigating bias in language models.
Techniques applied:
  *  Dimensionality reduction with PCA
  *  Visualization using heatmaps
  *  Differential activation analysis between contrastive groups

| Article                                                                 | Notebook                                                                 |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [From Biased to Balanced: Visualizing and Fixing Bias in Transformer Models](https://medium.com/data-science-collective/from-biased-to-balanced-visualizing-and-fixing-bias-in-transformer-models-d1a82f35393c?sk=abd12073ee311c3752da3219a5baf20f) | [8_1_transformer_activations_visualization.ipynb](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/8_1_transformer_activations_visualization.ipynb) |

### Targeted Pruning for Bias Mitigation in Transformer Models
This notebook introduces a novel pruning method designed to mitigate bias in LLMs. By using pairs of contrastive prompts (e.g., "Black man" vs. "white man"), the method identifies neurons that respond differently depending on demographic cues. These neurons are then selectively removed based on a hybrid scoring system that combines bias contribution and structural importance.

The technique is implemented using the [optipfair library](https://github.com/peremartra/optipfair), which provides detailed visualizations of layer-wise activations and internal bias metrics. You can explore the model’s internal behavior interactively via the companion Hugging Face Space: [🌐 Optipfair Bias Analyzer on Hugging Face](https://huggingface.co/spaces/oopere/optipfair-bias-analyzer). 

The results speak for themselves: with just 0.13% of the parameters pruned, the model’s internal bias metric was reduced by 22%, with minimal performance loss. This proof of concept demonstrates that bias-aware pruning can be both precise and efficient—offering a practical tool for building fairer AI systems.
| [Notebook: 8_2_Targeted_Pruning_for_Bias_Mitigation.ipynb](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/8_2_Targeted_Pruning_for_Bias_Mitigation.ipynb) |
| ---------------------------------------------------------|

_____________
<h1>🚀2- Projects.</h1>

## [Natural Language to SQL.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/P1-NL2SQL).
In this straightforward initial project, we are going to develop a SQL generator from natural language. We'll begin by creating the prompt to implement two solutions: one using OpenAI models running on Azure, and the other with an open-source model from Hugging Face.
| Article | Notebook |
| --- | --- |
| [Create a NL2SQL prompt for OpenAI](https://medium.com/towards-artificial-intelligence/create-a-superprompt-for-natural-language-to-sql-conversion-for-openai-9d19f0efe8f4) | [Prompt creation for OpenAI](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_OpenAI.ipynb) |
| WIP | [Prompt creation for defog/SQLCoder](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_SQLCoder.ipynb) |
| [Inference Azure Configuration.](https://pub.towardsai.net/how-to-set-up-an-nl2sql-system-with-azure-openai-studio-2fcfc7b57301) | [Using Azure Inference Point](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/NL2SQL_OpenAI_Azure.ipynb) |

## [Create and publish an LLM.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P2-MHF/readme.md) 
In this small project we will create a new model aligning a microsoft-phi-3-model with DPO and then publish it to Hugging Face. 
| Article | Notebook |
| --- | --- |
| WIP | [Aligning with DPO a phi3-3 model.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P2-MHF/Aligning_DPO_phi3.ipynb)

_____________
<h1>🚀3- Architecting Enterprise Solutions.</h1>

## [Architecting a NL2SQL Solution for immense Enterprise Databases](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/E1-NL2SQL%20for%20big%20Databases).
In this initial solution, we design an architecture for an NL2SQL system capable of operating on a large database. The system is intended to be used with two or three different models. In fact, we use three models in the example. 

It's an architecture that enables a fast project kickoff, providing service for only a few tables in the database, allowing us to add more tables at our pace.

## [Decoding Risk: Transforming Banks with Customer Embeddings.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/E2-Transforming%20Banks%20With%20Embeddings)
In this solution, we explore the transformative power of embeddings and large language models (LLMs) in customer risk assessment and product recommendation in the financial industry. We'll be altering the format in which we store customer information, and consequently, we'll also be changing how this information travels within the systems, achieving important advantages. 

_____________
### Contributing to the course: 
Please, if you find any problems, open an [issue](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/issues) . I will do my best to fix it as soon as possible, and give you credit.  

If you'd like to make a contribution or suggest a topic, please don't hesitate to start a [Discussion](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/discussions). I'd be delighted to receive any opinions or advice.

Don't be shy, share the course on your social networks with your friends. Connect with me on [LinkedIn](https://www.linkedin.com/in/pere-martra/) or [Twitter](https://twitter.com/PereMartra) and feel free to share anything you'd like or ask any questions you may have.

Give a Star ⭐️ to the repository. It helps me a lot, and encourages me to continue adding lessons. It's a nice way to support free Open Source courses like this one. 

_____________
# References & Papers used in the Course: 
* Tom Kocmi, Christian Federmann, [Large Language Models Are State-of-the-Art Evaluators of Translation Quality](https://arxiv.org/abs/2302.14520). Evaluating LLMs with LLMs. 

* Pere Martra, [Introduction to Large Language Models with OpenAI](https://doi.org/10.1007/979-8-8688-0515-8_1)

* [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). LangChain & Agents Section. Medical Assistant Sample.   

* [The Power of Scale for Parameter-Efficient Prompt Tuning](https://doi.org/10.48550/arXiv.2104.08691). Fine Tuning & Optimization Section. Prompt Tuning Sample. 

* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Fine Tuning & Optimization Section. LoRA Fine-Tuning Sample. 

* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). Fine Tuning & Optimization Section. QLoRA Fine-Tuning Sample.

* [How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings]> (https://arxiv.org/abs/2305.11853). Project. Natural Language to SQL. 

* Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, Pavlo Molchanov, "Compact Language Models via Pruning and Knowledge Distillation," arXiv preprint arXiv:2407.14679, 2024. Available at: [https://doi.org/10.48550/arXiv.2407.14679](https://doi.org/10.48550/arXiv.2407.14679).

* He, S., Sun, G., Shen, Z., & Li, A. (2024). What matters in transformers? not all attention is needed. arXiv preprint arXiv:2406.15786. https://doi.org/10.48550/arXiv.2406.15786

* Kim, B. K., Kim, G., Kim, T. H., Castells, T., Choi, S., Shin, J., & Song, H. K. (2024). Shortened llama: A simple depth pruning for large language models. arXiv preprint arXiv:2402.02834, 11. https://doi.org/10.48550/arXiv.2402.02834

```
@software{optipfair2025,
  author = {Pere Martra},
  title = {OptiPFair: A Library for Structured Pruning of Large Language Models},
  year = {2025},
  url = {https://github.com/peremartra/optipfair}
}
```

* Martra, P. (2024, December 26). Exploring GLU Expansion Ratios: Structured Pruning in Llama-3.2 Models. https://doi.org/10.31219/osf.io/qgxea


