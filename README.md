# Large Language Models Course: Learn by Doing LLM Projects.
**This practical free hands on course about Large Language models and their applications is 👷🏼still in development👷🏼. I will be posting the different lessons as I complete them.**

The course provides a hands-on experience using models from OpenAI and the Hugging Face library. We are going to see and use a lot of tools and practice with small projects that will grow as we can apply the new knowledge acquired. 

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
| [Article](https://medium.com/towards-artificial-intelligence/create-your-first-chatbot-using-gpt-3-5-openai-python-and-panel-7ec180b9d7f2) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/Vertical%20Chat.ipynb) |
| --- | --- |

### How to Create a Natural Language to SQL Translator Using OpenAI API.
Following the same framework utilized in the previous article to create the ChatBot, we made a few modifications to develop a Natural Language to SQL translator. In this case, the Model needs to be provided with the table structures, and adjustments were made to the prompt to ensure smooth functionality and avoid any potential malfunctions. With these modifications in place, the translator is capable of converting natural language queries into SQL queries.
| [Article](https://pub.towardsai.net/how-to-create-a-natural-language-to-sql-translator-using-openai-api-e1b1f72ac35a) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/nl2sql.ipynb)
| --- | --- |

### Brief Introduction to Prompt Engineering with OpenAI.
We will explore prompt engineering techniques to improve the results we obtain from Models. Like how to format the answer and obtain a structured response using Few Shot Samples. 
| WIP | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/Prompt_Engineering_OpenAI.ipynb)
| --- | --- |

## [Vector Databases with LLMs.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/2-Vector%20Databases%20with%20LLMs) 
A brief introduction to Vector Databases, a technology that will accompany us in many lessons throughout the course. We will work on an example of Retrieval Augmented Generation using information from various news datasets stored in ChromaDB.

### Influencing Language Models with Personalized Information using a Vector Database. 
If there's one aspect gaining importance in the world of large language models, it's exploring how to leverage proprietary information with them. In this lesson, we explore a possible solution that involves storing information in a vector database, ChromaDB in our case, and using it to create enriched prompts.
|[Article](https://pub.towardsai.net/harness-the-power-of-vector-databases-influencing-language-models-with-personalized-information-ab2f995f09ba?sk=ea2c5286fbff8430e5128b0c3588dbab) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/how-to-use-a-embedding-database-with-a-llm-from-hf.ipynb) |
| --- | --- |

## [LangChain.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/3-LangChain)
LangChain has been one of the libraries in the universe of large language models that has contributed the most to this revolution. 
It allows us to chain calls to Models and other systems, allowing us to build applications based on large language models. In the course, we will use it several times, creating increasingly complex projects.

### Retrieval Augmented Generation (RAG). Use the Data from your DataFrames with LLMs.
In this lesson, we used LangChain to enhance the notebook from the previous lesson, where we used data from two datasets to create an enriched prompt. This time, with the help of LangChain, we built a pipeline that is responsible for retrieving data from the vector database and passing it to the Language Model. The notebook is set up to work with two different datasets and two different models. One of the models is trained for Text Generation, while the other is trained for Text2Text Generation.
| [Article](https://medium.com/towards-artificial-intelligence/query-your-dataframes-with-powerful-large-language-models-using-langchain-abe25782def5) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/ask-your-documents-with-langchain-vectordb-hf.ipynb) |
| --- | --- |

### Create a Moderation system using LangChain. 
We will create a comment response system using a two-model pipeline built with LangChain. In this setup, the second model will be responsible for moderating the responses generated by the first model.

One effective way to prevent our system from generating unwanted responses is by using  a second model that has no direct interaction with users to handle response generation. 

This approach can reduce the risk of undesired responses generated by the first model in response to the user's entry. 


I will create separate notebooks for this task. One will involve models from OpenAI, and the others will utilize open-source models provided by Hugging Face. The results obtained in the three notebooks are very different. The system works much better with the OpenAI, and LLAMA2 models. 
| Article | Notebook |
| --- | --- |
|[OpenAI article](https://pub.towardsai.net/create-a-self-moderated-commentary-system-with-langchain-and-openai-406a51ce0c8d?sk=b4903b827e44642f7f7c311cebaef57f) | [OpenAI notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/LangChain_OpenAI_Moderation_System.ipynb) |
|[GPT-J Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/HF_gpt_j_6b_Moderation_System.ipynb) | No Article |
|[Llama2-7B Article](https://levelup.gitconnected.com/create-a-self-moderated-comment-system-with-llama-2-and-langchain-656f482a48be?sk=701ead7afb80e015ea4345943a1aeb1d) | [Llama2-7B Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/HF_LLAMA2_LangChain_Moderation_System.ipynb) |

### Create a Data Analyst Assistant using a LLM Agent. 
Agents are one of the most powerful tools in the world of Large Language Models. The agent is capable of interpreting the user's request and using the tools and libraries at its disposal until it achieves the expected result.

With LangChain Agents, we are going to create in just a few lines one of the simplest yet incredibly powerful agents. The agent will act as a Data Analyst Assistant and help us in analyzing data contained in any Excel file. It will be able to identify trends, use models, make forecasts. In summary, we are going to create a simple agent that we can use in our daily work to analyze our data.

| [Article](https://pub.towardsai.net/create-your-own-data-analyst-assistant-with-langchain-agents-722f1cdcdd7e) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/LangChain_Agent_create_Data_Scientist_Assistant.ipynb) |
| --- | --- |

### Create a Medical ChatBot with LangChain and ChromaDB. 
In this example, two technologies seen previously are combined: agents and vector databases. Medical information is stored in ChromaDB, and a LangChain Agent is created, which will fetch it only when necessary to create an enriched prompt that will be sent to the model to answer the user's question.

In other words, a RAG system is created to assist a Medical ChatBot.

**Attention!!! Use it only as an example. Nobody should take the boot's recommendations as those of a real doctor. I disclaim all responsibility for the use that may be given to the ChatBot. I have built it only as an example of different technologies.**
| [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/3-LangChain/langchain_retrieval_agent.ipynb) | Article WIP |
| --- | --- |

## [Evaluating LLMs.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/4-Evaluating%20LLMs)
The metrics used to measure the performance of Large Language Models are quite different from the ones we've been using in more traditional models. We're shifting away from metrics like Accuracy, F1 score, or recall, and moving towards metrics like BLEU, ROUGE, or METEOR. 

These metrics are tailored to the specific task assigned to the language model. 

In this section, we'll explore examples of several of these metrics and how to use them to determine whether one model is superior to another for a given task. We'll delve into practical scenarios where these metrics help us make informed decisions about the performance of different models.

### Evaluating Summarization with ROUGE. 
We will explore the usage of the ROUGE metric to measure the quality of summaries generated by a language model. 
We are going to use two T5 models, one of them being the t5-Base model and the other a t5-base fine-tuned specifically designed for creating summaries.

| [Article](https://medium.com/towards-artificial-intelligence/rouge-metrics-evaluating-summaries-in-large-language-models-d200ee7ca0e6) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/rouge-evaluation-untrained-vs-trained-llm.ipynb) |
| --- | --- |

### Monitoring, Testing and Evaluating LLMs with LangSmith. 
In this initial example, you can observe how to use LangSmith to monitor the traffic between the various components that make up the Agent. The agent is a RAG system that utilizes a vectorial database to construct an enriched prompt and pass it to the model. LangSmith captures both the use of the Agent's tools and the decisions made by the model, providing information at all times about the sent/received data, consumed tokens, the duration of the query, and all of this in a truly user-friendly environment.
| Article WIP | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/4-Evaluating%20LLMs/langsmith_Medical_Assistant_Agent.ipynb) |
| ------ | ------ |

## [Fine Tuning & Optimization.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/5-Fine%20Tuning) 
In the FineTuning & Optimization section, we will explore different techniques such as Prompt Fine Tuning or LoRA, and we will use the Hugging Face PEFT library to efficiently fine-tune Large Language Models. We will explore techniques like quantization to reduce the weight of the Models. 

### Prompt tuning using PEFT Library from Hugging Face. 
In this notebook, two models are trained using Prompt Tuning from the PEFT library. This technique not only allows us to train by modifying the weights of very few parameters but also enables us to have different specialized models loaded in memory that use the same foundational model.

Prompt tuning is an additive technique, and the weights of the pre-trained model are not modified. The weights that we modify in this case are those of virtual tokens that we add to the prompt.
| [Article](https://medium.com/towards-artificial-intelligence/fine-tuning-models-using-prompt-tuning-with-hugging-faces-peft-library-998ae361ee27) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/Prompt_Tuning_PEFT.ipynb) |
| --- | --- |

### Fine-Tuning with LoRA using PEFT from Hugging Face. 
After a brief explanation of how the fine-tuning technique LoRA works, we will fine-tune a model from the Bloom family to teach it to construct prompts that can be used to instruct large language models.
|[Article](https://levelup.gitconnected.com/efficient-fine-tuning-with-lora-optimal-training-for-large-language-models-266b63c973ca?sk=85d7b5d78e64e568faedfe07a35f81bd) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb)
| --- | --- |

### Fine-Tuning a 7B Model in a single 16GB GPU using QLoRA.
We are going to see a brief introduction to quantization, used to reduce the size of big Large Language Models. With quantization, you can load big models reducing the memory resources needed. It also applies to the fine-tuning process, you can fine-tune the model in a single GPU without consuming all the resources. 
After the brief explanation we see an example about how is possible to fine-tune a Bloom 7B Model ina a t4 16GB GPU on Google Colab. 
| [Article](https://medium.com/towards-artificial-intelligence/qlora-training-a-large-language-model-on-a-16gb-gpu-00ea965667c1) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/QLoRA_Tuning_PEFT.ipynb) |
| --- | --- |

_____________
<h1>🚀2- Projects.</h1>

## [Natural Language to SQL](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main/P1-NL2SQL).
In this straightforward initial project, we are going to develop a SQL generator from natural language. We'll begin by creating the prompt to implement two solutions: one using OpenAI models running on Azure, and the other with an open-source model from Hugging Face.
| Article | Notebook |
| --- | --- |
| WIP | [Prompt creation for OpenAI](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_OpenAI.ipynb) |
| WIP | [Prompt creation for defog/SQLCoder](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_SQLCoder.ipynb) |
| [Inference Azure Configuration.](https://pub.towardsai.net/how-to-set-up-an-nl2sql-system-with-azure-openai-studio-2fcfc7b57301) | [Using Azure Inference Point](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/NL2SQL_OpenAI_Azure.ipynb) |

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
# Papers used in the Course: 
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). LangChain & Agents Section. Medical Assistant Sample.   

[The Power of Scale for Parameter-Efficient Prompt Tuning](https://doi.org/10.48550/arXiv.2104.08691). Fine Tuning & Optimization Section. Prompt Tuning Sample. 

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Fine Tuning & Optimization Section. LoRA Fine-Tuning Sample. 

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). Fine Tuning & Optimization Section. QLoRA Fine-Tuning Sample.

[How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings](https://arxiv.org/abs/2305.11853). Project. Natural Language to SQL. 

