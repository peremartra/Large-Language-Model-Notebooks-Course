# Large Language Models Course: Learn Building LLM Projects.
I'm currently developing a course on Large Language Models (LLMs) and their applications. This repository will house all the notebooks utilized throughout the course.

The course provides a hands-on experience using models from OpenAI and the Hugging Face library. We are going to see and use a lot of tools and practice with small projects that will grow as we can apply the new knowledge"

Some of the topics and technologies covered in the course include:
 
* Chatbots. 

* Code Generation.

* OpenAI API.

* Hugging Face. 

* Vector databases.

* LangChain.

* Transfer Learning. 

* Knowdledge Distillation.

Each notebook is supported with a Medium article where the code is explained in detail. 

## Create Your First Chatbot Using GPT 3.5, OpenAI, Python and Panel.
We will be utilizing OpenAI GPT-3.5 and Panel to develop a straightforward Chatbot tailored for a fast food restaurant. During the course, we will explore the fundamentals of prompt engineering, including understanding the various OpenAI roles, manipulating temperature settings, and how to avoid Prompt Injections. 
* Article: https://medium.com/towards-artificial-intelligence/create-your-first-chatbot-using-gpt-3-5-openai-python-and-panel-7ec180b9d7f2
* Notebook: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/Vertical%20Chat.ipynb

## How to Create a Natural Language to SQL Translator Using OpenAI API.
Following the same framework utilized in the previous article to create the ChatBot, we made a few modifications to develop a Natural Language to SQL translator. In this case, the Model needs to be provided with the table structures, and adjustments were made to the prompt to ensure smooth functionality and avoid any potential malfunctions.

With these modifications in place, the translator is capable of converting natural language queries into SQL queries. 
* Article: https://pub.towardsai.net/how-to-create-a-natural-language-to-sql-translator-using-openai-api-e1b1f72ac35a
* Notebook: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/nl2sql.ipynb

## Influencing Language Models with Personalized Information using a Vector Database. 
If there's one aspect gaining importance in the world of large language models, it's exploring how to leverage proprietary information with them. In this lesson, we explore a possible solution that involves storing information in a vector database, ChromaDB in our case, and using it to create enriched prompts.
* Article: https://pub.towardsai.net/harness-the-power-of-vector-databases-influencing-language-models-with-personalized-information-ab2f995f09ba
* Notebook: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/how-to-use-a-embedding-database-with-a-llm-from-hf.ipynb

## LangChain
LangChain has been one of the libraries in the universe of large language models that has contributed the most to this revolution. 
It allows us to chain calls to Models and other systems, allowing us to build applications based on large language models. In the course, we will use it several times, creating increasingly complex projects.

### Retrieval Augmented Generation (RAG). Use the Data from your DataFrames with LLMs.
In this lesson, we used LangChain to enhance the notebook from the previous lesson, where we used data from two datasets to create an enriched prompt. This time, with the help of LangChain, we built a pipeline that is responsible for retrieving data from the vector database and passing it to the Language Model. The notebook is set up to work with two different datasets and two different models. One of the models is trained for Text Generation, while the other is trained for Text2Text Generation.
* Article: 
* Notebook: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/ask-your-documents-with-langchain-vectordb-hf.ipynb
_____________
The course will consist of a minimum of 12 articles. To stay updated on new articles, don't forget to follow the repository or starring it. This way, you'll receive notifications whenever new content is added. 
