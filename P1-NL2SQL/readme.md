In this straightforward initial project, we are going to develop a SQL generator from natural language. We'll begin by creating the prompt to implement two solutions: one using OpenAI models running on Azure, and the other with an open-source model from Hugging Face.

<img width="443" alt="image" src="https://github.com/peremartra/Large-Language-Model-Notebooks-Course/assets/7319142/cce799f1-b6b9-4020-bf60-4842436025d1">


## OpenAI + Azure.
In this initial project step, we'll start with a previously created prompt in a very basic way, allowing us to generate SQL language from user requests. We will modify the prompt to adhere to the best practices published in the paper from the University of Ohio. Adapting it to the specific needs of OpenAI models. 
### Prompt Creation. 
In this section, we will create two prompts, one for OpenAI models and another for an SQLCoder model based on Mistral.

xAlthough both prompts are based on the same paper, there are slight differences in the creation process. 

Besides creating the prompt, we conduct a few tests and observe how both models generate SQL commands correctly.
* Article. [Create a SuperPrompt for Natural Language to SQL Conversion for OpenAI.](https://pub.towardsai.net/create-a-superprompt-for-natural-language-to-sql-conversion-for-openai-9d19f0efe8f4?sk=88889b3417c97481e6a907e3aef74ca2)
* Notebook. [Prompt creation for OpenAI.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_OpenAI.ipynb)

### Azure Configuration. 
In Azure, we will configure the Open Services using the prompt created earlier for OpenAI models. This allows us to conduct tests and set up an inference endpoint to call for obtaining SQL commands.
* Article. [How To Set up a NL2SQL System With Azure OpenAI Studio.](https://medium.com/towards-artificial-intelligence/how-to-set-up-an-nl2sql-system-with-azure-openai-studio-2fcfc7b57301)
* Notebook. [Using the inference point on Azure.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/NL2SQL_OpenAI_Azure.ipynb) 


## defog/SQLcoder on Azure & AWS.  
We continue adapting the prompt created for OpenAI to SQLCoder, an open-source model from Hugging Face trained from the super-efficient Mistral 7B.
* Prompt creation for defog/SQLCoder: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_SQLCoder.ipynb

# Papers used in the project:
* [How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings](https://arxiv.org/abs/2305.11853)
