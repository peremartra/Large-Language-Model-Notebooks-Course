In this straightforward initial project, we are going to develop a SQL generator from natural language. We'll begin by creating the prompt to implement two solutions: one using OpenAI models running on Azure, and the other with an open-source model from Hugging Face.

<img width="540" alt="image" src="https://github.com/peremartra/Large-Language-Model-Notebooks-Course/assets/7319142/98b926c8-b4d7-4bb7-9ce2-0d87b9d29f72">

## Promp Creation for OpenAI. 
In this initial project step, we'll start with a previously created prompt in a very basic way, allowing us to generate SQL language from user requests. We will modify the prompt to adhere to the best practices published in the paper from the University of Ohio.
* Prompt creation for OpenAI. Article: https://pub.towardsai.net/create-a-superprompt-for-natural-language-to-sql-conversion-for-openai-9d19f0efe8f4?sk=88889b3417c97481e6a907e3aef74ca2
* Prompt creation for OpenAI. Notebook: [https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_enhaced_ohaio.ipynb](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_OpenAI.ipynb)

## Prompt creation for defog/SQLcoder.  
We continue adapting the prompt created for OpenAI to SQLCoder, an open-source model from Hugging Face trained from the super-efficient Mistral 7B.
* Prompt creation for defog/SQLCoder: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_SQLCoder.ipynb

* # Papers used in the project:
* [How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings](https://arxiv.org/abs/2305.11853)
