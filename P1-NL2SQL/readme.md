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

In this straightforward initial project, we are going to develop a SQL generator from natural language. We'll begin by creating the prompt to implement two solutions: one using OpenAI models running on Azure, and the other with an open-source model from Hugging Face.

<img width="443" alt="image" src="https://github.com/peremartra/Large-Language-Model-Notebooks-Course/assets/7319142/cce799f1-b6b9-4020-bf60-4842436025d1">

## Prompt Creation
We will create two NL2SQL prompts that adhere to the best practices published in the paper from the University of Ohio (see below). Adapting them to the specific needs of OpenAI and SQLCoder Models. First we will create trhe OpenAI Prompt, then continue adapting the prompt created to SQLCoder, an open-source model trained by Defog from a super efficient Code Llama 7B Model, available on from Hugging Face.

Although both prompts are based on the same paper, there are slight differences in the creation process. 

Besides creating the prompt, we conduct a few tests and observe how both models generate SQL commands correctly.
| Article | Notebook |
| --------- | --- |
| [Create a NL2SQL prompt for OpenAI](https://pub.towardsai.net/create-a-superprompt-for-natural-language-to-sql-conversion-for-openai-9d19f0efe8f4?sk=88889b3417c97481e6a907e3aef74ca2) | [Notebook for OpenAI](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_OpenAI.ipynb) | 
| Article WIP | [Notebook for SQLCoder](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_SQLCoder.ipynb) |

## Azure Configuration. 
In Azure, we will configure the Open Services using the prompt created earlier for OpenAI models. This allows us to conduct tests and set up an inference endpoint to call for obtaining SQL commands.
| Article | Notebook | 
| --------- | --- |
| [How To Set up a NL2SQL System With Azure OpenAI Studio.](https://medium.com/towards-artificial-intelligence/how-to-set-up-an-nl2sql-system-with-azure-openai-studio-2fcfc7b57301) | [Using the inference point on Azure.](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/NL2SQL_OpenAI_Azure.ipynb) |

# Papers used in the project:
* [How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings](https://arxiv.org/abs/2305.11853)
