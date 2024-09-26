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
In this first section of the course, **you will learn to work with the OpenAI API** by creating two small projects. You'll delve into OpenAI's roles and how to provide the necessary instructions to the model through the prompt to make it behave as desired.

The first project is a restaurant chatbot where the model will take customer orders. Building upon this project, we will construct an SQL statement generator. Here, you'll attempt to create a secure prompt that only accepts SQL creation commands and nothing else.

#Introduction to LLMs with OpenAI API. 
### Create Your First Chatbot Using GPT 3.5, OpenAI, Python and Panel.
| Description | Article | Notebook |
| -------- | ---| --- |
| We will be utilizing OpenAI GPT-3.5 and gpt-4o-mini to develop a straightforward Chatbot tailored for a fast food restaurant. During the article, we will explore the fundamentals of prompt engineering, including understanding the various OpenAI roles, manipulating temperature settings, and how to avoid Prompt Injections. |  [article panel](https://medium.com/towards-artificial-intelligence/create-your-first-chatbot-using-gpt-3-5-openai-python-and-panel-7ec180b9d7f2) <br>-<br> [article gradio](https://ai.plainenglish.io/create-a-simple-chatbot-with-openai-and-gradio-202684d18f35?sk=e449515ec7a803ae828418011bbaca52)| [notebook panel](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_1-First_Chatbot_OpenAI.ipynb) <br>-<br> [notebook gradio](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_1-First_Chatbot_OpenAI_Gradio.ipynb)|

### How to Create a Natural Language to SQL Translator Using OpenAI API. 
| Description | Article | Notebook |
| -------- | ---| --- |
| Following the same framework utilized in the previous article to create the ChatBot, we made a few modifications to develop a Natural Language to SQL translator. In this case, the Model needs to be provided with the table structures, and adjustments were made to the prompt to ensure smooth functionality and avoid any potential malfunctions. With these modifications in place, the translator is capable of converting natural language queries into SQL queries. | [article](https://pub.towardsai.net/how-to-create-a-natural-language-to-sql-translator-using-openai-api-e1b1f72ac35a) <br>-<br> [article gradio](https://medium.com/towards-artificial-intelligence/first-nl2sql-chat-with-openai-and-gradio-b1de0d6541b4)| [notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_2-Easy_NL2SQL.ipynb) <br>-<br> [notebok Gradio](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_2-Easy_NL2SQL_Gradio.ipynb)|

### Brief Introduction to Prompt Engineering with OpenAI. 
| Description | Article | Notebook |
| -------- | ---| --- |
|We will explore prompt engineering techniques to improve the results we obtain from Models. Like how to format the answer and obtain a structured response using Few Shot Samples.| [Article](https://medium.com/gitconnected/influencing-a-large-language-model-response-with-in-context-learning-b212f0eaa113) | [notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/1-Introduction%20to%20LLMs%20with%20OpenAI/1_3-Intro_Prompt_Engineering.ipynb)
