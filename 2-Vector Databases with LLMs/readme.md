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

A brief introduction to Vector Databases, a technology that will accompany us in many lessons throughout the course. We will work on an example of Retrieval Augmented Generation using information from various news datasets stored in ChromaDB.

### Influencing Language Models with Personalized Information using a Vector Database. 
| Description | Article | Notebook |
| -------- | --- | ---|
| If there's one aspect gaining importance in the world of large language models, it's exploring how to leverage proprietary information with them. In this lesson, we explore a possible solution that involves storing information in a vector database, ChromaDB in our case, and using it to create enriched prompts. | [Article](https://pub.towardsai.net/harness-the-power-of-vector-databases-influencing-language-models-with-personalized-information-ab2f995f09ba?sk=ea2c5286fbff8430e5128b0c3588dbab) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/2_1_Vector_Databases_LLMs.ipynb)| 

### Creating a ChromaDB server. 
| Description | Server | Client |
| ------- | --- | --- |
|In these two notebooks, a ChromaDB server is created from which information is served to a second notebook that contains the client. It is a small example of how to set up a local ChromaDB server that can be used by multiple clients. | [ChromaDB Server](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/2_2-ChromaDB%20Sever%20mode.ipynb) | [ChromaDB Client](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/2_3-ChromaDB%20Client.ipynb) | 

### Semantic Cache for RAG systems ###
| Description | Article | Notebook |
| -------- | --- | ---|
| We enhanced the RAG system by introducing a semantic cache layer capable of determining if a similar question has been asked before. If affirmative, it retrieves information from a cache system created with Faiss instead of accessing the Vector Database. The inspiration and base code of the semantic cache present in this notebook exist thanks to the course: https://maven.com/boring-bot/advanced-llm/1/home from Hamza Farooq.| WIP | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/semantic_cache_chroma_vector_database.ipynb) |
