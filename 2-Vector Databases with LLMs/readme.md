A brief introduction to Vector Databases, a technology that will accompany us in many lessons throughout the course. We will work on an example of Retrieval Augmented Generation using information from various news datasets stored in ChromaDB.

### Influencing Language Models with Personalized Information using a Vector Database. 
| Description | Article | Notebook |
| -------- | --- | ---|
| If there's one aspect gaining importance in the world of large language models, it's exploring how to leverage proprietary information with them. In this lesson, we explore a possible solution that involves storing information in a vector database, ChromaDB in our case, and using it to create enriched prompts. | [Article](https://pub.towardsai.net/harness-the-power-of-vector-databases-influencing-language-models-with-personalized-information-ab2f995f09ba?sk=ea2c5286fbff8430e5128b0c3588dbab) | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/how-to-use-a-embedding-database-with-a-llm-from-hf.ipynb)| 

### Semantic Cache for RAG systems ###
| Description | Article | Notebook |
| -------- | --- | ---|
| We enhanced the RAG system by introducing a semantic cache layer capable of determining if a similar question has been asked before. If affirmative, it retrieves information from a cache system created with Faiss instead of accessing the Vector Database. The inspiration and base code of the semantic cache present in this notebook exist thanks to the course: https://maven.com/boring-bot/advanced-llm/1/home from Hamza Farooq.| WIP | [Notebook](https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/2-Vector%20Databases%20with%20LLMs/semantic_cache_chroma_vector_database.ipynb) |
