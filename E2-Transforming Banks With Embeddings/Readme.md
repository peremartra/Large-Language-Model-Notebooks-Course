# Decoding Risk: Transforming Banks with Customer Embeddings.

If we were tasked with creating a system using LLM to enhance a bank's decision-making regarding risk assessment or product approval for its clients, the initial thought might be to provide the LLM with client and product information, enabling it to make decisions on whether to grant the product to the client or not.

But let me tell you, this wouldn't work. We can't solve everything by providing information to an LLM and expecting it to generate an output based on the input data.

What we're going to set up is a project that will integrate with the bank's other systems, aiming not only to enhance decision-making regarding risk but also to improve maintenance and streamline the overall development of the entity's information systems.

We'll be altering the format in which we store customer information, and consequently, we'll also be changing how this information travels within the systems.

*Before we proceed, **a disclaimer:** All diagrams and descriptions of the solutions presented here are simplified to the maximum extent. This is a project that will likely span years and requires an immense development team. Here, I'm only presenting the idea, with a brief outline of how it should be and the advantages it can bring.*

# Actual Client Risk System. 

Let's start by looking at an example of what a current risk system for any institution might look like.

<img width="368" alt="image" src="https://github.com/peremartra/private_test/assets/7319142/58865dbc-040f-4ef6-be0f-1946c88d21fc">

I'm going to detail the flow of a request:
1. A loan request is initiated from one of the channels.
2. The operation data + customer identifier is passed.
3. The risk application collects customer data.
4. The risk application collects product data.
5. With customer, product, and operation data, a decision is made.
6. The decision is returned to the channel that initiated the request.

The process doesn't seem overly complicated, but let's delve a bit deeper. What are we talking about when we refer to user data?

Is it a snapshot of the moment? In other words: age, gender, account balances, committed balances, available credit cards, relationships with other entities, investments, funds, pension plans, salary, and so forth?

Well, now we have a better idea about the amount of data that must be taken into account, and we are talking about just a snapshot, the current moment. But wouldn't it be better if we made decisions based on how all these variables have been changing over time? This implies considering even more data.

As you can imagine, obtaining all this data involves a very high number of calls to different systems within the bank. I've simplified it by referring to "user data," but that's not a singular entity. Behind that label, there's a myriad of applications, each with its own set of data that we need to request.

In summary, calculating the customer's position is typically done in a batch process that updates every X days, depending on the bank.

The decision of whether to grant a loan is made by an algorithm or a traditional machine learning model that takes specific product data and precalculated risk positions as input.

# How can a Large Language Model (LLM) help us improve this process and, above all, simplify it?

In this project, the key lies not in the deployment of LLMs but rather in crafting embeddings to encapsulate the entirety of customer information. This is a pivotal decision requiring meticulous justification. Let's delve into some advantages of employing embeddings. 

* **Improved Feature Representation**: Can effectively represent complex and unstructured data sources, such as text data from customer interactions, financial reports, and social media profiles. By transforming these raw texts into numerical vectors that capture their underlying semantic meaning, embeddings enable credit risk models to incorporate richer and more informative features, leading to more accurate risk assessments.
* **Similarity Analysis**: Eenable measuring similarity between different entities. This can be useful in identifying similar customer profiles or similar transactions, helping to identify potential risks based on historical patterns.
* **Enhanced Risk Identification**: Can identify hidden patterns and relationships within vast amounts of data, allowing credit risk models to uncover subtle signals that may not be apparent from traditional numerical data. This improved ability to identify and understand risk factors can lead to more precise risk assessments and better-informed lending decisions.
* **Handling Missing Data**: Can handle missing data more gracefully than traditional methods. The model can learn meaningful representations even when certain features are missing, providing more robust credit risk assessments.
* **Reduced Dimensionality**: Exhibit reduced dimensionality compared to the original data representations. This facilitates storage, transmission, and processing of information.
* **Transferability Convenience**: Given that embeddings are numerical representations, they can be easily transferred across various systems and components within the framework. This enables consistency in the utilization of data representations across different segments of the infrastructure.

These are just a few of the advantages that storing customer information in embeddings can provide. Let's say I am fully convinced, and possibly, even if we haven't entirely convinced the leadership, we have certainly piqued their curiosity to inquire about how the project should be approached. In other words, a preliminary outline of the solution.

## First picture of the solution. 
The first step is to select a model and train it with a portion of the data we want to store as embeddings. The model selection and training process are crucial. The entity has sufficient data to train a model from scratch using only its own data, without having to resort to a pre-trained model with financial data, which are not particularly abundant. 

The challenge we face is not the quantity of data but rather the selection of which data to use and determining the cost we are willing to assume for training our model.

From now on, let's refer to this kind of model as FMFD (Foundational Model on Financial Data). And we need to train at least thre of them, one to create the embeddings of the client, the other with the data of the product, and one trained to return the credit default rate. 

The latter model introduces an interesting feature to the project. Classical models are designed to solve a problem by returning a single output. That is, a traditional machine learning model might provide a binary output indicating whether to approve or deny credit, while another could indicate the likelihood of credit default. **In contrast, when using an LLM (Multi-Output Language Model), the output can be multiple, and from a single call, we could obtain various pieces of information**. We might even obtain a list of recommended products, the maximum allowable credit, or an acceptable risk-associated interest rate. It all depends on the data used to train this model.

<img width="681" alt="image" src="https://github.com/peremartra/private_test/assets/7319142/7647c7d5-3de2-4058-ba3c-a5d500ae9112">

Starting from the point where we have already trained the three necessary models, we store the customer embeddings in an embeddings database. This could be a standard database or a vector database. For now, we won't delve into the advantages and disadvantages of using one type of database over another. 

The crucial point is that these embeddings can be continuously calculated as the customer's situation evolves, as it is a lightweight process that can be performed online. Although we also have the option of employing a batch process at regular intervals to handle multiple embeddings simultaneously and optimize resource utilization.

Let's explore what happens when we receive a transaction that needs to be accepted or rejected using our system. With the transaction data, we generate embeddings for the product to be granted, employing the pre-trained model with product data. This yields specific embeddings for the transaction.

To make the decision, we'll need the third model, to which we'll feed both embeddings – one for the customer and one for the transaction. There are several options for combining these embeddings:

* Concatenate the two vectors.
* Add them to a single vector, using, for example, vector summation.
* Pass the data as two independent embeddings.
* Pass the embeddings in a list.

My preference is to pass them using a single concatenated vector. With this vector as input, the third model is capable of generating a response that, depending on the training performed, can provide more or less information.

# Conclusion. 

With this solution, we have a system that utilizes nearly real-time information about the customer's position, along with all the historical data, to analyze whether they can obtain one of our financial products. We've increased the quality and quantity of information used for making decisions regarding our customers' risk operations. 

But that's not all; the use of embeddings also simplifies system maintenance and opens up a world of possibilities that are challenging to detail at this moment. We have a system that adapts better to changes in information because the entire system deals with embeddings of fixed length, as opposed to a multitude of fields that can vary.

The compact size of the embeddings even allows them to reside on the client's device, enabling it to make decisions independently without the need to make a call to the bank's systems.

## Preparatory Steps when initiating the project. 
As one can imagine, we are not discussing a two-month project here. This initiative would redefine how a banking entity stores and processes information about its clients and products. It's a transformation that could span years, impacting a significant portion of its information systems department to varying degrees.

A project of this magnitude necessitates a phased approach, allowing for pivots throughout its lifespan and a gradual introduction.

My recommendation is start  with a proof of concept, utilizing a single model but trained using QLoRA for the three identified tasks: Client Embeddings, Product Embeddings, and Decision Making.

By adopting this approach, we can have three models operational swiftly, with minimal resource requirements for both training and inference. For each of these three models, decisions should be made regarding the data we want/need to use for their training, or in this case, fine-tuning.
__________________________
Check the FinGPT Model, it can be a good option to use as our FMFD. https://arxiv.org/abs/2306.06031

This solution has been inspired by this article [The Shaky Foundations of Foundation Models in Healthcare](https://hai.stanford.edu/news/shaky-foundations-foundation-models-healthcare) at Stanford University 
