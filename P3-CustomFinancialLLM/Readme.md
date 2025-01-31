In this project, we are going to optimize a financial model specialized in sentiment detection for financial reports.

This is a clear example of a typical task one might encounter in a company. They have an LLM that works well but want to reduce its resource consumption while maintaining its performance. The reasons for this can vary—perhaps they simply want to lower usage costs, or maybe they need to deploy it on smaller devices, such as mobile phones.

In this case, we will start with the FinGPT model and create a version that can perform the same tasks with the same efficiency while using fewer resources.

**Steps to follow:**
* Take the base model and measure its performance. We need to determine which benchmarks will provide the insights necessary to confirm that the optimized model consumes fewer resources while maintaining performance.
* Decide which types of pruning can be applied:
  * Which part of the model? Embeddings / Attention / MLP.
  * Which type of pruning? Width / Depth.
* Apply pruning.
* Recover any lost performance using Knowledge Distillation.

This is a fairly standard process in the optimization of customized models. We go beyond fine-tuning by modifying the model’s structure to better fit our needs while preserving the knowledge we want to retain.

________________________

