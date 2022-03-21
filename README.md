This is a Python code for the paper below: <br/>
*Multiple-Source Domain Adaptation via Coordinated Domain Encoders and Paired Classifiers*, Payam Karisani. AAAI 2022. [Link](https://arxiv.org/abs/2201.11870)

**Pre-requirements**
- Python (>= 3.7.0)
- Numpy (>= 1.21)
- Pytorch (>= 1.8)

**Input**<br/>
The input file should contain one document per line. Each line should have 4 attributes (tab separated):
1) A unique document id (integer)
2) A binary label (integer):
	- The number 1 for negative documents
	- The number 3 for positive documents
3) Domain (string): a keyword specifying the domain of the document
4) Document body (string)

See the file “sample.data” for a sample input.

**Training and Evaluation**<br/>
Below you can see an example command to run the code. This command tells the code to read the input data, to separate the documents based on their domains, to iteratively assume one domain is the target domain and the rest are the source domains, to train the model, to test the model on the assumed target domain, and to print the result--F1, Precision, Recall, Accuracy.

The code is ran for the specified number of iterations, and the average results are printed at the end of the execution.

```
python -m CEPC.src.MainThread --cmd da_m_mine1 \
--itr 5 \
--model_path /user/desktop/bert-base-uncased/ \
--data_path /user/desktop/data/sample.data \
--output_dir /user/desktop/output \
--device 0 \
--seed 666 
```

The arguments are explained below:
- “--itr”: The number of iterations to run the experiment with different random seeds
- “--model_path”: The path to the huggingface pretrained bert
- “--data_path”: The path to the input data file
- “--output_dir”: A directory to be used for temporary files (the results are printed on screen only)
- “--device”: GPU identifier
- “--seed”: Random seed

**Datasets**
- You can find the Illness dataset here: [Link](https://github.com/p-karisani/illness-dataset)
- The Crisis and Tuning datasets are in the directory "sandoogh". These are meta-datasets and collected from the data published by other researchers. Please make sure to cite the original articles if you intend to use them, see the paper for the references.

**Notes**
- A pre-requisite to the algorithm is a grid search to obtain the hyper-parameters, you cannot set the values manually (see the paper). The grid search may make the code look a bit slow, specifically if your dataset is large. During the development I used a caching module and a few programming tricks to speed up the work. The code here doesn't include the caching module--because it won't make your experiments faster.
- The code uses the huggingface pretrained bert model: [Link](https://github.com/huggingface/transformers)
- The batch size is set to 50



