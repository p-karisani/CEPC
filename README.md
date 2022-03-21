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
Below you can see an example command to run the code. This command tells the code to use a subset of the documents in the training and the unlabeled sets to train a model and evaluate in the test set—F1 measure is printed at the end of the execution.
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

**Notes**
- The code uses the huggingface pretrained bert model: [Link](https://github.com/huggingface/transformers)
- The batch size is set to 50



