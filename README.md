# Similarity-and-Top-N

# Project Title
Predicted the number of labels in LCSH-subjectheadings. And compute the cosine similarity between labels and input texts.
## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)
6. [Contact](#contact)

## Introduction


This project aims to enhance multi-label classification for LCSH subject headings by predicting the number of relevant labels and computing cosine similarities between labels and input texts.
Our cover 1,000,000 records using a robust multi-stage pipeline. First, data is categorized into 21 classes based on the first letter of Library of Congress Classification (LCC) numbers. Then, different embedding models like SciBERT, BERT, and Transformer are used to compute text representations. Various regression models, including Linear Regression, Random Forest, and XGBoost, are employed to predict the number of labels for each record.
Once the label count is determined, the system computes cosine similarities between the input text and potential labels within the category. 




## Features
- Predicted the number of labels in LCSH-subjectheadings by using our new data(1000,000 records-bibli2).
- We divide all data into 21 categories by the first letter of LCC numbers. 
- Then compute the cosine similarity between all the labels in specific category and their input text(abstract+toc+title).
- Compute the param take rank top 5, top 10, top 20, top 50, and top N.  
- We use different embedding model sci-BERT, BERT, and transformer.
- We use different regression models linear, random forest, XGboost.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/llm4cat/Similarity-and-Top-N.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Similarity-and-Top-N
    ```
3. Create a new virtual environment(Where we run is in Linux with sever and we manage our environment by conda )
   ```bash
      conda create --name bert_env python=3.8
      conda activate bert_env
    ```

4. Install dependencies:
   ```bash
   python --version # make sure you already have install python
   pip install pandas  
   pip install scikit-learn
   pip install torch
   pip install transformers 
   ```

## Usage
 Run the application:
   
  1.Extract the origin data(json file)
   ```bash
   python ext.py
   ```
   
  2.Clean the data
   Get the mega data(Filter the data )
   ```bash
   python clean2.py
   ```
  3.Predicted the number of labels
   ```bash
   python pre-num-new-xxx-xxx.py (depend on which models)
   ```
  4.Compute the cosine similarity
   ```bash
   python sim_topx.py (depend on which model and top)
   ```
  5.Combine them compute TopN (N is decided by step3)
  ```bash
   python sim-topn.py
   ```
  
   
   



## License
This project is licensed under the BSD 3 License. See the `LICENSE` file for details.

## Contact
For questions or issues, please contact:
- **Jinyu Liu** - JinyuLiu@my.unt.edu
- Project Link: [GitHub Repository]https://github.com/llm4cat/Similarity-and-Top-N

---

Thank you for using this project! We appreciate your contributions and feedback.

