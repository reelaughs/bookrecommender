# bookrecommender
Group project for Graph Analysis, Learning and Application module, NTU MSc Information Systems

Our task was that we had to design, implement, and evaluate a recommender system; and the algorithm requirements were that we had to utilize Model-based Collaborative Filtering (CF) techniques or Graph Learning techniques.

We were expected to produce source code as well as a technical report including a proper introduction section, literature review, task definition, methodology, setup, results, analysis and conclusion.  

We eventually decided to implement an existing hybrid recommender system which combines content-based and collaborative filtering techniques, with Singular Value Decomposition (SVD) integration and also in conjunction with natural language processing (NLP) models such as Bidirectional Encoder Representations from Transformers (BERT); to provide personalised and relevant recommendations to users based on their preference. We also sought to evaluate model performance by metrics including precision and recall. 

Setup/System requirements: 
The recommender system was developed in Python, utilizing libraries such as _pandas, numpy, sklearn, surprise, torch, transformers_, and _tqdm. _

The libraries _pandas_ and _numpy_ were imported for data handling and manipulation; the _random_ library for sampling as well as the library _sklearn_ for similarity calculations and splitting the dataset into training and testing datasets via _cosine_similarity_ and _train_test_split. _

The _BERT_ algorithm demands substantial computational resources because of its intricate architecture. To address this, the _torch_ and _transformers_ libraries were imported. In particular, _torch.device_ is used to check for GPU availability and allocate the model for processing which decreases time needed.
 
The _surprise_ library prepares data for recommender systems by providing tools to load, split, and manage datasets efficiently. In order to prepare the data for the _Surprise_ library, a reader was set for the data ranging from a 1-10 rating scale before splitting the dataset into training and test datasets. The _tqdm_ library was imported to track progress. 


The model was designed with a train-test split of 80-20. The trained model was expected to generate top 5 personalised recommendations for a sample size of 100 users. For evaluation, different similarity thresholds were applied, specifically 0.70, 0.75, 0.80, 0.85, and 0.90. At each threshold,  precision, recall, and F1-score were also calculated. 



