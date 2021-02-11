# NLP_Analysis_HarryPotterBooks

*Natural Language Processing Project to analyze Harry Potter Books content.*  

### Table of Contents   
1. Data and Preprocessing    
2. Analysis      
   2.1 Word Embeddings   
   2.2 Analysis of most important terms: TF-IDF-TFIDF  
   2.3 Pointwise Mutual Information (PMI)](#PMI)   
   2.4 Language Generation using HP Books](#language-generation)  
   2.5 Topic Modeling](#topic)    
      2.5.1 LDA Topic Modeling](#LDA)  
      2.5.2 Dynamic Topic Modeling](#dynamic)  
	 2.6 Matrix Factorization](#matrix)  
	 2.7 Clustering](#clustering)  
3. Visualization](#visualization)  
   3.1 Word Embeddings Visualization 
   3.2 TSNE Word2Vec Visualization 
	 3.3 Word2Vec Book1 Book7 Visualization 
	 3.4 PCA Words Embeddings Visualization  
	 3.5 Unigrams TF-IDF-TFIDF Visualization scaled by TF
	 3.6 Bigrams TF-IDF-TFIDF Visualization scaled by TF 
	 3.7 Harry - Ron - Hermione Occurrencies  Visualization  
	 3.8 Topic Modeling Visualization  
	 3.9 Clusering Visualization 
   

###### Description of the content and type of the dataset
For this project I downloaded the 7 Harry Potter Books in .txt format from  http://www.glozman.com/textpages.html webiste. In the end, I had 7 .txt files each containg one Harry Potter book.   

##### Preprocessing 
The first preprocessing step was to read each of the txt files, transform it into a unique string and then split it into sentences (first splitting the documents, and then applying nlp .sents method).
  Then I have used RegEx to remove any number and special character appearing in the text. After that, I took the sentences and I have expanded all the contractions. Once the sentences were expanded I have tokenized the senteces, put them into lower case and retreived only the lemmas form of the tokens. I have also removed punctuation,retrieved only the content words ('NOUN','VERB','PROPN', 'ADJ','ADV') and removed stopwords. This allows to significantly reduce noise and retreive only the informative part of the text.
  I have also created an 'Instances' object which simply contains the grouped tokens for each sentence. 
Lastly, I have created a DataFrame contaning the book number, the tokens and the instances.   
  The initial input were .txt files whereas the output is a DataFrame containing the tokenized sentences and the instances. The length of the final instances is smaller than the number of original sentences as there have been a noise reduction.

| Book | Length Original Text | N^ Original Senteces | N^ Preprocesses Sentences  
| --- | --- | --- | --- | 
| 1 | 442066 | 7542 | 6647 |
| 2 | 489397 | 7931 | 6837 |
| 3 | 612445 | 10792 | 9345 |
| 4 | 1106719 | 17804 | 15609 |
| 5 | 1481713 | 21609 | 18664 |
| 6 | 984950 | 13939 | 12309 |
| 7 | 1132501 | 16926 | 14971 |

###### Description of the research questions  
Investigate the content of Harry Potter Books by analyzing their text. The research aims at giving a general words and topics analysis for the whole saga. In particular it will put a focus on the change in the characters co-occurrences along the books, and at understanding the dynamics of three main topics between the first and the last book. 

 - What are the most similar words to a given based on HP corpus?
 - What are the most frequent words-bigrams in HP?
 - Does the Harry-Ron/Harry-Hermione/Hermione-Ron occurence frequency changes along the books?
 - What is the PMI measure?
 - Are there any relevant distinct topics in HP?
 - What is the dynamic of some topics along the different books?
 
All these research questions are answered relying on advanced NLP Techniques. There are also provided several visualizations for the results found. 
 
 
