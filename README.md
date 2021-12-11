# Amazon Topic Modeling
This project analysed the products on Amazon by creating topics based on customers reviews. Each of the reviews will be analysed using the Latent Dirichlet Allocation (**LDA**) algorithm to classify text to a particular topic. The most important (semantic and syntactic) keywords will be used to derive the topic cluster.
 
The dataset is created using **web crawling** in the Electronics category. To process the dataset, the following steps were performed:

Remove punctuation, Hypertext Transfer Protocols, whitespaces.  
Lowercase the words.  
Words that have fewer than 3 characters are removed.  
All stopwords are removed.  
Apply **Contraction**.  
Apply **Stemming**.  
Apply **Lemmatization**. 
Apply **Tokenization**: Split the sentences into words to prepare for LDA.  
Build **Bigrams**: a sequence of two adjacent elements from a string of tokens.  
Convert all the list of words into the **BoW** format.  

# About using TF-IDF before LDA
LDA only needs a bag-of-word vector and TFIDF corpus is not needed for LDA modelling based on the paper of 2003 (entitled "Latent Dirichlet Allocation") from Blei (who developed LDA). The algorithm is a word generating model, which assumes a word is generated from a multinomial distribution. It doesn't make sense to say 0.6 word (tf-idf frequency weight) is generated from some distribution.
