# Amazon Topic Modeling
This project analysed the products on Amazon by creating topics based on customers reviews. Each of the reviews will be analysed using the Latent Dirichlet Allocation (**LDA**) algorithm to classify text to a particular topic. The most important (semantic and syntactic) keywords will be used to derive the topic cluster.
 
The dataset is created using **Web Scrapping** from the utils.py in the Electronics category. The following preprocessing procedure steps were performed:

“*” Remove punctuation, Hypertext Transfer Protocols, whitespaces.  
“*” Lowercase the words.  
“*” Words that have fewer than 3 characters are removed.  
“*” Remove stopwords.  
“*” Apply **Contraction**.  
“*” Apply **Stemming**.  
“*” Apply **Lemmatization**.  
“*” Apply **Tokenization**: Split the sentences into words to prepare for LDA.  
“*” Build **Bigrams**: a sequence of two adjacent elements from a string of tokens.  
“*” Convert all the lists of words into the **BoW** format.  

# About using TF-IDF before LDA
LDA only needs a Bag-of-Word vector and TF-IDF corpus is not needed for LDA modelling based on the paper of 2003 (entitled "Latent Dirichlet Allocation") from Blei (who developed LDA). The algorithm is a word probabilistic generative model, which assumes a word is generated from a multinomial distribution. It doesn't make sense to say 0.6 word (tf-idf frequency weight) is generated from some distribution.
