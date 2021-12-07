if __name__ == "__main__":

    # Importing the Libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import sys
    # Statistical tests tools
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    import re
    import tensorflow as tf
    from scipy.stats.mstats import normaltest  # D'Agostino K^2 Test
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    import gensim  # gensim for LDA
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel, LdaModel, LdaMulticore, TfidfModel
    # Visualization tools
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    from nltk.stem import WordNetLemmatizer
    from gensim.test.utils import datapath
    import pyLDAvis
    import pyLDAvis.sklearn
    import pyLDAvis.gensim_models  # dont skip this
    import IPython
    import Pyro4
    import contractions as ct
    from GridSearch import compute_coherence_values
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    print(type(dir_path))

    temp_file = datapath("path")
    print(temp_file)

    warnings.filterwarnings('ignore', category=DeprecationWarning)

    print(__name__)  # __name__ = __main__
    nltk.download('wordnet')
    nltk.download('stopwords')

    print("Training on GPU...") if tf.test.is_gpu_available() else print("Training on CPU...")

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    np.random.seed(42)

    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', None)

    df = pd.read_json('D:/Akisp/Downloads/dataset/Electronics_final.json', lines=True)
    print("Shape of data: {}".format(df.shape))
    print(df.head(20))

    # df = pd.read_csv('D:/Akisp/Downloads/dataset/Amazon Reviews.csv', encoding='latin-1')
    # print("Shape of data:", df.shape, "\n")
    # print(df.head())

    # The variable named "reviewText" is the object of our analysis
    df_check_dupl = df.duplicated(subset=['reviewText'], keep=False)
    df = df.drop_duplicates(subset=['reviewText'], keep='first')
    print("Shape of data after removing duplicates:\n{}\n".format(df.shape))

    print("The variables of the Amazon dataset are:\n{}\n".format(df.columns))

    print("{}\n".format(df.info()))
    print("Checking null values in the reviews BEFORE clearing\n{}\n".format(df.isnull().sum()))

    df = df.dropna(subset=['reviewText'])  # 1 null value found and dropped
    print("Checking null values in the reviews AFTER clearing\n{}\n".format(df.isnull().sum()))

    df['default_length_text'] = df['reviewText'].str.len()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title("Distribution of number of words before preprocessing", fontsize=18)
    ax.set_xlabel("Number of words")
    sns.distplot(df['default_length_text'], color="b", bins=50, ax=ax)
    plt.show()

    # Clearing, Stemming, Lemmatisation
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    lemm = WordNetLemmatizer()

    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    def Contractions(text):
        tokens = []
        for token in text.split():
            tokens.append(ct.fix(token, slang=True))
        return " ".join(tokens)

    def Stemming(text, stem=False):
        text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words and len(token) > 2:  # Number of characters > 2
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)

    def Lemmatization(text):
        tokens = []
        for token in text.split():
            tokens.append(lemm.lemmatize(token))
        return " ".join(tokens)

    # Applying Stemming, Lemmatization and Contractions fixes
    df["reviewText"] = df["reviewText"].apply(lambda x: Contractions(x))
    print("\nDataset after fixing Contractions\n{}\n".format(df["reviewText"]))
    df["reviewText"] = df["reviewText"].apply(lambda x: Stemming(x))
    df["reviewText"] = df["reviewText"].apply(lambda x: Lemmatization(x))
    print("\nDataset after Stemming, Lemmatization and fixing Contractions\n{}\n".format(df["reviewText"]))

    # Tokenization to prepare for Bigrams
    def Tokenization(text):
        tokens = []
        for token in text.split():
            tokens.append(token)
        return tokens

    df["reviewText"] = df["reviewText"].apply(lambda x: Tokenization(x))

    # Build a sequence of two adjacent elements from a string of tokens
    bigram = gensim.models.Phrases(df["reviewText"], min_count=8, threshold=100)  # higher threshold fewer phrases.
    # Faster way to build a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    print("Bigram: {}\n".format(bigram_mod))


    def Bigrams(texts):
        bigrams_tokens = bigram_mod[texts]
        return bigrams_tokens


    df["reviewText"] = df["reviewText"].apply(lambda x: Bigrams(x))
    print("\nAfter bigrams processing:\n{}\n".format(df["reviewText"]))

    # Debugging the reviews by joining the tokens to complete strings
    df["for_text_metrics"] = df["reviewText"].str.join(' ')


    def text_metrics(reviews):
        review_lengths = reviews.str.len()
        print("The average number of words in a review is: {}.".format(np.mean(review_lengths)))
        print("The minimum number of words in a review is: {}.".format(min(review_lengths)))  # To index 461 einai adeio
        print("The maximum number of words in a review is: {}.".format(max(review_lengths)))
        print("The type of the object review lenghts is: {}".format(type(review_lengths)))  # Pandas series
        threshold_length = 40
        print("There are {} documents with over {} words.".format(sum(review_lengths > threshold_length),
                                                                  threshold_length))
        small_reviews = df.loc[review_lengths <= threshold_length, "reviewText"]
        # small_reviews = review_lengths[review_lengths <= threshold_length]
        print("The length <= 40 of each review:\n{}\n".format(small_reviews))
        return review_lengths, small_reviews


    # print("\nAfter preprocessing there are still null values but in different form\n", df.isnull().sum())

    # Finding and dropping reviews that have only 0 or 1 word
    one_word_search = df.loc[np.array(list(map(len, df["reviewText"]))) <= 1, "reviewText"]
    print("Reviews with only 1 word or empty strings: \n{}\n".format(one_word_search))
    df = df.drop(one_word_search.index)
    df = df.reset_index(drop=True)

    review_lengths, small_reviews = text_metrics(df["reviewText"])
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title("Distribution of number of words after preprocessing", fontsize=18)
    ax.set_xlabel("Number of words")
    sns.distplot(review_lengths, bins=50, ax=ax)
    plt.show()

    # create dictionary with bag-of-words (BoW) format
    id2word = corpora.Dictionary(df["reviewText"])
    print("id2word_old: {}".format(id2word))
    print("Number of unique words in initital documents: {}\n".format(len(id2word)))

    # ignore words that appear in less than 5 documents (misspells, unimportant words) or more than 90% documents
    id2word.filter_extremes(no_below=4, no_above=0.9)
    id2word.compactify()
    print("id2word_NEW: {}".format(id2word))
    print("Number of unique words after removing rare and common words: {}\n".format(len(id2word)))
    review_lengths = text_metrics(df["reviewText"])

    # create corpus
    corpus = [id2word.doc2bow(text) for text in df["reviewText"]]

    # sample
    print("The id2word for 3rd document is:", corpus[2])

    # About using TFIDF before LDA
    # LDA only needs a bag-of-word vector and TFIDF corpus is not needed for LDA modelling based
    # on the paper of 2003 (entitled "Latent Dirichlet Allocation") from Blei (who developed LDA).
    # LDA is a word generating model, which assumes a word is generated from a multinomial distribution.
    # It doesn't make sense to say 0.5 word(tf-idf weight) is generated from some distribution.
    # TF-ID implementation only for testing but not actual use!
    # tfidf = TfidfModel(corpus=corpus, id2word=id2word, smartirs='ntc')
    # corpus_tfidf = tfidf[corpus]
    # print("\nTFID:", tfidf)

    # human-readable format of corpus (term-frequency)
    for cp in corpus[2:3]:  # Example for the 3rd review text
        print("\n# Example for the 3rd review text")
        for id, freq in cp:
            print((id2word[id], freq))

    # Once again, getting rid of everything that is "falsy".
    # Finding and dropping reviews with empty lists or 1 word after filtering the extreme words
    corpus = [cp for cp in corpus if len(cp) > 1]

    print("hello world")

    # for review in corpus:  # words that occur more frequently across the documents get smaller weights
    #     print([[id2word[id], freq] for id, freq in review])

    # build LDA models across a range of number of topics and learning decay
    # LDA is a three-level hierarchical Bayesian model, in which each item of a collection
    # is modeled as a finite mixture over an underlying set of topics.
    num_topics_range = [8]
    learning_decay_range = [0.50]
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=df["reviewText"],
                                                            learning_decay_range=learning_decay_range,
                                                            num_topics_range=num_topics_range)

    coherence_df = pd.DataFrame(coherence_values, columns=['learning_decay', 'num_topics', 'coherence_value',
                                                           'perplexity_value'])
    print("\nGrid Search results:\n{}\n".format(coherence_df))

    best_coh_value = coherence_df.loc[coherence_df["coherence_value"] == coherence_df["coherence_value"].max(), :]
    print("The best parameters from Grid Search are:\n{}\n".format(best_coh_value))

    best_lda_model = LdaMulticore(corpus=corpus, id2word=id2word, decay=best_coh_value["learning_decay"].iloc[0],
                                  num_topics=best_coh_value["num_topics"].iloc[0], random_state=42,
                                  passes=10, per_word_topics=True)

    print("The best LDA model after hyperparameter tuning is: {}\n".format(best_lda_model))

    # save model to disk (no need to use pickle module)
    best_lda_model.save("LDA.model")

    # Load a potentially pretrained model from disk.
    # lda = best_lda_model.load("LDA.model")

    topics = best_lda_model.print_topics()
    for topic in topics:
        print("topic: {}".format(topic))

    best_coherence_model_lda = CoherenceModel(model=best_lda_model, texts=df["reviewText"], dictionary=id2word,
                                              coherence='c_v')
    best_coherence_lda = best_coherence_model_lda.get_coherence()
    print('\nTotal Coherence Score: {}\n'.format(best_coherence_lda))
    for i, score_pertopic in enumerate(best_coherence_model_lda.get_coherence_per_topic()):
        print('The score of topic {}: is: {}'.format(i, score_pertopic))

    # Grid Search plotting
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=coherence_df, x="num_topics", y="coherence_value", hue="learning_decay", marker='o')
    plt.show()

    # Plotting the mean + error
    sns.pointplot(x="num_topics", y="coherence_value", hue="learning_decay",
                  data=coherence_df, dodge=True, join=False)

    # # # Would need a good way to show three error bars
    # plt.errorbar(x=results['param_n_components'],
    #              y=results.mean_test_score,
    #              yerr=results.std_test_score,
    #              fmt='none')

    # Visualize the topics
    # pyLDAvis.enable_notebook()  # This is only for Jupyter Notebook
    vis = pyLDAvis.gensim_models.prepare(best_lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')
    print(vis)

    test_doc = np.array(["I bought these on Prime Day because I needed some less expensive headphones to use while mowing the lawn. "
                         "Right out of the box, it took only seconds to connect to my iPhone. "
                         "The sound quality wasn't quite what you would get with an expensive set, but I didn't expect that. Still, the sound quality was great and they were powerful enough that I was concerned that my very loud lawnmower wasn't working correctly. "
                         "Luckily the mower was fine and the headphones were just that good! I also liked the fact that it came with a case, charging cable and aux cable. This product exceeded my expectations for sure!"])

    df_test = pd.DataFrame(data=test_doc, columns=["TEST"])
    df_test["TEST"] = df_test["TEST"].apply(lambda x: Contractions(x))
    df_test["TEST"] = df_test["TEST"].apply(lambda x: Stemming(x))
    df_test["TEST"] = df_test["TEST"].apply(lambda x: Lemmatization(x))
    df_test["TEST"] = df_test["TEST"].apply(lambda x: Tokenization(x))
    df_test["TEST"] = df_test["TEST"].apply(lambda x: Bigrams(x))
    print(df_test)
    bow_test_doc = id2word.doc2bow(df_test["TEST"])
    print(best_lda_model.get_document_topics(bow_test_doc))
