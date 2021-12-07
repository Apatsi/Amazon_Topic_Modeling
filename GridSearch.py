from gensim.models import CoherenceModel, LdaMulticore


# compute coherence value at various values of decay and num_topics
def compute_coherence_values(corpus, dictionary, texts, learning_decay_range, num_topics_range):
    coherence_values = []
    model_list = []
    for num_topics in num_topics_range:
        for learning_decay in learning_decay_range:
            lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                                     random_state=100, decay=learning_decay, passes=10, per_word_topics=True)
            model_list.append(lda_model)

            coherencemodel = CoherenceModel(model=lda_model, texts=texts,dictionary=dictionary, coherence='c_v')
            coherence_values.append((learning_decay, num_topics, coherencemodel.get_coherence(),
                                     lda_model.log_perplexity(corpus))) # a measure of how good the model is. The lower the better.

    return model_list, coherence_values




