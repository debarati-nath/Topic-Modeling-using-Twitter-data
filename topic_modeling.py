# LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tweets,
                                           id2word=id2word,
                                           num_topics=6,
                                           random_state=150,
                                           update_every=1,
                                           chunksize=1000,
                                           passes=5,
                                           alpha='auto',
                                           iterations=2000,
                                           per_word_topics=True)

# Print the Keyword in the 20 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus_tweets]

# Compute Perplexity
# a measure of how good the model is. lower the better
base_perplexity = lda_model.log_perplexity(corpus_tweets)
print('\nPerplexity: ', base_perplexity)

# Compute Coherence Score
coherence_model = CoherenceModel(model=lda_model, texts=tweets_lemmatized,
                                   dictionary=id2word, coherence='c_v')
coherence_lda_model_base = coherence_model.get_coherence()
print('\nCoherence Score: ', coherence_lda_model_base)

#LdaMulticore model
model_5_2 = LdaMulticore(corpus=corpus_tweets,
                       id2word=id2word,
                       num_topics=20,
                       random_state=150,
                       chunksize=2000,
                       passes=10,
                       decay=0.5,
                       alpha=0.01,
                       iterations=2000)

pprint(model_5_2.print_topics())
doc_lda = model_5_2[corpus_tweets]

# Compute Perplexity
# a measure of how good the model is. lower the better
base_perplexity = model_5_2.log_perplexity(corpus_tweets)
print('\nPerplexity: ', base_perplexity)

# Compute Coherence Score
coherence_model = CoherenceModel(model=model_5_2, texts=tweets_lemmatized,
                                   dictionary=id2word, coherence='c_v')
coherence_lda_model_base = coherence_model.get_coherence()
print('\nCoherence Score: ', coherence_lda_model_base)

#Defining a function to loop over number of topics to be used to find an optimal number of tipics
#Compute c_v coherence for various number of topics
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values_topic = []
    model_list_topic = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list_topic.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values_topic.append(coherencemodel.get_coherence())
    return model_list_topic, coherence_values_topic

model_list_topic, coherence_values_topic = compute_coherence_values(dictionary=id2word,
                                                        corpus=corpus_tweets,
                                                        texts=tweets_lemmatized,
                                                        start=2, limit=200, step=6)