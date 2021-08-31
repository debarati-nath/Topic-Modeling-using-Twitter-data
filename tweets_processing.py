# Load the tweets from google drive
tweets_raw = pd.read_csv("tweets_raw.csv")
# Display the first five rows from csv file
display(tweets_raw.head(5))
# Print the summary statistics
print(tweets_raw.describe())
# Print the info
print(tweets_raw.info())

def process_tweets(tweet):
    # Remove links
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # Remove mentions and hashtags
    tweet = re.sub(r'\@\w+|\#', '', tweet)

    # Tokenize the words and remove stopwords - different process
    # tokenized = word_tokenize(tweet)
    # Remove the stop words
    # tokenized = [token for token in tokenized if token not in stopwords.words('english')]
    # Lemmatize the words for NLTK
    # lemmatizer = WordNetLemmatizer()
    # tokenized = [lemmatizer.lemmatize(token, pos='a') for token in tokenized]
    # Remove non-alphabetic characters and keep the words contains three or more letters
    # tokenized = [token for token in tokenized if token.isalpha() and len(token)>2]

    return tweet     #return tokenized if tokenizer is used

# Call the function - process_tweets and store the result into a new column named "Processed"
tweets_raw["Processed"] = tweets_raw["Content"].str.lower().apply(process_tweets)

# Print the first fifteen rows of Processed
display(tweets_raw[["Processed"]].head(15))

# Convert to list
tweets = tweets_raw["Processed"].values.tolist()
print(tweets[:3])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc = True removes punctuations

tweets_words = list(sent_to_words(tweets))

print(tweets_words[:1])

# Build the bigram and trigram models if needed
bigram = gensim.models.Phrases(tweets_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[tweets_words], threshold=100)

# Way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
# display trigram example
print(trigram_mod[bigram_mod[tweets_words[2]]])

stopwords = set(STOPWORDS)

# Custom stopwords
custom_stopwords = ['hi','\n','\n\n', '&amp;', ' ', '.', '-', 'got', "it's", 'it’s', "i'm", 'i’m', 'im', 'want', 'like', '$', '@']

# Customize stop words by adding to the default list

nlp = English()
STOP_WORDS = nlp.Defaults.stop_words.union(custom_stopwords)
# ALL_STOP_WORDS = spacy + gensim + wordcloud
ALL_STOP_WORDS = STOP_WORDS.union(SW).union(stopwords)

# Define functions for stopwords, bigrams, trigrams and lemmatization - another process to remove stopwords and do lemmitization
def remove_stopwords(texts):
  # doc_list = []
  # for doc in texts:
  #   print('This are docs',doc)
  #   word_list = []
  #   for word in texts:
  #     if word not in ALL_STOP_WORDS:
  #       word_list.append(word)
  #     doc_list.append(word_list)
  # return word_list
   return [[word for word in simple_preprocess(str(doc)) if word not in ALL_STOP_WORDS] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
tweets_words_nostops = remove_stopwords(tweets_words)
print(tweets_words_nostops)
# Using Bigrams
tweets_words_bigrams = make_bigrams(tweets_words_nostops)

# load spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
tweets_lemmatized = lemmatization(tweets_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(tweets_lemmatized[:1])

# Create Dictionary - corpus
id2word = corpora.Dictionary(tweets_lemmatized)

# Create Corpus
tweets = tweets_lemmatized

# Term Document Frequency
corpus_tweets = [id2word.doc2bow(tweet) for tweet in tweets]

# Display tweets
print(corpus_tweets[:])
id2word[100] #id2word is an optional dictionary that maps the word_id to a token
# Display corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus_tweets[:10]]


