# import libraries
import pandas as pd
import numpy as np
import os
import pandas as pd
import nltk
import re
from pprint import pprint

#nltk.download('punkt')
#nltk.download('wordnet')
nltk.download('stopwords')

# Import word_tokenize and stopwords from nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer

!python - m spacy download en_core_web_lg
!pip install pyldavis
!pip install chart_studio
!pip install --upgrade autopep8
!pip install plotly==4.*

# import Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from wordcloud import STOPWORDS

# import spacy for lemmatization
import spacy
from spacy.lang.en import English
spacy.load('en')

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

#for Visualizations
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
#import chart_studio
#import chart_studio.plotly as py
#import chart_studio.tools as tls

