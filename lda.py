import pandas as pd
import gensim
import pyLDAvis.gensim
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/atom/Desktop/dissertation/news.csv")

# data analysis
df['word_count'] = df['title'].apply(lambda x: len(x.split()))
average_word_count_title = df['word_count'].mean()
print("title:", average_word_count_title)

df['word_count2'] = df['body'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
average_word_count_body = df['word_count2'].mean()
print("body:", average_word_count_body)

df['category'] = df['category']. map(lambda x: x.lower())
category_counts = df['category'].value_counts()

# category bar chart
plt.bar(category_counts.index, category_counts.values, color= 'grey')
plt.xlabel('Category')
plt.ylabel('Amount of data point')
plt.xticks(rotation=45, ha='right')
for i, count in enumerate(category_counts.values):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()

# Load the regular expression library
import re
# convert all float to string type
df = df.astype(str)
# split traing set and testing set
def train_test_split(df, test_size=0.2, random_state= 100):
    train_df = df.sample(frac=1 - test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    return train_df, test_df

train_df1, test_df1 = train_test_split(df, test_size=0.2, random_state=1)
train_df2 = train_df1
test_df2 = test_df1
train_df3 = train_df1
test_df3 = test_df1
train_df3['full'] = train_df3['body'].str.cat(train_df3['title'], sep=' ')
test_df3['full'] = test_df3['body'].str.cat(test_df3['title'], sep=' ')

element_counts_train = train_df1['category'].value_counts()
print(element_counts_train)

element_counts_test = test_df1['category'].value_counts()
print(element_counts_test)

df['word_count'] = df['word_count'].astype(float)
df['word_count2'] = df['word_count2'].astype(float)
df['word_count_full']= df['word_count']+df['word_count2']
average_length_title = df.groupby('category')['word_count'].mean()
average_length_body = df.groupby('category')['word_count2'].mean()
average_length_full = df.groupby('category')['word_count_full'].mean()
# Remove punctuation
train_df1['title'] = train_df1['title'].map(lambda x: re.sub('[,.!?\']', '', x))
train_df1['body'] = train_df1['body'].map(lambda x: re.sub('[,.!?\']', '', x))
train_df2['title'] = train_df2['title'].map(lambda x: re.sub('[,.!?\']', '', x))
train_df2['body'] = train_df2['body'].map(lambda x: re.sub('[,.!?\']', '', x))
train_df3['full'] = train_df3['full'].map(lambda x: re.sub('[,.!?\']', '', x))

test_df1['title'] = test_df1['title'].map(lambda x: re.sub('[,.!?\']', '', x))
test_df1['body'] = test_df1['body'].map(lambda x: re.sub('[,.!?\']', '', x))
test_df2['title'] = test_df2['title'].map(lambda x: re.sub('[,.!?\']', '', x))
test_df2['body'] = test_df2['body'].map(lambda x: re.sub('[,.!?\']', '', x))
test_df3['full'] = test_df3['full'].map(lambda x: re.sub('[,.!?\']', '', x))

# convert to lowercase
train_df1['category'] = train_df1['category']. map(lambda x: x.lower())
train_df1['title'] = train_df1['title']. map(lambda x: x.lower())
train_df1['body'] = train_df1['body']. map(lambda x: x.lower())
train_df2['category'] = train_df2['category']. map(lambda x: x.lower())
train_df2['title'] = train_df2['title']. map(lambda x: x.lower())
train_df2['body'] = train_df2['body']. map(lambda x: x.lower())
train_df3['category'] = train_df3['category']. map(lambda x: x.lower())
train_df3['full'] = train_df3['full']. map(lambda x: x.lower())

test_df1['category'] = test_df1['category']. map(lambda x: x.lower())
test_df1['title'] = test_df1['title']. map(lambda x: x.lower())
test_df1['body'] = test_df1['body']. map(lambda x: x.lower())
test_df2['category'] = test_df2['category']. map(lambda x: x.lower())
test_df2['title'] = test_df2['title']. map(lambda x: x.lower())
test_df2['body'] = test_df2['body']. map(lambda x: x.lower())
test_df3['category'] = test_df3['category']. map(lambda x: x.lower())
test_df3['full'] = test_df3['full']. map(lambda x: x.lower())

# We start by tokenizing the text and removing stopwords
## environment
from gensim.utils import simple_preprocess
import ssl
ssl._create_default_https_context = ssl._create_unverified_context # the verify close
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['like','one','thing','said','say','also','told','know','think','make','get','even',
                   'people','year','years','time','new','news','would','could',
                   'many','first','us','trump','giuliani','day','two','twice','much','want'])

# Do simple preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

#  remove stop words
def remove_stopwords(texts): # define remove_stopwords
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# PART1 : Model comparation (body VS title VS full)
# ------------------------------------------ boby ------------------------------------------------------
data_body = train_df1.body.values.tolist()
data_body_test = test_df1.body.values.tolist()
data_body = remove_stopwords(data_body)
data_body_test = remove_stopwords(data_body_test)
data_words_body = list(sent_to_words(data_body))
data_words_body_test = list(sent_to_words(data_body_test))

# Do lemmatization
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
def lemmatize_lists(lists):
    lemmatizer = WordNetLemmatizer()
    lemmatized_lists = []
    for word_list in lists:
        lemmatized_list = [lemmatizer.lemmatize(word) for word in word_list]
        lemmatized_lists.append(lemmatized_list)
    return lemmatized_lists
data_words_body = lemmatize_lists(data_words_body)
data_words_body_test = lemmatize_lists(data_words_body_test)

# convert the tokenized object into a corpus and dictionary
import gensim.corpora as corpora
# Create Dictionary
idword_body = corpora.Dictionary(data_words_body)
idword_body_test = corpora.Dictionary(data_words_body_test)
# Create Corpus
texts_body = data_words_body
texts_body_test = data_words_body_test
# Term Document Frequency
corpus_body = [idword_body.doc2bow(text) for text in texts_body]
corpus_body_test = [idword_body.doc2bow(text) for text in texts_body_test]

# Build LDA model (body)
ldamodel_body = gensim.models.ldamodel.LdaModel(corpus_body, num_topics=14,random_state=100, id2word = idword_body)
# Print the Keyword in the 10 topics
from pprint import pprint
pprint(ldamodel_body.print_topics())

# printing the topic associations with the documents (training set)
topic_body = []
for i in ldamodel_body[corpus_body]:
    max_point_body = max(i, key=lambda x: x[1])
    topic_body.append(max_point_body[0])

train_df1['body_topic']= topic_body
train_df1 = train_df1.sort_values(by='category')

# train_df1_art = train_df1[train_df1['category'] == 'arts & culture']
# counts = train_df1_art['body_topic'].value_counts()

# # bar chart
# import matplotlib.pyplot as plt
# counts.plot(kind='bar')

# classification discussion part (label topics)
result1_body = train_df1.groupby('category')['body_topic'].apply(lambda x: x.value_counts().idxmax())
result2_body = train_df1.groupby('category')['body_topic'].apply(lambda x: x.value_counts(normalize=True).nlargest(4))

# Visualize the topics
import pyLDAvis.gensim_models
import pyLDAvis
import matplotlib

d = pyLDAvis.gensim_models.prepare(ldamodel_body, corpus_body, idword_body)
pyLDAvis.display(d)
pyLDAvis.save_html(d, 'lda_body.html')

# classification result (body testing set)
topic_body_test = []
for i in ldamodel_body[corpus_body_test]:
    max_point_body_test = max(i, key=lambda x: x[1])
    topic_body_test.append(max_point_body_test[0])

test_df1['body_topic']= topic_body_test
test_df1 = test_df1.sort_values(by='category')

# label the topics
test_category_mapping = {"arts & culture": 4,
                         "business": 9,
                         "comedy": 5,
                         "crime": 7,
                         "education": 11,
                         "entertainment": 1,
                         "environment": 2,
                         "media": 13,
                         "politics": 0,
                         "religion": 12,
                         "science": 6,
                         "sports": 8,
                         "tech": 10,
                         "women": 3}

test_df1["label"] = test_df1["category"].apply(lambda x: test_category_mapping[x])
# model classification accuracy(body)
count_same_elements = test_df1.groupby(["body_topic", "label"]).size().reset_index(name="count")
number_accurate_body = (test_df1["body_topic"] == test_df1["label"]).sum()
rate_accurate_body = number_accurate_body/1375

# ------------------------------------------ title ------------------------------------------------------
data_title = train_df1.title.values.tolist()
data_title_test = test_df1.title.values.tolist()
data_title = remove_stopwords(data_title)
data_title_test = remove_stopwords(data_title_test)
data_words_title = list(sent_to_words(data_title))
data_words_title_test = list(sent_to_words(data_title_test))

# Do lemmatization
data_words_title = lemmatize_lists(data_words_title)
data_words_title_test = lemmatize_lists(data_words_title_test)

# Create Dictionary
idword_title = corpora.Dictionary(data_words_title)
idword_title_test = corpora.Dictionary(data_words_title_test)
# Create Corpus
texts_title = data_words_title
texts_title_test = data_words_title_test
# Term Document Frequency
corpus_title = [idword_title.doc2bow(text) for text in texts_title]
corpus_title_test = [idword_title.doc2bow(text) for text in texts_title_test]

# Build LDA model (title)
ldamodel_title = gensim.models.ldamodel.LdaModel(corpus_title, num_topics=14,random_state=100, id2word = idword_title)
# Print the Keyword in the 10 topics
from pprint import pprint
pprint(ldamodel_title.print_topics())

# printing the topic associations with the documents (training set)
topic_title = []
for i in ldamodel_title[corpus_title]:
    max_point_title = max(i, key=lambda x: x[1])
    topic_title.append(max_point_title[0])

train_df2['title_topic']= topic_title
train_df2 = train_df2.sort_values(by='category')

# train_df1_art = train_df1[train_df1['category'] == 'arts & culture']
# counts = train_df1_art['body_topic'].value_counts()

# # bar chart
# import matplotlib.pyplot as plt
# counts.plot(kind='bar')

# classification discussion part (label topics)
result1_title = train_df2.groupby('category')['title_topic'].apply(lambda x: x.value_counts().idxmax())
result2_title = train_df2.groupby('category')['title_topic'].apply(lambda x: x.value_counts(normalize=True).nlargest(4))

d2 = pyLDAvis.gensim_models.prepare(ldamodel_title, corpus_title, idword_title)
pyLDAvis.display(d2)
pyLDAvis.save_html(d2, 'lda_title.html')

# ------------------------------------------ full ------------------------------------------------------

data_full = train_df3.full.values.tolist()
data_full_test = test_df3.full.values.tolist()
data_full = remove_stopwords(data_full)
data_full_test = remove_stopwords(data_full_test)
data_words_full = list(sent_to_words(data_full))
data_words_full_test = list(sent_to_words(data_full_test))

# Do lemmatization
data_words_full = lemmatize_lists(data_words_full)
data_words_full_test = lemmatize_lists(data_words_full_test)

# Create Dictionary
idword_full = corpora.Dictionary(data_words_full)
idword_full_test = corpora.Dictionary(data_words_full_test)
# Create Corpus
texts_full = data_words_full
texts_full_test = data_words_full_test
# Term Document Frequency
corpus_full = [idword_full.doc2bow(text) for text in texts_full]
corpus_full_test = [idword_full.doc2bow(text) for text in texts_full_test]

# Build LDA model (full)
ldamodel_full = gensim.models.ldamodel.LdaModel(corpus_full, num_topics=14,random_state=100, id2word = idword_full)
# Print the Keyword in the 10 topics
from pprint import pprint
pprint(ldamodel_full.print_topics())

# printing the topic associations with the documents (training set)
topic_full = []
for i in ldamodel_full[corpus_full]:
    max_point_full = max(i, key=lambda x: x[1])
    topic_full.append(max_point_full[0])

train_df3['full_topic']= topic_full
train_df3 = train_df3.sort_values(by='category')

# classification discussion part (label topics)
result1_full = train_df3.groupby('category')['full_topic'].apply(lambda x: x.value_counts().idxmax())
result2_full = train_df3.groupby('category')['full_topic'].apply(lambda x: x.value_counts(normalize=True).nlargest(4))

# Visualize the topics
d3 = pyLDAvis.gensim_models.prepare(ldamodel_full, corpus_full, idword_full)
pyLDAvis.display(d3)
pyLDAvis.save_html(d3, 'lda_full.html')

# classification result (full testing set)
topic_full_test = []
for i in ldamodel_full[corpus_full_test]:
    max_point_full_test = max(i, key=lambda x: x[1])
    topic_full_test.append(max_point_full_test[0])

test_df3['full_topic']= topic_full_test
test_df3 = test_df3.sort_values(by='category')

# label the topics
test_category_mapping_full = {"arts & culture": 6,
                         "business": 3,
                         "comedy": 2,
                         "crime": 7,
                         "education": 8,
                         "entertainment": 11,
                         "environment": 4,
                         "media": 1,
                         "politics": 0,
                         "religion": 13,
                         "science": 12,
                         "sports": 5,
                         "tech": 10,
                         "women": 9}

test_df3["label"] = test_df3["category"].apply(lambda x: test_category_mapping_full[x])
# model classification accuracy(full)
count_full = test_df3.groupby(["full_topic", "label"]).size().reset_index(name="count")
number_accurate_full = (test_df3["full_topic"] == test_df3["label"]).sum()
rate_accurate_full = number_accurate_full/1375

# result = test_df3[test_df3['full_topic'] == test_df3['label']].groupby('category').size().reset_index(name='Count')
#------------------------------------ PART2: Model optimisation ----------------------------------------------
# Compute perplexity and coherance
min_topics = 2
max_topics = 14
topics_range = range(min_topics, max_topics+1)

# # perplexity
# perplexity_values = []
#
# for num_topics in topics_range:
#     lda_model2 = gensim.models.ldamodel.LdaModel(corpus=corpus_full, id2word=idword_full, num_topics=num_topics, random_state=19)
#     perplexity2 = lda_model2.log_perplexity(corpus_full)
#     perplexity_values.append(perplexity2)
#
# # plot (Perplexity)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(topics_range, perplexity_values, marker='o')
# plt.title('Perplexity vs Number of Topics')
# plt.xlabel('Number of Topics')
# plt.ylabel('Perplexity')
# plt.xticks(topics_range)
# plt.grid(True)
# plt.show()

# corherance
# body text MODEL A
from gensim.models.coherencemodel import CoherenceModel
coherence_valuesA = []
for num_topics in topics_range:
    lda_model4 = gensim.models.ldamodel.LdaModel(corpus=corpus_body, id2word=idword_body, num_topics=num_topics, random_state=89)
    coherence_model1 = CoherenceModel(model=lda_model4, texts=data_words_body, dictionary=idword_body, coherence='c_v')
    coherence_score1 = coherence_model1.get_coherence()
    coherence_valuesA.append(coherence_score1)
corherence_model_body_original= CoherenceModel(model=lda_model4, texts=data_words_body, dictionary=idword_body, coherence='c_v').get_coherence()

# plot (Coherence)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(topics_range, coherence_valuesA, marker='o')
plt.xlabel('Number of Topics (K)')
plt.ylabel('Coherence Score')
plt.xticks(topics_range)
plt.grid(True)
plt.show()


# full text MODEL B
coherence_values = []
for num_topics in topics_range:
    lda_model3 = gensim.models.ldamodel.LdaModel(corpus=corpus_full, id2word=idword_full, num_topics=num_topics, random_state=19)
    coherence_model = CoherenceModel(model=lda_model3, texts=data_words_full, dictionary=idword_full, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_values.append(coherence_score)
corherence_model_full_original= CoherenceModel(model=lda_model3, texts=data_words_full, dictionary=idword_full, coherence='c_v').get_coherence()
# plot (Coherence)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(topics_range, coherence_values, marker='o')
plt.xlabel('Number of Topics (K)')
plt.ylabel('Coherence Score')
plt.xticks(topics_range)
plt.grid(True)
plt.show()

# --------------------------- Hyperparameter Tuning ---------------------------
# model A
from  .model_selection import ParameterGrid
param_grid = {'alpha': [0.01, 0.21, 0.41, 0.61, 0.81,'symmetric', 'asymmetric'],'eta': [0.01, 0.21, 0.41, 0.61, 0.81,'symmetric']}
param_combinations = list(ParameterGrid(param_grid))
coherence_results_A = []

for params in param_combinations:
    lda_model_param_A = gensim.models.ldamodel.LdaModel(corpus=corpus_body, id2word=idword_body,random_state=89, num_topics=9, **params)
    coherence_model_param_A = CoherenceModel(model=lda_model_param_A, texts=data_words_body, dictionary=idword_body, coherence='c_v')
    coherence_score_param_A = coherence_model_param_A.get_coherence()
    coherence_results_A.append({'alpha': params['alpha'], 'eta': params['eta'], 'coherence_score': coherence_score_param_A})

results_parm_A = pd.DataFrame(coherence_results_A)
results_parm_A = results_parm_A.sort_values(by='coherence_score', ascending=False)

# model B
from sklearn.model_selection import ParameterGrid
param_grid = {'alpha': [0.01, 0.21, 0.41, 0.61, 0.81,'symmetric', 'asymmetric'],'eta': [0.01, 0.21, 0.41, 0.61, 0.81,'symmetric']}
param_combinations = list(ParameterGrid(param_grid))

coherence_results = []

for params in param_combinations:
    lda_model_param = gensim.models.ldamodel.LdaModel(corpus=corpus_full, id2word=idword_full,random_state=19, num_topics=11, **params)
    coherence_model_param = CoherenceModel(model=lda_model_param, texts=data_words_full, dictionary=idword_full, coherence='c_v')
    coherence_score_param = coherence_model_param.get_coherence()
    coherence_results.append({'alpha': params['alpha'], 'eta': params['eta'], 'coherence_score': coherence_score_param})

# 转换为DataFrame并排序
results_parm = pd.DataFrame(coherence_results)
results_parm = results_parm.sort_values(by='coherence_score', ascending=False)



