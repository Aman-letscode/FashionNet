import nltk
nltk.download('punkt',quiet=True)
nltk.download('nps_chat')

from transformers import pipeline
import pandas as pd

from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import WordNetLemmatizer, PorterStemmer
from sentence_transformers import SentenceTransformer, util

import json
import numpy as np

import pickle
import ipyplot
import pymysql


from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask_cors import CORS

import pymongo
from pymongo.mongo_client import MongoClient


model_id = "rasta/distilbert-base-uncased-finetuned-fashion"
classifier = pipeline("text-classification", model=model_id)

def classify(text):
    preds = classifier(text, return_all_scores=True)
    if preds[0][0]['score']  <= preds[0][1]['score']:
        return "Not Fashion"
    else:
        return "Fashion"

def attribute_extraction(txt):
    tokenized = sent_tokenize(txt)

    attributes = []
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(wordsList)

    for i,w in enumerate(tagged) :
        if w[1] in ['NN','NNS','RB'] :
            ind =i
            attr = w[0]
            while tagged[ind-1][1] in ['JJ','VBN','NN','RB','VBD','EX']:
                    attr = tagged[ind-1][0] + ' ' +  attr
                    ind = ind - 1

            if len(attr.split())==1 and txt.split()[0].lower()=='will':
                attr = tagged[ind-1][0] + ' ' +  attr

            if classify(attr) == 'Fashion':
                attributes.append(attr)
            for a in attributes:
                for b in attributes:
                    if (a!=b) and (a in b):
                        attributes.remove(a)

            for a in attributes:
                if 'fit' in a :
                    attributes = list(map(lambda x: x.replace(a, a.replace(' fit','')), attributes))
                if 'match' in a :
                    attributes = list(map(lambda x: x.replace(a, a.replace(' match','')), attributes))

    return attributes


posts = nltk.corpus.nps_chat.xml_posts()[:10000]

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]

# 10% of the total data
size = int(len(featuresets) * 0.1)

# first 10% for test_set to check the accuracy, and rest 90% after the first 10% for training
train_set, test_set = featuresets[size:], featuresets[:size]

# get the classifer from the training set
classifiers = nltk.NaiveBayesClassifier.train(train_set)
# to check the accuracy - 0.67
# print(nltk.classify.accuracy(classifier, test_set))

question_types = ["whQuestion","ynQuestion"]
def is_ques_using_nltk(ques):
    question_type = classifiers.classify(dialogue_act_features(ques))
    return question_type in question_types


question_pattern = ["do i", "do you", "what", "who", "is it", "why","would you", "how","is there",
                    "are there", "is it so", "is this true" ,"to know", "is that true", "are we", "am i",
                   "question is", "tell me more", "can i", "can we", "tell me", "can you explain",
                   "question","answer", "questions", "answers", "ask"]

helping_verbs = ["is","am","can", "are", "do", "does"]
# check with custom pipeline if still this is a question mark it as a question

def is_question(question):
    question = question.lower().strip()
    if not is_ques_using_nltk(question):
        is_ques = False
        # check if any of pattern exist in sentence
        for pattern in question_pattern:
            is_ques  = pattern in question
            if is_ques:
                break

        # there could be multiple sentences so divide the sentence
        sentence_arr = question.split(".")
        for sentence in sentence_arr:
            if len(sentence.strip()):
                # if question ends with ? or start with any helping verb
                # word_tokenize will strip by default
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in helping_verbs:
                    is_ques = True
                    break
        return is_ques
    else:
        return True



model_semantick_id = "PriaPillai/distilbert-base-uncased-finetuned-query"
classifier_sem = pipeline("text-classification", model=model_semantick_id)


ps = PorterStemmer()
verb_pattern = [ps.stem(i) for i in ['match', 'suit', 'fit', 'wear', 'pair']]
# 'be', 'go', 'are'

def semantic_check_hard_coded(txt):
    tokenized = sent_tokenize(txt)
    verbs = []

    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(wordsList)

    for i,w in enumerate(tagged) :
        if w[1] in ['VB','VBD','VBN','VBG','VBP','VBZ'] :
            verbs.append(ps.stem(w[0]))

    for v in verbs:
        if v in verb_pattern :
            return True
    return False

def semantic_check(text):
    if semantic_check_hard_coded(text):
        return True
    preds = classifier_sem(text, return_all_scores=True)
    if preds[0][0]['score']  <= preds[0][1]['score']:
        return True
    else:
        return False

def extraction_pipeline(query):
    if not is_question(query):
        message = "I am not understanding you, please enter a question that is related to fashion"
        return message, []
    elif not semantic_check(query) :
        message = "I am not sure to get your query can you please try again ?"
        return message, []
    else:
        return "Working ...",attribute_extraction(query)
    


frame = pd.read_csv('image_id.csv')
frame = frame.drop(columns=["Unnamed: 0"])
data = pd.read_csv("data.csv")


def sample(x):
    return data["Attributes"][x]


def extract_from_sample(i):
    dic = eval(sample(i))
    a = [dic[k]['attrs'] for k in dic.keys()]

    occur = []
    for i, obj in enumerate(a):
        sent = ' '.join([d[0] for d in obj]) + ' ' + list(dic.keys())[i]
        occur.append(sent)

    return occur


def extract_image(attr1, attr2, k):
    match = []
    a = 0
    for i, d in enumerate(data['Attributes']):
        l = extract_from_sample(i)
        if (attr1 in l) and (attr2 in l):
            matching_urls = list(frame[frame['id'] == i]['URL'])


            if matching_urls:
                match.append(matching_urls[0])
            else:
                # Handle the case when no URL is found for the current 'i'
                match.append("No matching URL found for this ID")

            a = a + 1
            if a == k:
                break

    if len(match) >= 1:
        ipyplot.plot_images(match, max_images=20,
                            img_width=150, show_url=False)
    else:
        print("No image found")

    return match

# from simcse import SimCSE


model_sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')


with open('index.pkl', 'rb') as f:
    index = pickle.load(f)


lemmatizer = WordNetLemmatizer()

# model_SIMCSE.index = index
# items = index['sentences']
items = index['sentences']


def similar_items(attr, sentence_embeddings_tensor):
    similar_items = []
    attr_embedding = model_sentence_transformer.encode(
        [attr], convert_to_tensor=True)

    # Calculate cosine similarity between the attribute embedding and index embeddings
    cosine_scores = util.pytorch_cos_sim(
        attr_embedding, sentence_embeddings_tensor)[0]

    # Find items with cosine similarity above the threshold
    threshold = 0.779
    similar_indices = (cosine_scores > threshold).nonzero().squeeze(dim=-1)

    # Get the corresponding similar items
    for idx in similar_indices:
        similar_items.append(items[idx])  # Assuming 'items' is defined

    return similar_items


matrix = pd.read_csv('Final_co-occurence_polyvore_Adel.csv')


def matrix_search_advice(attr, k):
    match = []
    i = 0
    append = True

    for a in matrix['bigram']:
        if attr in a:
            a = tuple(a[1:-1].replace('\'', '').split(", "))

            if attr in a[0]:
                wrd = a[1]
            else:
                wrd = a[0]

            remove = False
            if (not wrd in match) and (not attr in wrd):
                match.append(wrd)
                i = i + 1
            if i == k:
                break

    return match, i


def matrix_search_match(attr, k):
    match = []
    i = 0

    for a in matrix['bigram']:
        if attr in a:
            a = tuple(a[1:-1].replace('\'', '').split(", "))

            if attr in a[0]:
                wrd = a[1]
            else:
                wrd = a[0]

            remove = False
            if (not wrd in match) and (not attr in wrd):
                for el in match:
                    # Calculate similarity using Sentence Transformers
                    similarity_score = util.pytorch_cos_sim(
                        model_sentence_transformer.encode(
                            [el], convert_to_tensor=True),
                        model_sentence_transformer.encode(
                            [wrd], convert_to_tensor=True)
                    )[0][0]

                    if similarity_score > 0.7:
                        remove = True

                if not remove:
                    match.append(wrd)
                    i = i + 1
            if i == k:
                break

    return match, i


def garment_matching(attr, k):           # Returns k best matches to the given attribute

    attr = " ".join([lemmatizer.lemmatize(i) for i in attr.split()])
    i = 0
    match = []

    if attr in items:
        if k == 5:
            match, i = matrix_search_match(attr, k)
        if k == 10:
            match, i = matrix_search_advice(attr, k)

    else:
        similar = similar_items(attr)
        stop = False
        ind = 0
        while (not stop) and (ind < len(similar)):
            print(len(similar))
            if similar[ind] in items:
                if k == 5:
                    match, i = matrix_search_match(similar[ind], k)
                if k == 10:
                    match, i = matrix_search_advice(similar[ind], k)
                if (i > 0):
                    stop = True
                    attr = similar[ind]
            ind = ind + 1

    if (i == 0):
        message = 'This attribute was not found for the garment matching try another attribute!'
        return message, []

    return attr, match


def garment_advice(attr1, attr2, k=10):
    match, i = garment_matching(attr1, k)

    if match is None:
        return attr1, None, False

    # Encode attr2 using Sentence Transformers
    attr2_embedding = model_sentence_transformer.encode(
        [attr2], convert_to_tensor=True)

    for el in match:
        # Calculate similarity using Sentence Transformers
        similarity_score = util.pytorch_cos_sim(
            model_sentence_transformer.encode([el], convert_to_tensor=True),
            attr2_embedding
        )[0][0]

        if similarity_score > 0.9:
            return attr1, el, True

    return attr1, None, False

    return None, None, False


def check_image(num, attr1, attr2):

    bound = eval(data['boudaries(X,y,Width,Height)'][num])
    if (attr1 in bound.keys()) and (attr2 in bound.keys()):
        x1, y1, x2, y2 = bound[attr1]
        a1, b1, a2, b2 = bound[attr2]

        percentage1 = (((x2-x1)/6) + ((y2-y1)/6)) / 2
        percentage2 = (((a2-a1)/6) + ((b2-b1)/6)) / 2

        center1 = np.array([x1 + (x2-x1)/2, y1 + (y2-y1)/2])
        center2 = np.array([a1 + (a2-a1)/2, b1 + (b2-b1)/2])

        dist = np.linalg.norm(center1 - center2)

#         print(percentage1, percentage2, dist)
#         print(center1, center2)

        if percentage1 < 20 or percentage2 < 20:
            return False
        else:
            return True
    else:
        #print("One of the attributes is not found in the image")
        return False


def new_extract_image(attr1, attr2, k):
    match = []
    a = 0
    for i, d in enumerate(data['Attributes']):
        l = extract_from_sample(i)
        if (attr1 in l) and (attr2 in l) and check_image(i, attr1, attr2):
            matching_urls = list(frame[frame['id'] == i]['URL'])


            if matching_urls:
                match.append(matching_urls[0])
            else:
                # Handle the case when no URL is found for the current 'i'
                match.append("No matching URL found for this ID")

            a = a + 1
            if a == k:
                break
    # else :
        #print("No image found")

    return match


nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def end_to_end(query):
    responses = []  # Initialize an empty list to store responses

    msg, attr = extraction_pipeline(query)

    if attr is None:
        responses.append(
            "An unknown problem occurred. Please contact support.")
    elif len(attr) == 1:  # Garment matching
        attr0, match = garment_matching(attr[0], 5)
        response = f"{attr[0].capitalize()} will match with the following attributes:"
        responses.append(response)
        for item in match:
            responses.append(f"- {item}")

        # responses.append(
        #     "Here are some images of your item with some good matches:")
        # for item in match:
        #     URL = new_extract_image(attr0, item, 1)
        #     URL = list(dict.fromkeys(URL))
        #     responses.extend(URL)
    elif len(attr) == 2:  # Garment advice
        attr1, attr2, g = garment_advice(attr[0], attr[1], 10)
        if g:
            response = f"{attr1.capitalize()} would be a good match with {attr2.capitalize()}."
            responses.append(response)
            # responses.append("Here are some images of that combo:")
            # URL = new_extract_image(attr1, attr2, 5)
            # URL = list(dict.fromkeys(URL))
            # responses.extend(URL)
        elif attr1 is None:
            responses.append("Those items are not commonly worn together!")
        else:
            responses.append(attr1)
    elif len(attr) == 0:
        responses.append(msg)
    else:
        responses.append(
            'More than 2 attributes were detected, this version only supports 1 attribute for garment matching and 2 for garment advice')

    return responses  # Return a list of responses


def recommend_product(sql, product_data):
    # Convert LDJSON data to a DataFrame
    productDf = pd.DataFrame(product_data)
    
    preprocessed_sql = sql.lower()
    preprocessed_products = productDf.applymap(
        lambda x: x.lower() if isinstance(x, str) else x
    )

    # Tokenize the sentence into words
    words = word_tokenize(preprocessed_sql)

    # Perform part-of-speech tagging
    pos_tags = pos_tag(words)

    # Find adjectives and nouns and combine them
    adjective_noun_pairs = []
    i = 0
    while i != len(pos_tags):
        if pos_tags[i][1].startswith('JJ'):
            j = i
            while not pos_tags[j][1].startswith('NN'):
                j += 1

            r = []
            for k in range(i, j + 1):
                if pos_tags[k][1].startswith('JJ') or pos_tags[k][1].startswith('RB') or pos_tags[k][1].startswith('NN'):
                    r.append(pos_tags[k][0])
            adjective_noun_pairs.append(r)
            i = j + 1

        elif pos_tags[i][1].startswith('NN'):
            adjective_noun_pairs.append([pos_tags[i][0]])
            i += 1
        else:
            i += 1

    # Create TF-IDF vectorizer
    productDf['Combined'] = preprocessed_products['product_name'].str.cat(
        preprocessed_products['product_name'], sep=' '
    )

    # Prepare the list of recommended products
    recommended_products = []
    for pair in adjective_noun_pairs:
        vectorizer = TfidfVectorizer()
        product_vectors = vectorizer.fit_transform(productDf['Combined'].fillna(''))
        query_vector = vectorizer.transform([" ".join(pair)])
        similarity_scores = cosine_similarity(query_vector, product_vectors).flatten()

        ranked_indices = similarity_scores.argsort()[::-1]
        ranked_products = productDf.iloc[ranked_indices]

        recommended_urls = ranked_products['product_url'].tolist()
        recommended_prices = ranked_products['sales_price'].tolist()
        recommended_names = ranked_products['product_name'].tolist()

        recommendations = list(zip(recommended_urls[:10], recommended_prices[:10], recommended_names[:10]))
        recommended_products.extend(recommendations)

    recommended_products_list = []
    for url, price, name in recommended_products:
        recommended_product_dict = {
            'name': name,
            'url': url,
            'price': price
        }
        
        recommended_products_list.append(recommended_product_dict)

    return recommended_products_list[:5]


def extract_gender(sql):
    preprocessed_sql = sql.lower()
    
    words = word_tokenize(preprocessed_sql)

    for word in words:
        if word=="man" or word=="woman" or word=="men" or word=="women": 
            return word
    
    return None

app = Flask(__name__)
CORS(app)



# Load product data from LDJSON file with UTF-8 encoding
with open('amazon.ldjson', 'r', encoding='utf-8') as ldjson_file:
    product_data = [json.loads(line) for line in ldjson_file]




timeout = 10
connection = pymysql.connect(
  charset="utf8mb4",
  connect_timeout=timeout,
  cursorclass=pymysql.cursors.DictCursor,
  db="FashionNet",
  host="mysql-f08618e-patents-pioneer.a.aivencloud.com",
  password="AVNS_Jr5rzjhptvYupdhcJwq",
  read_timeout=timeout,
  port=21454,
  user="avnadmin",
  write_timeout=timeout,
)
  
# try:
cursor = connection.cursor()



@app.route('/prompt', methods=['POST'])
def generate_prompt():
    data = request.get_json()
    query = data['query']
    print(query)

    user = data['user']
    gender = extract_gender(query)
    if gender is None:
        cursor.execute('SELECT * FROM user WHERE username = %s',(user))
        answer = cursor.fetchone()
        if answer:
            gender = answer['gender']
            if gender=="man":
                gender = "-men"
            else:
                gender = "-women"
    bot_response = end_to_end(query)
    print(bot_response)


    return jsonify({"answer": bot_response,"gender":gender})
    


@app.route('/suggest', methods=['POST'])
def generate_suggest():
    data = request.get_json()
    query = data['query'] + " for " + data['gender']
    print(query)
    
    recommendations = recommend_product(query, product_data)

    return jsonify({"answer": recommendations})
    


# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/login', methods=['POST'])
def login():

    
    data = request.get_json()
    print(data)
    username = data['username']
    password = data['password']
    
    
    cursor.execute('SELECT * FROM user WHERE username = %s AND password = %s', (username, password))
    #Fetch one record and return result
    account = cursor.fetchone()
    # If account exists in accounts table in out database
    if account:
        
        return jsonify({"status":"success","ok":True})
    else:
        return jsonify({"status":"failed","ok":False})
        
    
  

# http://localhost:5000/pythonlogin/register 
# This will be the registration page, we need to use both GET and POST requests
@app.route('/signup', methods=['POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    
    data = request.get_json()
    print(data)
    username = data['username']
    password = data['password']
    email = data['email']
    gender = data['gender']
        
    # user_doc = [{"username":username,"password":password,"email":email}]
        
    cursor.execute('SELECT * FROM user WHERE username = %s', (username))
    cursor.execute( "SELECT * FROM user WHERE username LIKE %s", [username] )
    account = cursor.fetchone()
        # # If account exists show error and validation checks
    if not username or not password or not email:
        print("Incorrect username/password!", "danger")
    if account:
        print("Account already exists!", "danger")
    else:
        # # Account doesnt exists and the form data is valid, now insert new account into accounts table
        cursor.execute('INSERT INTO user VALUES (NULL,%s, %s, %s,%s)', (username,email, password,gender))
        connection.commit();
        print("You have successfully registered!", "success")
        return jsonify({"status":"success", "ok":True})
    
    return jsonify({"status":"failed", "ok":False})
            
            



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3001)
    
    
    
# from flask_mysqldb import MySQL
# import MySQLdb.cursors

# # Enter your database connection details below
# app.config['MYSQL_HOST'] = 'mysql-f08618e-patents-pioneer.a.aivencloud.com:21454'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'AVNS_Jr5rzjhptvYupdhcJwq' #Replace ******* with  your database password.
# app.config['MYSQL_DB'] = 'FashionNet'


# # Intialize MySQL
# mysql = MySQL(app)

# cursor.execute("CREATE TABLE user (id INT PRIMARY KEY AUTO_INCREMENT,username VARCHAR(255) NOT NULL,email VARCHAR(255) NOT NULL,password VARCHAR(255) NOT NULL)")
#   cursor.execute("INSERT INTO mytest (username,email,password) VALUES (aman,,1234567890), (2)")
# finally:
#   connection.close()



  # u = user.find_one({"username":username, "password":password})
            
    # return render_template('auth/login.html',title="Login")

