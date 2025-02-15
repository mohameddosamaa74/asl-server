import cv2, os, nltk, time, joblib
import numpy as np
from statistics import mode
from signs import signs, letters
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Sign_text():
    lemmatizer = WordNetLemmatizer()
    text=""
    path = "videos/"
    words, ids = [], [] #words and ids corresponding to words

    def remove_punctuation(self, text=None)-> str:
        if text!=None:
            self.text= text
        self.text = self.text.lower()
        translator = str.maketrans('', '', "?:!.,;")
        self.text = self.text.translate(translator)
        return self.text

    def remove_stopwords(self, words=None)-> list:
        if words!=None:
            self.words=words        
        stop_words = {"are","should have","an",'the','hasn’t','won’t','doing','is',
                    'will','don’t','having','had','has','hadn’t','shouldn’t','a',
                    'was','were','am', 'have', 'has',"’ve", "’s", "n’t", "’", "s", "t",
                    "’d", "the", "are", "than", "should", "if", "some", "ll", "’ll",
                    "re", "ca", "an", "tahn", "of", "by", "then", "’", "lot",
                    'wasn’t','wouldn’t','didn’t','been','shall','can','could',
                    'haven’t','did',"does","were not",'would','that','there',
                    'won', "you’ll", "should’ve", "mightn’t", 'more', 'mustn', 'shouldn',
                    'hasn', 'didn', "didn’t", 'couldn', "you'd", "wasn't", 'whom', 'wouldn',
                    'wasn', 'most', 'own', 'through', 'few', 'o', 'haven', 'nor',
                    'mightn', 'very', 'too', 'so', 'do', "mustn't", "isn't", 'ours',
                    'y', "aren't", "don’t", 'm', "you’ve", "you’re", "shan’t", 'over', "hadn’t",
                    'such', 'and', "shouldn’t", 'out', "that’ll", 'our', "couldn’t", 'while',
                    'this', "won’t", 'ma', 'off', 'as', 'ain', 'about', 'aren', "hasn’t", 'hadn', 
                    'don', 'weren', 'into', "weren’t", 'shan', 'each', 'isn', 'himself', 'now', 
                    "doesn’t", 'but', 'needn', 'down', 'below', "haven’t", 'doesn', 'themselves', 
                    "wouldn’t", 'd', 'further', 'any', 'vebeing',"’d"}

        # stop_words = set(stopwords.words("english"))
        filtered_text = []
        for word in self.words:
            if word not in stop_words:
                if word == "n't":
                    filtered_text.append("not")
                else:
                    filtered_text.append(word)
        self.words = filtered_text
        return  self.words

    def lemmatize_word(self, words=None)-> list:
        if words!=None:
            self.words=words
        # provide context i.e. part-of-speech
        lemmas_v = [self.lemmatizer.lemmatize(word, pos ='v') for word in self.words]
        lemmas_n = [self.lemmatizer.lemmatize(word, pos ='n') for word in lemmas_v]
        lemmas_a = [self.lemmatizer.lemmatize(word, pos ='a') for word in lemmas_n]
        lemmas_r = [self.lemmatizer.lemmatize(word, pos ='r') for word in lemmas_a]
        self.words = lemmas_r
        return self.words

    def word_id(self, words=None)->list:
        if words!=None:
            self.words=words
        self.ids = []
        for word in self.words:
            id = signs.get(word)
            if id == None:
                for char in word:
                    self.ids.append(letters.get(char))
            else:
                self.ids.append(id)
        return self.ids
    
    def sign_gen(self, text=None): 
        """
            generate signs corresponding to text 
            take text and return generator of sign's video frames
        """ 
        if text != None:
            self.text = text
        self.remove_punctuation() # remove punctuation
        self.words = word_tokenize(self.text) # tokenize text 
        self.remove_stopwords()    # remove stop words
        self.lemmatize_word() # lemmatize string
        self.word_id()

        for id in self.ids:
            try:
                vid = cv2.VideoCapture(self.path + str(id) + ".mp4")
                for i in range(10):
                        success, frame = vid.read()  # read the camera frame
                # while True:
                for i in range(60):     
                    success, frame = vid.read()  # read the camera frame
                    if not success:
                        break
                    else:
<<<<<<< HEAD
#                        time.sleep(0.01)  
=======
#                         time.sleep(0.01)  
>>>>>>> bab8ab6af95eb0f1aafbf4b3823a9123be21ddeb
                        width = int((frame.shape[1])*0.50)
                        height = int((frame.shape[0])*0.50)  
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) 
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield frame
            except:
                continue


class Sign_predict():
    checkpoints = 'model.pkl'
    clf = joblib.load(checkpoints)

    def sign_predict(self, landmarks):
        try:
            pred = []
            for frame in landmarks:
                cleaned = []
                for mark in frame.landmark:
                    cleaned.append(mark.x)
                    cleaned.append(mark.y)
                cleaned = np.reshape(cleaned, (1,-1))
                if cleaned.sum():
                    pred.append(self.clf.predict(cleaned)[0])
            return mode(pred)
        except:
            return False
