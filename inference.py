import pickle

data = ["i Rahul tiwari, can't verify that the information provide is correct and updated"]
#loading the transform model
tfidf=pickle.load(open('tranform.pkl','rb'))


# loading the model
clf = pickle.load(open('model.pkl', 'rb'))

vect = tfidf.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)