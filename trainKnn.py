import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from PIL import Image
import joblib



lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('leapgestrecog/leapGestRecog/00/'):
    lookup[j] = count
    reverselookup[count] = j
    count = count + 1

x_data = []
y_data = []
datacount = 0
for i in range(10):
    for j in os.listdir(os.path.join('leapgestrecog/leapGestRecog/', '0' + str(i))):
        count = 0
        for k in os.listdir(os.path.join('leapgestrecog/leapGestRecog/', '0' + str(i), j)):
            img_path = os.path.join('leapgestrecog/leapGestRecog/', '0' + str(i), j, k)
            img = Image.open(img_path).convert('L')
            img = img.resize((320, 120))
            arr = np.array(img) / 255.0 
            x_data.append(arr.flatten())
            count += 1
        y_values = np.full((count, 1), lookup[j])
        y_data.append(y_values)
        datacount += count

x_data = np.array(x_data, dtype='float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


k = 3


knn_classifier = KNeighborsClassifier(n_neighbors=k)


knn_classifier.fit(X_train, y_train)


y_pred = knn_classifier.predict(X_test)

model_filename = 'knn_model.joblib'
joblib.dump(knn_classifier, model_filename)
