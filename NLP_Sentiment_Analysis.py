import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Conv1D,GlobalMaxPooling1D,Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

max_words=10000
max_length=500
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_words)

X_train[0]
y_train[0]

X_train=pad_sequences(X_train,maxlen=max_length)
X_test=pad_sequences(X_test,maxlen=max_length)

#Build a CNN model
model=Sequential([
    Embedding(input_dim=max_words,output_dim=128,input_length=max_length),
    Conv1D(128,5,activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss,accuracy=model.evaluate(X_test,y_test)
print(f"Accuracy on test data:{accuracy:.2f}")

word_index=imdb.get_word_index()
def predict_review(text):
    # Tokenize and pad the input text
    encoded = [word_index.get(word, 0) for word in text.lower().split()]
    padded = pad_sequences([encoded], maxlen=max_length)
    prediction = model.predict(padded)[0][0]
    return 'Positive' if prediction > 0.5 else 'Negative'

print(predict_review("This movie was really fantastic!"))