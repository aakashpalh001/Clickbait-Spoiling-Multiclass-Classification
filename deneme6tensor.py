import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json

# Veriyi yükle
train_df = pd.read_json("train.jsonl", lines=True)
validation_df = pd.read_json("validation.jsonl", lines=True)

# Temizleme fonksiyonu
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    return text

# `postText` içindeki her bir metin listesinin ilk elemanını al ve temizle
train_df['cleaned_text'] = train_df['postText'].apply(lambda x: clean_text(x[0] if x else ""))
validation_df['cleaned_text'] = validation_df['postText'].apply(lambda x: clean_text(x[0] if x else ""))

# Etiket haritası oluştur ve etiketleri sayısallaştır
label_map = {"phrase": 0, "passage": 1, "multi": 2}
train_df['label'] = train_df['tags'].apply(lambda x: label_map[x[0]] if x else None)
validation_df['label'] = validation_df['tags'].apply(lambda x: label_map[x[0]] if x else None)

# Metinleri ve etiketleri ayır
X_train = train_df['cleaned_text'].values
y_train = train_df['label'].values
X_val = validation_df['cleaned_text'].values
y_val = validation_df['label'].values

# Tokenize ve padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(list(X_train) + list(X_val))
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_train_pad = pad_sequences(X_train_seq, maxlen=200)
X_val_pad = pad_sequences(X_val_seq, maxlen=200)

# Modeli oluştur
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=200),
    LSTM(units=256, return_sequences=True),
    LSTM(units=256),
    Dense(units=3, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
batch_size = 50
num_epochs = 2
model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=num_epochs, batch_size=batch_size, verbose=1)

def predict_spoiler_type(model, tokenizer, text):
    # Temizleme işlemi
    cleaned_text = clean_text(text)
    
    # Tokenize ve padding işlemleri
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    
    # Tahmin yapma
    predicted_probs = model.predict(padded_sequence)[0]
    predicted_label_idx = np.argmax(predicted_probs)
    
    # Etiketi dönüştürme
    label_map_reverse = {v: k for k, v in label_map.items()}
    predicted_label = label_map_reverse[predicted_label_idx]
    
    return predicted_label

# Örnek bir metin ile tahmin yapma
sample_text = "This is a great movie!"
predicted_spoiler_type = predict_spoiler_type(model, tokenizer, sample_text)
print("Predicted Spoiler Type:", predicted_spoiler_type)

# Tahminleri gerçekleştir
test_probs = model.predict(X_val_pad)

# En yüksek olasılığa sahip sınıfın endeksini seçme
test_predictions = np.argmax(test_probs, axis=1)

# UUID'leri ve tahminleri birleştir
test_results = []
for uuid, label in zip(validation_df['uuid'].values, test_predictions):
    test_results.append({"uuid": uuid, "spoilerType": int(label)})  # int64 --> int dönüşümü

# JSON dosyasına yaz
with open('test_predictions.json', 'w') as f:
    json.dump(test_results, f, indent=4)

print("Tahminler 'test_predictions.json' dosyasına kaydedildi.")

# Doğrulama fonksiyonu
def evaluate_performance(model, X_val_pad, y_val):
    val_loss, val_accuracy = model.evaluate(X_val_pad, y_val, verbose=0)
    test_predictions = np.argmax(model.predict(X_val_pad), axis=1)
    report = classification_report(y_val, test_predictions, target_names=label_map.keys(), output_dict=True)
    accuracy = report['accuracy']
    f1_score = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    return accuracy, f1_score, precision, recall, val_loss, val_accuracy

# Model performansını değerlendirme
accuracy, f1_score, precision, recall, val_loss, val_accuracy = evaluate_performance(model, X_val_pad, y_val)

# Sonuçları yazdırma
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1_score:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Etiketlerin dağılımını görselleştirme
train_df['tags'].apply(lambda x: x[0] if x else None).value_counts().plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.show()

# Değerlendirme metriklerini grafik olarak görselleştirme
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
values = [accuracy, f1_score, precision, recall]

plt.figure(figsize=(10, 5))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.ylim(0, 1)
plt.show()
