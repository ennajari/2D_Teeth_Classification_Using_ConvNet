import tensorflow as tf
from src.data_loader import load_dataset
from src.model import create_model
from src.preprocess import create_data_augmentation, prepare_for_training

# Charger et préparer les données
images, labels = load_dataset('chemin/vers/images', 'chemin/vers/labels')
ds = tf.data.Dataset.from_tensor_slices((images, labels))
ds = prepare_for_training(ds)

# Créer et compiler le modèle
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(ds, epochs=10)  # Ajustez le nombre d'époques selon vos besoins

# Sauvegarder les poids du modèle
model.save_weights('model_weights.h5')