import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from tensorflow.keras.preprocessing import image  # ✅ keep this import

st.title("Which Celebrity Do You Look Like?")

uploaded_image = st.file_uploader("Choose an image")
if uploaded_image is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_image.read())
        temp_file_path = temp_file.name

    input_image = Image.open(temp_file_path)
    st.image(input_image, caption='Uploaded Image', use_column_width=True)

    # ✅ Load models outside image loop to improve performance
    detector = MTCNN()
    model = VGGFace(model='resnet50', include_top=False,
                    input_shape=(224, 224, 3), pooling='avg')

    # ✅ Load precomputed embeddings and filenames
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)

    img = cv2.imread(temp_file_path)
    results = detector.detect_faces(img)

    if results:
        x, y, width, height = results[0]['box']
        x, y = abs(x), abs(y)  # ✅ handle negative coordinates safely
        face = img[y:y + height, x:x + width]

        image_resized = Image.fromarray(face).resize((224, 224))
        face_array = np.asarray(image_resized).astype('float32')
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)

        # ✅ Predict embeddings
        features = model.predict(face_array).flatten()

        # ✅ Compute cosine similarities
        similarity = [cosine_similarity(features.reshape(1, -1), feat.reshape(1, -1))[0][0]
                      for feat in feature_list]

        index_pos = sorted(list(enumerate(similarity)),
                           reverse=True, key=lambda x: x[1])[0][0]

        matched_image = cv2.imread(filenames[index_pos])
        st.image(matched_image, caption='Matched Celebrity', use_column_width=True)
        st.success(f"Confidence: {round(similarity[index_pos] * 100, 2)}%")
    else:
        st.warning("No face detected. Please try another image.")
