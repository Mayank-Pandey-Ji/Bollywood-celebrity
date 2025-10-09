# import os
# import pickle
# actors = os.listdir('data')

# filename = []
# for actor in actors:
#     for file in os.listdir(os.path.join('data', actor)):
#         filename.append(os.path.join('data', actor, file))

# pickle.dump(filename, open('filenames.pkl', 'wb'))

from tensorflow.keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import pickle



filenames = pickle.load(open('filenames.pkl', 'rb'))
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

features = []
for file in filenames:
    features.append(feature_extractor(file, model))

pickle.dump(features, open('embeddings.pkl', 'wb'))