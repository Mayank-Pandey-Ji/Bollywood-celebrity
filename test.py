## load img -> face detection and extract features
## find the cosine distance of current image with all the features
## recommend that image with least distance

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

detector = MTCNN()

sample_img = cv2.imread('sample/pppppp.jpg')

results = detector.detect_faces(sample_img)

x,y,height,width = results[0]['box']

face = sample_img[y:y+height, x:x+width]

## extract features

image = Image.fromarray(face)
image = image.resize((224, 224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')
face_array = np.expand_dims(face_array, axis=0)
face_array = preprocess_input(face_array)

features = model.predict(face_array).flatten()


## find cosine distance of this image with all the features

similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]

temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)