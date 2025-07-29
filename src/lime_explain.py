
import lime
import lime.lime_image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

model = tf.keras.models.load_model('models/covid_ct_vgg16.h5')

def explain_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array.astype('double'), 
                                             model.predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    plt.imshow(mark_boundaries(temp/255.0, mask))
    plt.axis('off')
    plt.savefig('outputs/lime_explanation.png')

if __name__ == "__main__":
    explain_image('data/sample_ct_image.png')
