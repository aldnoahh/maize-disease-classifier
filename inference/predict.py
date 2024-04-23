"""
sudo docker run --rm --runtime=nvidia --shm-size=12g --ulimit memlock=-1 --ulimit stack=67108864 --name triton --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/ubuntu/models:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1
"""

import numpy as np
import tensorflow as tf
import onnxruntime as rt





class onnx_backend():
    def __init__(self,path_model="model.onnx"):
        ###### ONNX ############
        self.model_path = path_model
        self.sess = rt.InferenceSession(self.model_path,providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
        # Get input details
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.class_names = ['Broken','Foreign','Insect','Mold']




    def preprocess_and_decode(self,img_str, new_shape=[224,224]):
        img_str = img_str.replace("/","_").replace("+","-")
        img = tf.io.decode_base64(img_str)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, new_shape, method=tf.image.ResizeMethod.BILINEAR)
        # if you need to squeeze your input range to [0,1] or [-1,1] do it here
        return img
    def onnx_inference(self,image):
        #img = tf.keras.preprocessing.image.load_img('broken.png', target_size=(224, 224))
        img = self.preprocess_and_decode(image)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        input_data = np.array([img_array])
        # Run inference
        outputs = self.sess.run(None, {self.input_name: input_data})

        # Get the output data
        output_name = self.sess.get_outputs()[0].name
        output_data = outputs[0]

        # Process the output data (e.g., print results, make decisions)
        #print(f"Output for {input_name}: {output_data}")

        #generate argmax for predictions
        class_id = np.argmax(output_data, axis = 1)
        #print(class_id)
        # transform classes number into classes name
        #print(class_names[class_id.item()])
        return self.class_names[class_id.item()]

