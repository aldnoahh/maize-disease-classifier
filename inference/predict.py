import tritonclient.grpc as grpc_client
from tritonclient.utils import triton_to_np_dtype
import numpy as np
import tensorflow as tf

MODEL_NAME = "resnet"
MODEL_VERSION = "1"
model_height = 224
model_width = 224


DATA_TYPE = "FP32"

triton_url = "localhost:801"
triton_client = grpc_client.InferenceServerClient(url=triton_url)


def preprocess_and_decode(img_str, new_shape=[224,224]):
    img = tf.io.decode_base64(img_str)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, new_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    # if you need to squeeze your input range to [0,1] or [-1,1] do it here
    return img

def triton_inference(image):
	
	image = preprocess_and_decode(image)
    
    input_name = "input_4"
    output_name = "dense_24"  # Adjust based on your model's output names

    input_shape = (1, 3, model_height, model_width)  # Adjust dimensions based on your model
    input_dtype = triton_to_np_dtype(DATA_TYPE)  # Use the appropriate data type
    input_data = np.array(image, dtype=input_dtype)
    inputs = [grpc_client.InferInput(input_name, input_shape, DATA_TYPE)]
    inputs[0].set_data_from_numpy(input_data)
    
    outputs = [grpc_client.InferRequestedOutput(output_name)]

    results = triton_client.infer(MODEL_NAME, inputs, outputs=outputs, model_version=MODEL_VERSION)        
    
    output_stream = [results.as_numpy(output_name)]



    return output_stream
