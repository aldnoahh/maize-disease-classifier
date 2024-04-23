"""
sudo docker run --rm --runtime=nvidia --shm-size=12g --ulimit memlock=-1 --ulimit stack=67108864 --name triton --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/ubuntu/models:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1
"""
#import tritonclient.grpc as grpc_client
#from tritonclient.utils import triton_to_np_dtype
import numpy as np
import tensorflow as tf
import onnxruntime as rt

class_names = ['Broken','Foreign','Insect','Mold']


MODEL_NAME = "resnet"
MODEL_VERSION = "1"
model_height = 224
model_width = 224
DATA_TYPE = "FP32"
triton_url = "triton:8001"
#triton_client = grpc_client.InferenceServerClient(url=triton_url)

###### ONNX ############
model_path = "model.onnx"
sess = rt.InferenceSession(model_path) #,providers = ['CUDAExecutionProvider'])
# Get input details
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape

def preprocess_and_decode(img_str, new_shape=[224,224]):
    img_str = img_str.replace("/","_").replace("+","-")
    img = tf.io.decode_base64(img_str)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, new_shape, method=tf.image.ResizeMethod.BILINEAR)
    # if you need to squeeze your input range to [0,1] or [-1,1] do it here
    return img

# def triton_inference(image):
	
# 	image = preprocess_and_decode(image)
    
#     input_name = "input_4"
#     output_name = "dense_24"  # Adjust based on your model's output names

#     input_shape = (1, 3, model_height, model_width)  # Adjust dimensions based on your model
#     input_dtype = triton_to_np_dtype(DATA_TYPE)  # Use the appropriate data type
#     input_data = np.array(image, dtype=input_dtype)
#     inputs = [grpc_client.InferInput(input_name, input_shape, DATA_TYPE)]
#     inputs[0].set_data_from_numpy(input_data)
    
#     outputs = [grpc_client.InferRequestedOutput(output_name)]

#     results = triton_client.infer(MODEL_NAME, inputs, outputs=outputs, model_version=MODEL_VERSION)        
    
#     output_stream = [results.as_numpy(output_name)]



#     return output_stream



def onnx_inference(image):
    #img = tf.keras.preprocessing.image.load_img('broken.png', target_size=(224, 224))
    img = preprocess_and_decode(image)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    input_data = np.array([img_array])
    # Run inference
    outputs = sess.run(None, {input_name: input_data})

    # Get the output data
    output_name = sess.get_outputs()[0].name
    output_data = outputs[0]

    # Process the output data (e.g., print results, make decisions)
    #print(f"Output for {input_name}: {output_data}")

    #generate argmax for predictions
    class_id = np.argmax(output_data, axis = 1)
    #print(class_id)
    # transform classes number into classes name
    #print(class_names[class_id.item()])
    return class_names[class_id.item()]

