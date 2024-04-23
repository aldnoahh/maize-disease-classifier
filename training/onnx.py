import onnxruntime as rt
import numpy as np
import tensorflow as tf
import time
class_names = ['Broken','Foreign','Insect','Mold']

st =time.time()
# Load the ONNX model
model_path = "model.onnx"
sess = rt.InferenceSession(model_path,providers = ['CUDAExecutionProvider'])

# Get input details
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape

# Prepare your input data (replace with your actual data)
# Ensure the data shape matches the model's input shape
#input_data = np.random.rand(*input_shape).astype(np.float32)  # Example random input
img = tf.keras.preprocessing.image.load_img('broken.png', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
input_data = np.array([img_array])
# Run inference
outputs = sess.run(None, {input_name: input_data})

# Get the output data
output_name = sess.get_outputs()[0].name
output_data = outputs[0]

# Process the output data (e.g., print results, make decisions)
print(f"Output for {input_name}: {output_data}")

 #generate argmax for predictions
class_id = np.argmax(output_data, axis = 1)
print(class_id)
# transform classes number into classes name
print(class_names[class_id.item()])

print(f"Time taken is {time.time()-st} seconds")