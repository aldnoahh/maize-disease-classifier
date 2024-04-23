import onnx
import onnx_tensorrt

# Load the ONNX model
model = onnx.load("model.onnx")

# Create a TensorRT engine from the ONNX model
engine = onnx_tensorrt.InferenceSession(model)

# Save the TensorRT engine
engine.save("model.trt")

# Load the TensorRT engine
engine = onnx_tensorrt.InferenceSession("model.trt")

# Run inference on the TensorRT engine
output = engine.run(["input"], {"input": input_data})