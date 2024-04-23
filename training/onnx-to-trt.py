import tensorrt as trt
import onnx

# Set TRT_LIBRARY if needed
#export TRT_LIBRARY=/path/to/tensorrt/lib/libnvinfer.so

onnx_model_path = "model.onnx"
logger = trt.Logger(trt.Logger.VERBOSE)

builder = trt.Builder(logger)
network = builder.create_network()

parser = trt.OnnxParser(network, logger)
if not parser.parse(onnx.load(onnx_model_path).SerializeToString()):
    for error in range(parser.num_errors):
        print(parser.get_error(error))
    raise RuntimeError("Failed to parse ONNX model")

# ... (Optional configuration)

builder.set_flag(trt.BuilderFlag.FP16_MODE)

engine = builder.build_engine(network)

# ... (Optional save engine)
engine_path = "model.plan"
with open(engine_path, "wb") as f:
    f.write(engine.serialize())
