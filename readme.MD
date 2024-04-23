# Maize Issue Classifier using ResNet50
1. Custom training on provided Maize issue dataset.
2. Multi-class classifcation Task.
3. Train ResNet50 using Keras
4. Optimized inference using ONNX and GPU.
5. FastAPI based REST API endpoint for calling.
6. Containerized solution using Docker.
7. Future? Explore TensorRT with Triton for further optimization. Snippets present.

## API
- Endpoint: `localost:12000/infer`
- Json Payload: `{ "image" : "BASE64 Encoded String" }`
- Resposes: 
a. Successful: {"success": true,"status_code": 200,"message": "Class Name"}
b. Unsucessful: ({"success": false,"status_code": 400,"message": "Unable to Recognize"})
c. Exception: {"success": false, "status_code": 404, "message": "Error in inference: Message"}
