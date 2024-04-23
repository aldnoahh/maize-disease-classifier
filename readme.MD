# Maize Issue Classifier using ResNet50
1. Custom training on provided Maize issue dataset.
2. Multi-class classifcation Task.
3. Train ResNet50 using Keras
4. Optimized inference using ONNX and GPU.
5. FastAPI based REST API endpoint for calling.
6. Containerized solution using Docker.
7. Future? Explore TensorRT with Triton for further optimization. Snippets present.

## Inference
0. Pre-requisites: Must have Docker installed, and docker driver as Nvidia in case of GPUs.
1. Go to inference folder and build image
```
git clone https://github.com/aldnoahh/maize-issue-classifier.git
cd maize-issue-classifier/inference
sudo docker build -t inference .
```
2. Run container using the image
```
sudo docker run --rm -dit --name maize -p12000:12000 --runtime=nvidia --gpus=all inference:latest
```
OR
```
sudo docker run --rm -dit --name maize -p12000:12000 inference:latest
```
Wait for 1-2 mins.

3. Test run </br>
3.1 Check if APIs working
3.1.1 Go to `localhost:12000/docs` on web browser to check Swagger UI
3.1.2 Run 'curl localhost:12000' on terminal. It should return `{"detail":"Not Found"}`
3.2 Test using CURL with a base64 image
```
cd maize-issue-classifier/inference/misc
bash curl-test-broken.sh
```


## API
- Endpoint: `localost:12000/infer`
- Json Payload: `{ "image" : "BASE64 Encoded String" }`
- Resposes: 
a. Successful: {"success": true,"status_code": 200,"message": "Class Name"}
b. Unsucessful: ({"success": false,"status_code": 400,"message": "Unable to Recognize"})
c. Exception: {"success": false, "status_code": 404, "message": "Error in inference: Message"}
