### API
- Endpoint ```localost:12000/infer"
- Json Payload: `{ "image" : "BASE64 Encoded String" }
- Resposes: 
a. Successful: {"success": true,"status_code": 200,"message": "Class Name"}
b. Unsucessful: ({"success": false,"status_code": 400,"message": "Unable to Recognize"})
c. Exception: {"success": false, "status_code": 404, "message": "Error in inference: Message"}
