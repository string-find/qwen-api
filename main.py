from fastapi import FastAPI, File, UploadFile
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import json

# Initialize FastAPI app
app = FastAPI()

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Qwen2.5-VL-7B-Instruct model and processor
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
try:
    # Load model in half-precision (float16) with flash attention for optimization
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).to(device)
    # Configure processor with pixel limits for memory efficiency
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    model.eval()  # Set to evaluation mode
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define the object detection endpoint
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), classes: str = "person,car"):
    # Parse classes, remove duplicates
    class_list = list(set([cls.strip() for cls in classes.split(",") if cls.strip()]))
    if not class_list:
        return {"detections": [], "message": "No valid classes provided"}

    # Read and preprocess image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}

    max_size = 640
    original_width, original_height = image.size
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    resized_width, resized_height = image.size
    width_scale = original_width / resized_width
    height_scale = original_height / resized_height

    # Create refined prompt
    prompt = (
        f"Detect the following objects in the image: {', '.join(class_list)}. "
        "Output only a JSON list of detected objects with their bounding boxes, "
        "each as {{'class': 'object_name', 'box': [x_min, y_min, x_max, y_max]}}. "
        "Do not include any other text or explanations."
    )

    # Prepare inputs (unchanged)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate output (unchanged)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Robust JSON parsing
    import re
    match = re.search(r'\[.*\]', output_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            detections = json.loads(json_str)
            if not isinstance(detections, list):
                detections = [detections]
        except json.JSONDecodeError:
            detections = []
    else:
        detections = []

    # Scale bounding boxes to original size (optional)
    for detection in detections:
        box = detection['box']
        detection['box'] = [
            box[0] * width_scale,
            box[1] * height_scale,
            box[2] * width_scale,
            box[3] * height_scale
        ]

    return {"detections": detections}

# Simple root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Qwen2.5-VL-7B-Instruct Object Detection API is running"}
