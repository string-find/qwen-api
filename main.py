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
    """
    Detect objects in an image using Qwen2.5-VL-7B-Instruct.

    Args:
        file (UploadFile): The uploaded image file.
        classes (str): Comma-separated list of object classes (e.g., "person,car").

    Returns:
        dict: JSON with detected objects and their bounding boxes.
    """
    # Parse the classes parameter
    class_list = [cls.strip() for cls in classes.split(",")]

    # Read and preprocess the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize image to reduce memory usage (optional, adjust as needed)
    max_size = 640
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Create prompt for object detection
    prompt = (
        f"Detect the following objects in the image: {', '.join(class_list)}. "
        "Provide their bounding boxes in JSON format as a list of dictionaries, "
        "each with 'class' and 'box' keys, where 'box' is [x_min, y_min, x_max, y_max]."
    )

    # Prepare input messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Parse the output into JSON
    try:
        detections = json.loads(output_text)
        if not isinstance(detections, list):
            detections = [detections]
    except json.JSONDecodeError:
        detections = []  # Fallback if JSON parsing fails

    return {"detections": detections}

# Simple root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Qwen2.5-VL-7B-Instruct Object Detection API is running"}
