from flask import Flask, request, jsonify
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import os
import uuid
import graycode
import math
import hashlib
import re
import whisper

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
DECIMAL_KEEP = 10**0  
embeddings = {}  # Store {user_id: {"xored_hex": ..., "pin_hash": ...}}
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

n_samples = 16000
n_segments = 5
gpu = torch.cuda.is_available()

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck

# Load RawNet3 model globally
model = RawNet3(
    Bottle2neck,
    model_scale=8,
    context=True,
    summed=True,
    encoder_type="ECA",
    nOut=256,
    out_bn=False,
    sinc_stride=10,
    log_sinc=True,
    norm_sinc="mean",
    grad_mult=1,
)
if gpu:
    model = model.to("cuda")
model.load_state_dict(
    torch.load("./models/weights/model.pt", map_location="cpu")["model"]
)
model.eval()

# Load Whisper small model from Vercel Blob Storage
# def load_whisper_model(blob_url):
#     model_path = "./whisper_small_model.bin"
#     if not os.path.exists(model_path):
#         response = requests.get(blob_url, stream=True)
#         with open(model_path, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 if chunk:
#                     f.write(chunk)
#     whisper_model = whisper.load_model("small", download_root="./")
#     return whisper_model

# Load mô hình khi ứng dụng khởi động
whisper_model = whisper.load_model("large")

NUMBER_MAP = {
    "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4",
    "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "0": "0"
}

def save_hex_embeddings(key, xored_hex, pin_hash, filename):
    with open(filename, "a") as f:
        f.write(f"{key}\t{xored_hex}\t{pin_hash}\n")

def load_hex_embeddings(filename):
    if os.path.exists(filename):
        embeddings = {}
        with open(filename, "r") as f:
            for line in f:
                key, xored_hex, pin_hash = line.strip().split("\t")
                embeddings[key] = {"xored_hex": xored_hex, "pin_hash": pin_hash}
        return embeddings
    return {}

def jaccard_similarity(hex1, hex2):
    bin1 = bin(int(hex1, 16))[2:].zfill(max(len(bin(int(hex1, 16))[2:]), len(bin(int(hex2, 16))[2:])))
    bin2 = bin(int(hex2, 16))[2:].zfill(max(len(bin(int(hex2, 16))[2:]), len(bin(int(hex1, 16))[2:])))
    intersection = sum(c1 == c2 == "1" for c1, c2 in zip(bin1, bin2))
    union = sum(c1 == "1" or c2 == "1" for c1, c2 in zip(bin1, bin2))
    return intersection / union if union != 0 else 0.0

def float_to_gray_with_sign(value, decimal_keep, num_bits=32):
    sign_bit = "0" if value >= 0 else "1"
    exponent = int(math.log10(decimal_keep))
    abs_value = abs(round(value, exponent)) * decimal_keep
    int_value = int(abs_value)
    gray_value = graycode.tc_to_gray_code(int_value)
    value_63bit = f"{gray_value:b}".zfill(num_bits)[:num_bits]
    return sign_bit + value_63bit

def embedding_to_gray_with_sign(embedding_np, decimal_keep):
    return "".join(float_to_gray_with_sign(x, decimal_keep) for x in embedding_np)

def binary_to_hex(binary_string):
    if len(binary_string) % 4 != 0:
        binary_string = binary_string.zfill(len(binary_string) + (4 - len(binary_string) % 4))
    return hex(int(binary_string, 2))[2:].upper()

def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()

def pin_to_binary(pin):
    pin_int = int(pin)
    pin_binary = bin(pin_int)[2:].zfill(20)
    return pin_binary

def xor_with_pin(binary_str, pin):
    pin_binary = pin_to_binary(pin)
    repeated_pin = (pin_binary * (len(binary_str) // len(pin_binary) + 1))[:len(binary_str)]
    xored = "".join("1" if b1 != b2 else "0" for b1, b2 in zip(binary_str, repeated_pin))
    return xored

def extract_speaker_embd(model, fn: str, n_samples: int, n_segments: int, gpu: bool) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(fn)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = F.resample(audio, sample_rate, 16000)
    audio = audio.squeeze(0)
    if len(audio) < n_samples:
        shortage = n_samples - len(audio)
        audio = torch.nn.functional.pad(audio, (0, shortage), mode="constant")
    audios = []
    startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
    for asf in startframe:
        audios.append(audio[int(asf): int(asf) + n_samples])
    audios = torch.stack(audios, dim=0)
    if gpu:
        audios = audios.cuda()
    with torch.no_grad():
        embeddings = model(audios)
        embedding = embeddings.mean(dim=0)
    return embedding

def transcribe_and_check_numbers(audio_path, expected_numbers):
    global whisper_model  
    result = whisper_model.transcribe(audio_path, language="vi")
    transcription = result["text"].lower().strip()
    print(f"Transcription: {transcription}")

    words = transcription.split()
    transcribed_digits = ""
    for word in words:
        if word in NUMBER_MAP:
            transcribed_digits += NUMBER_MAP[word]
    
    if len(transcribed_digits) != 6 or transcribed_digits != expected_numbers:
        transcribed_digits = "".join(re.findall(r'\d', transcription))
    
    return transcribed_digits == expected_numbers

@app.route('/register', methods=['POST'])
def register():
    audio_file = request.files.get('audio')
    user_id = request.form.get('user_id')
    pin = request.form.get('pin')
    
    if not user_id or not pin or not audio_file:
        return jsonify({"message": "user_id, pin, and audio are required"}), 400
    
    if not pin.isdigit() or len(pin) != 6:
        return jsonify({"message": "PIN must be a 6-digit number"}), 400

    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_audio_{uuid.uuid4()}.wav")
    audio_file.save(audio_path)
    
    embedding = extract_speaker_embd(model, audio_path, n_samples, n_segments, gpu)
    embedding_np = embedding.cpu().numpy()
    binary_representation = embedding_to_gray_with_sign(embedding_np.ravel(), DECIMAL_KEEP)
    xored_binary = xor_with_pin(binary_representation, pin)
    xored_hex = binary_to_hex(xored_binary)
    pin_hash = hash_pin(pin)
    
    embeddings[user_id] = {"xored_hex": xored_hex, "pin_hash": pin_hash}
    save_hex_embeddings(user_id, xored_hex, pin_hash, "./hex_embeddings.json")
    os.remove(audio_path)
    
    return jsonify({"message": f"Registration successful for user_id {user_id}", "user_id": user_id}), 200

@app.route('/verify', methods=['POST'])
def verify():
    global embeddings
    embeddings = load_hex_embeddings("./hex_embeddings.json")
    audio_file = request.files.get('audio')
    pin = request.form.get('pin')
    random_number = request.form.get('random_number')
    
    if not audio_file or not pin or not random_number:
        return jsonify({"message": "audio, pin, and random_number are required"}), 400
    
    if not pin.isdigit() or len(pin) != 6:
        return jsonify({"message": "PIN must be a 6-digit number"}), 400
    
    if not random_number.isdigit() or len(random_number) != 6:
        return jsonify({"message": "random_number must be a 6-digit number"}), 400

    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_audio_{uuid.uuid4()}.wav")
    audio_file.save(audio_path)
    
    if not transcribe_and_check_numbers(audio_path, random_number):
        os.remove(audio_path)
        return jsonify({"message": f"Spoken numbers do not match the expected sequence: {random_number}", "random_number": random_number}), 400

    embedding = extract_speaker_embd(model, audio_path, n_samples, n_segments, gpu)
    embedding_np = embedding.cpu().numpy()
    input_binary = embedding_to_gray_with_sign(embedding_np.ravel(), DECIMAL_KEEP)
    input_hex = binary_to_hex(input_binary)
    
    input_pin_hash = hash_pin(pin)
    matches = []
    
    candidates = {uid: data for uid, data in embeddings.items() if data["pin_hash"] == input_pin_hash}
    
    for user_id, data in candidates.items():
        stored_xored_hex = data["xored_hex"]
        stored_xored_binary = bin(int(stored_xored_hex, 16))[2:].zfill(len(input_binary))
        recovered_binary = xor_with_pin(stored_xored_binary, pin)
        recovered_hex = binary_to_hex(recovered_binary)
        
        similarity = jaccard_similarity(input_hex, recovered_hex)
        if similarity > 0.35:
            matches.append({"user_id": user_id})
    
    os.remove(audio_path)
    if matches:
        print(matches)
        print(similarity)
        return jsonify({"message": "Verification successful", "matches": matches, "random_number": random_number}), 200
    
    all_matches = [
        {"user_id": uid, 
         "similarity": jaccard_similarity(input_hex, binary_to_hex(xor_with_pin(bin(int(data["xored_hex"], 16))[2:].zfill(len(input_binary)), pin))), 
         "pin_match": data["pin_hash"] == input_pin_hash}
        for uid, data in embeddings.items()
    ]
    print("All matches (for debugging):", all_matches)
    return jsonify({"message": "Voice or PIN does not match the registered voice and PIN", "matches": [], "random_number": random_number}), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port, debug=True)
