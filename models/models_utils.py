import json
import os
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve
import graycode
from scipy.interpolate import interp1d
import math

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck


def save_hex_embeddings(hex_embeddings, filename):
    with open(filename, "w") as f:
        json.dump(hex_embeddings, f)


def load_hex_embeddings(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            hex_embeddings = json.load(f)
        return hex_embeddings
    else:
        return {}


def float_to_gray_with_sign(value, decimal_keep, num_bits=2):
    """Chuyển đổi float thành Gray code với 1 bit dấu và 63 bit giá trị."""
    sign_bit = "0" if value >= 0 else "1"  # 0 cho dương, 1 cho âm
    exponent = int(math.log10(decimal_keep))

    abs_value = (
        abs(round(value, exponent)) * decimal_keep
    )  # Nhân với hằng số để loại bỏ phần thập phân
    int_value = int(abs_value)  # Chuyển thành số nguyên
    gray_value = graycode.tc_to_gray_code(int_value)  # Chuyển thành Gray code
    value_63bit = f"{gray_value:b}".zfill(num_bits)[:num_bits]
    return sign_bit + value_63bit  # 1 bit dấu + 63 bit giá trị Gray code


def embedding_to_gray_with_sign(embedding_np, decimal_keep):
    """Chuyển đổi toàn bộ embedding thành chuỗi Gray code với 63 bit giá trị và 1 bit dấu."""
    return "".join(float_to_gray_with_sign(x, decimal_keep) for x in embedding_np)


def jaccard_similarity(hex1, hex2):
    """Tính độ tương đồng Jaccard giữa hai chuỗi hexadecimal."""
    bin1 = bin(int(hex1, 16))[2:]
    bin2 = bin(int(hex2, 16))[2:]

    # Đảm bảo độ dài bằng nhau bằng cách đệm thêm các số 0
    max_len = max(len(bin1), len(bin2))
    bin1 = bin1.zfill(max_len)
    bin2 = bin2.zfill(max_len)

    # Tính toán độ tương đồng Jaccard
    intersection = sum(c1 == c2 == "1" for c1, c2 in zip(bin1, bin2))  # Số bit 1 chung
    union = sum(
        c1 == "1" or c2 == "1" for c1, c2 in zip(bin1, bin2)
    )  # Tổng số bit 1 (bit giống hoặc khác)

    if union == 0:  # Trường hợp tránh chia cho 0
        return 0.0
    else:
        return intersection / union


def read_trial_list(trial_file):
    trials = []
    with open(trial_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label = int(parts[0])
                path1 = parts[1]
                path2 = parts[2]
                trials.append((label, path1, path2))
    return trials


def extract_speaker_embd(
    model, fn: str, n_samples: int, n_segments: int = 10, gpu: bool = False
) -> torch.Tensor:
    audio, sample_rate = sf.read(fn)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Chỉ lấy kênh đầu tiên

    if sample_rate != 16000:
        raise ValueError(
            f"RawNet3 supports 16k sampling rate only. Input data's sampling rate is {sample_rate}."
        )

    if len(audio) < n_samples:
        shortage = n_samples - len(audio) + 1
        audio = np.pad(audio, (0, shortage), "wrap")

    audios = []
    startframe = np.linspace(0, len(audio) - n_samples, num=n_segments)
    for asf in startframe:
        audios.append(audio[int(asf) : int(asf) + n_samples])

    audios = np.stack(audios, axis=0).astype(np.float32)
    audios = torch.from_numpy(audios)
    if gpu:
        audios = audios.to("cuda")

    with torch.no_grad():
        embeddings = model(audios)
        # embedding = embeddings.mean(dim=0)

    return embeddings


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # Tìm điểm mà FPR và FNR giao nhau
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # eer = interp1d(fpr, 1 - tpr)(eer_threshold)
    return eer


def save_embeddings(embeddings, filename):
    # embeddings là dictionary với key là file_path và value là numpy array
    np.savez_compressed(filename, **embeddings)


def load_embeddings(filename):
    if os.path.exists(filename):
        data = np.load(filename)
        embeddings = {file_path: data[file_path] for file_path in data.files}
        return embeddings
    else:
        return {}


def main_hashing(embedding_np, decimal_keep):
    binary_representation = embedding_to_gray_with_sign(
        embedding_np.ravel(), decimal_keep
    )
    hex_representation = hex(eval("0b" + binary_representation))
    # binary_representation = int(binary_representation, 2)
    # hex_representation = hex(binary_representation)
    return hex_representation


def main():
    import pandas as pd  # Thêm import pandas để sử dụng dataframe

    n_segments = 5
    n_samples = 16000
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
    gpu = False
    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    model.load_state_dict(
        torch.load(
            "./models/weights/model.pt",
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    model.eval()
    print("RawNet3 initialised & weights loaded!")

    # Đường dẫn tới thư mục dữ liệu và tệp thử nghiệm
    data_dir = "./test_data/wav"  # Thay bằng đường dẫn tới thư mục VoxCeleb1 của bạn
    trial_file = "./trials/test_speaker.txt"  # Đường dẫn tới tệp veri_test.txt

    # Đọc danh sách cặp thử nghiệm
    trials = read_trial_list(trial_file)

    # Lấy danh sách các tệp âm thanh cần trích xuất embedding
    file_set = set()
    for label, path1, path2 in trials:
        file_set.add(path1)
        file_set.add(path2)
    file_list = list(file_set)

    # Tên file lưu trữ embeddings gốc
    embeddings_file = "./embeddings.npz"

    # Tải embeddings đã lưu nếu có
    embeddings = load_embeddings(embeddings_file)

    # Trích xuất embedding cho các tệp chưa có embeddings
    for file_path in tqdm(file_list):
        if file_path in embeddings:
            continue  # Đã có embedding, bỏ qua
        full_path = os.path.join(data_dir, file_path)
        embedding = extract_speaker_embd(
            model,
            fn=full_path,
            n_samples=n_samples,
            n_segments=n_segments,
            gpu=gpu,
        )
        embedding_np = embedding.cpu().numpy()
        embeddings[file_path] = embedding_np  # Lưu embedding gốc

        # Lưu embeddings sau mỗi lần thêm mới
        save_embeddings(embeddings, embeddings_file)
    with open("all_embeddings.txt", "w") as f:
        for file_path, embedding_np in embeddings.items():
            f.write(f"{file_path}\t{','.join(map(str, embedding_np))}\n")

    # Tính EER dựa trên cosine similarity của embedding gốc
    scores_cosine = []
    labels_cosine = []
    for label, path1, path2 in tqdm(trials):
        if path1 in embeddings and path2 in embeddings:
            embedding1 = embeddings[path1]
            embedding2 = embeddings[path2]

            # Tính cosine similarity
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            similarity = float(
                cos(torch.from_numpy(embedding1), torch.from_numpy(embedding2))
            )

            scores_cosine.append(similarity)
            labels_cosine.append(label)
        else:
            print(f"Missing embeddings for {path1} or {path2}, skipping this pair.")

    if len(scores_cosine) > 0:
        eer_cosine = compute_eer(labels_cosine, scores_cosine)
        print(f"EER based on Cosine Similarity: {eer_cosine * 100:.2f}%")
    else:
        print("No scores to compute EER based on Cosine Similarity.")

    # Danh sách lưu kết quả EER cho mỗi DECIMAL_KEEP
    eer_results = []

    # Vòng lặp qua các giá trị DECIMAL_KEEP từ 10^8 đến 10^15
    for exponent in range(1, 2):  # Từ 8 đến 15
        decimal_keep = 10**exponent
        print(f"\nProcessing with DECIMAL_KEEP = 10^{exponent}")

        hex_embeddings_file = f"./hex_embeddings_quantized_round_{exponent}.json"

        hex_embeddings = load_hex_embeddings(hex_embeddings_file)

        # Trích xuất hex embeddings cho các tệp chưa có
        for file_path in tqdm(file_list):
            if file_path in hex_embeddings:
                continue  # Đã có hex embedding, bỏ qua
            embedding_np = embeddings[file_path]

            hex_representation = main_hashing(embedding_np, decimal_keep)
            hex_embeddings[file_path] = hex_representation

            # Lưu hex_embeddings sau mỗi lần thêm mới
            save_hex_embeddings(hex_embeddings, hex_embeddings_file)

        # Tính toán khoảng cách Hamming và EER
        scores = []
        labels = []
        for label, path1, path2 in tqdm(trials):
            if path1 in hex_embeddings and path2 in hex_embeddings:
                hex1 = hex_embeddings[path1]
                hex2 = hex_embeddings[path2]

                similarity = jaccard_similarity(hex1, hex2)
                scores.append(similarity)
                labels.append(label)
            else:
                print(
                    f"Missing hex embeddings for {path1} or {path2}, skipping this pair."
                )

        # Tính EER
        if len(scores) > 0:
            with open("label_scores.txt", "w") as f:
                json.dump({"labels": labels, "scores": scores}, f)
            eer = compute_eer(labels, scores)
            print(
                f"EER based on Hamming Distance with DECIMAL_KEEP=10^{exponent}: {eer * 100:.2f}%"
            )
            eer_results.append({"DECIMAL_KEEP": exponent, "EER_Hamming": eer * 100})
        else:
            print(f"No scores to compute EER for DECIMAL_KEEP=10^{exponent}.")

    # Tạo dataframe từ kết quả EER
    df = pd.DataFrame(eer_results)
    print("\nKết quả EER cho các giá trị DECIMAL_KEEP:")
    print(df)


if __name__ == "__main__":
    main()
