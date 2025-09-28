from flask import Flask, request, jsonify, send_file
import torch
from torchvision import transforms
import io
from flask import render_template
import segmentation_models_pytorch as smp
import numpy as np
import tifffile
from PIL import Image

model = smp.Unet(
    encoder_name="resnet50",
    classes=1,
    activation="sigmoid",
    decoder_use_batchnorm=True,
    in_channels=5,
)

# Load only weights
state_dict = torch.load(
    r"C:\Users\ABDELRAHMAN\Desktop\intern gui\unet_resnet_best_model.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)
model.eval()


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = tifffile.imread(io.BytesIO(img_bytes))

    img = np.array(img)
    if img.ndim != 3 or img.shape[2] < 12:
        return jsonify({"error": "Image does not have enough bands"}), 400

    # Select bands
    band_1 = img[..., 11]
    band_2 = img[..., 10]
    band_3 = img[..., 7]
    band_4 = img[..., 6]
    band_5 = img[..., 5]

    # Stack and prepare tensor
    image_out = np.stack([band_1, band_2, band_3, band_4, band_5], axis=-1)
    image_out = torch.from_numpy(image_out).float().permute(2, 0, 1).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(image_out)                     # (1, 1, H, W)
        output = output.squeeze().cpu().numpy()       # (H, W)
        output = (output > 0.5).astype("uint8") * 255 # Binary mask

    # Convert mask to PNG
    mask_img = Image.fromarray(output)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

@app.route("/band/<int:index>", methods=["POST"])
def get_band(index):
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    import tifffile, numpy as np, io
    from PIL import Image

    file = request.files["file"]
    img_bytes = file.read()
    img = tifffile.imread(io.BytesIO(img_bytes))

    if img.ndim != 3 or img.shape[2] <= index:
        return jsonify({"error": f"Image does not have band {index}"}), 400

    band = img[..., index].astype("float32")

    # Normalize to 0–255 for visibility
    band = (255 * (band - band.min()) / (band.max() - band.min() + 1e-8)).astype("uint8")

    band_img = Image.fromarray(band)
    buf = io.BytesIO()
    band_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
