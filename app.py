import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import pydicom
import numpy as np

import torch
import torch.nn.functional as F

from datetime import datetime
import uuid
from pathlib import Path

from flask import (Flask, request, jsonify, send_file)

from flask_cors import CORS
from flask_restx import (Api, Resource, fields)

from rt_utils import RTStructBuilder
import time


def now():
    return time.perf_counter()


# =====================================================
# Flask
# =====================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

api = Api(
    app,
    version="2.0.0",
    title="CHAOS RTSTRUCT API",
    description="DICOM Series Segmentation To RTSTRUCT",
    doc="/swagger/"
)

# =====================================================
# Device
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# Load Model
# =====================================================

MODEL_PATH = "UNET_CHAOS.pt"

model = torch.jit.load(
    MODEL_PATH,
    map_location=device
)

model.eval()
BATCH_SIZE = 32   # 分批次预测，避免OOM

print("Model loaded")


# =====================================================
# DICOM Utils
# =====================================================

def load_dicom_series(dicom_dir):

    slices = []
    for f in os.listdir(dicom_dir):
        path = os.path.join(dicom_dir, f)

        try:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "ImagePositionPatient"):
                slices.append(ds)
        except:
            pass

    if len(slices) == 0:

        raise Exception("No valid DICOM slices found.")

    slices.sort(
        key=lambda s:
        float(
            s.ImagePositionPatient[2]
        )
    )

    return slices


def dicom_to_hu(ds):

    img = ds.pixel_array.astype(np.int16)

    slope = float(
        getattr(ds, "RescaleSlope", 1)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0)
    )

    img = img * slope + intercept

    return img.astype(np.int16)


def normalize_ct(img):

    min_val = -100
    max_val = 250

    img = np.clip(img, min_val, max_val)

    img = (img - min_val) / (max_val - min_val)

    return img.astype(np.float32)


def preprocess_volume(slices):

    tensors = []

    for ds in slices:

        hu_img = dicom_to_hu(ds)

        hu_img = normalize_ct(hu_img)

        tensor = torch.from_numpy(hu_img).float()

        tensors.append(tensor)

    volume = torch.stack(tensors, dim=0)

    volume = volume.unsqueeze(1)

    volume = F.interpolate(
        volume,
        size=(256,256),
        mode="bilinear",
        align_corners=False
    )

    return volume


def predict_volume(slices):

    height = slices[0].Rows
    width = slices[0].Columns

    volume_tensor = preprocess_volume(slices)

    volume_tensor = volume_tensor.to(device)

    pred_list = []

    with torch.no_grad():

        total_slice = volume_tensor.shape[0]

        for start in range(0, total_slice, BATCH_SIZE):

            end = min(start + BATCH_SIZE, total_slice)

            batch_tensor = volume_tensor[start:end]

            pred_batch = model(batch_tensor)

            pred_list.append(pred_batch.cpu())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    pred = torch.cat(pred_list, dim=0)

    pred = pred.squeeze(1).numpy()

    depth = pred.shape[0]

    mask_3d = np.zeros(
        (height, width, depth),
        dtype=bool
    )

    for i in range(depth):

        mask = (pred[i] > 0.5).astype(np.uint8)

        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_3d[:, :, i] = mask

    return mask_3d


def predict_volume_all(dicom_dir):

    slices = load_dicom_series(dicom_dir)

    print(
        f"Predicting "
        f"{len(slices)} slices "
        f"in one batch..."
    )

    return predict_volume(slices)


def generate_rtstruct(
    dicom_dir,
    output_path,
    roi_name="Liver_AI"
):

    mask_3d = predict_volume_all(dicom_dir)

    rtstruct = (
        RTStructBuilder.create_new(
            dicom_series_path=dicom_dir
        )
    )

    rtstruct.add_roi(
        mask=mask_3d,
        name=roi_name,
        color=[255, 0, 0]
    )

    rtstruct.save(
        output_path
    )


# =====================================================
# Swagger
# =====================================================

health_response = api.model(
    "HealthResponse",
    {
        "status": fields.String(),
        "timestamp": fields.String()
    }
)

error_response = api.model(
    "ErrorResponse",
    {
        "status": fields.String(),
        "message": fields.String()
    }
)

# =====================================================
# Namespace
# =====================================================

ns_info = api.namespace(
    "info",
    description="Info"
)

ns_infer = api.namespace(
    "infer",
    description="Inference"
)

# =====================================================
# Health
# =====================================================

@ns_info.route("/health")
class Health(Resource):

    def get(self):

        return {
            "status": "healthy",
            "timestamp":
            datetime.utcnow().isoformat() + "Z"
        }


# =====================================================
# Infer
# =====================================================

@ns_infer.route("/")
class Infer(Resource):

    @ns_infer.response(
        200,
        "RTSTRUCT"
    )
    @ns_infer.response(
        400,
        "Bad Request",
        error_response
    )
    @ns_infer.response(
        500,
        "Internal Error",
        error_response
    )
    def post(self):

        total_start = now()
        today = datetime.now().strftime("%Y-%m-%d")

        upload_dir = (Path("uploads") / today / uuid.uuid4().hex)

        upload_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = request.files.getlist(
                "files"
            )
            print(len(files))

            if len(files) == 0:

                return {
                    "status": "error",
                    "message":
                    "No dicom uploaded"
                }, 400

            print(
                f"Receive "
                f"{len(files)} slices"
            )

            for file in files:

                save_path = os.path.join(
                    upload_dir,
                    file.filename
                )

                file.save(
                    save_path
                )

            rtstruct_path = os.path.join(
                upload_dir,
                "RTSTRUCT_AI.dcm"
            )

            generate_rtstruct(
                dicom_dir=upload_dir,
                output_path=rtstruct_path,
                roi_name="Liver_AI"
            )

            return send_file(
                rtstruct_path,
                mimetype="application/dicom",
                as_attachment=False,
                download_name="RTSTRUCT_AI.dcm"
            )

        except Exception as e:

            return {
                "status": "error",
                "message": str(e)
            }, 500

        finally:

            total_cost = now() - total_start

            print(
                f"[TIME] Total Request: "
                f"{total_cost:.3f}s"
            )


# =====================================================
# Home
# =====================================================

@app.route("/")
def home():

    return jsonify({

        "name":
        "CHAOS RTSTRUCT API",

        "model":
        "UNET_CHAOS.pt",

        "swagger":
        "/swagger/",

        "endpoint":
        "POST /infer/"
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )