from flask import Flask, request, send_file, jsonify
import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
import os
import json
from mtcnn import MTCNN
import hashlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__, static_folder="static")
app.config["DATASET_PATH"] = "dataset"
app.config["EMBEDDINGS_DB"] = "embeddings_db.json"

# Initialize FaceNet
facenet = FaceNet()
detector = (
    MTCNN()
)  # Replaced HaarCascade with MTCNN for improved detection on masked faces

# Load existing embeddings or create new
if os.path.exists(app.config["EMBEDDINGS_DB"]):
    with open(app.config["EMBEDDINGS_DB"], "r") as f:
        embeddings_db = json.load(f)
else:
    embeddings_db = {}

# -----------------------------
# Utility functions
# -----------------------------


def detect_mask(face_img):
    # face_img is RGB uint8 160x160x3
    lower = face_img[80:, :, :]  # lower half
    avg_std = np.mean([np.std(lower[:, :, i]) for i in range(3)])
    if avg_std < 25:  # arbitrary threshold; tune based on testing
        return "with_mask"
    else:
        return "without_mask"


def get_face_embedding_and_mask(img):
    # Convert to RGB for MTCNN
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_img)
    if len(detections) == 0:
        return None, None
    # Take the first detected face (assuming one primary face)
    face = detections[0]
    x, y, w, h = face["box"]
    # Extract face region from original BGR image
    face_img = img[y : y + h, x : x + w]
    # Convert to RGB for FaceNet
    face_img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
    face_img_resized = cv.resize(face_img_rgb, (160, 160))
    face_img_exp = np.expand_dims(face_img_resized, axis=0)
    emb = facenet.embeddings(face_img_exp)
    mask_status = detect_mask(face_img_resized)
    return emb[0], mask_status


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# -----------------------------
# Routes
# -----------------------------


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/register", methods=["POST"])
def register():
    """
    Register a new person by uploading one or more images.
    Note: For mask robustness, register images both with and without masks.
    """
    person_name = request.form.get("name")
    files = request.files.getlist("image")

    if not person_name or not files:
        return jsonify({"success": False, "message": "Missing name or image(s)"}), 400

    if person_name not in embeddings_db:
        embeddings_db[person_name] = []

    embs_added = 0
    with_mask_count = 0
    without_mask_count = 0

    for file in files:
        img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_COLOR)
        emb, mask_status = get_face_embedding_and_mask(img)
        if emb is None:
            continue
        embeddings_db[person_name].append(
            {"embedding": emb.tolist(), "mask_status": mask_status}
        )
        embs_added += 1
        if mask_status == "with_mask":
            with_mask_count += 1
        else:
            without_mask_count += 1

    if embs_added == 0:
        return (
            jsonify({"success": False, "message": "No faces detected in any images"}),
            400,
        )

    # Persist to disk
    with open(app.config["EMBEDDINGS_DB"], "w") as f:
        json.dump(embeddings_db, f)

    return jsonify(
        {
            "success": True,
            "message": f"Registered {person_name} with {embs_added} image(s) successfully!",
            "with_mask": with_mask_count,
            "without_mask": without_mask_count,
        }
    )


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Recognize an uploaded face by comparing to registered embeddings.
    """
    if not embeddings_db:
        return jsonify({"success": False, "message": "No registered users yet"}), 400

    file = request.files["image"]
    img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_COLOR)
    emb, input_mask = get_face_embedding_and_mask(img)
    if emb is None:
        return jsonify({"success": False, "message": "No face detected"}), 400

    # Compare with all registered embeddings
    best_name = None
    best_score = 0
    best_emb = None
    threshold = (
        0.6  # Adjust for stricter/looser matching; may need tuning for masked faces
    )

    for name, person_embs in embeddings_db.items():
        for e in person_embs:
            score = cosine_similarity(emb, np.array(e["embedding"]))
            if score > best_score:
                best_score = score
                best_name = name
                best_emb = e

    if best_score < threshold:
        return jsonify({"success": False, "message": "Unknown person"}), 200

    mask_match = best_emb["mask_status"] == input_mask

    return jsonify(
        {
            "success": True,
            "name": best_name,
            "confidence": round(best_score, 2),
            "mask_status": input_mask,
            "mask_match": mask_match,
        }
    )


@app.route("/users", methods=["GET"])
def get_users():
    return jsonify(
        {
            "success": True,
            "total_users": len(embeddings_db),
            "users": list(embeddings_db.keys()),
        }
    )


@app.route("/reset_database", methods=["POST"])
def reset_database():
    embeddings_db.clear()
    with open(app.config["EMBEDDINGS_DB"], "w") as f:
        json.dump(embeddings_db, f)
    return jsonify({"success": True, "message": "Database reset successfully"})


@app.route("/debug_save", methods=["POST"])
def debug_save():
    with open(app.config["EMBEDDINGS_DB"], "w") as f:
        json.dump(embeddings_db, f)
    return jsonify(
        {
            "success": True,
            "message": "Database saved successfully",
            "users": list(embeddings_db.keys()),
        }
    )


@app.route("/user/<name>", methods=["DELETE"])
def delete_user(name):
    if name in embeddings_db:
        del embeddings_db[name]
        with open(app.config["EMBEDDINGS_DB"], "w") as f:
            json.dump(embeddings_db, f)
        return jsonify(
            {"success": True, "message": f"User {name} deleted successfully"}
        )
    else:
        return jsonify({"success": False, "message": "User not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
