import os, requests, pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OWNER = os.environ.get("GITHUB_OWNER", "vaibhavtamang12")
REPO = os.environ.get("GITHUB_REPO", "kolam")
ASSET_NAME = os.environ.get("GITHUB_ASSET_NAME", "model.pkl")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

MODEL_PATH = "/tmp/model.pkl"

def download_release_asset():
    api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest"
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(api_url, headers=headers)
    r.raise_for_status()
    release = r.json()
    asset = next((a for a in release["assets"] if a["name"] == ASSET_NAME), None)
    if not asset:
        raise Exception(f"Asset {ASSET_NAME} not found in release.")
    url = asset["browser_download_url"]
    r2 = requests.get(url, stream=True)
    r2.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r2.iter_content(1024*1024):
            f.write(chunk)

# --- Load or download ---
if not os.path.exists(MODEL_PATH):
    print("Downloading model from GitHub Release...")
    download_release_asset()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- FastAPI ---
app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    try:
        pred = model.predict([data.features])
        return {"prediction": pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
