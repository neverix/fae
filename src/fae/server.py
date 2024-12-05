from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import io
from PIL import Image
import numpy as np
from .ensemble import (
    FluxEnsemble,
)
import uvicorn

app = FastAPI()

ensemble = FluxEnsemble()


class SampleRequest(BaseModel):
    prompts: List[str]
    width: int = 512
    height: int = 512
    sample_steps: int = 3


@app.post("/sample")
async def generate_images(request: SampleRequest):
    if len(request.prompts) == 0:
        raise HTTPException(status_code=400, detail="At least one prompt is required.")

    images = []

    for i, img in enumerate(
        ensemble.sample(
            request.prompts,
            width=request.width,
            height=request.height,
            sample_steps=request.sample_steps,
        )
    ):
        # Convert the generated PIL image to bytes for response
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        images.append(img_base64)

    return {"images": images}


# Run the app if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
