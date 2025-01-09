from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import io
from .ensemble import (
    FluxEnsemble,
)
from .interp_globals import post_double_stream, post_single_stream
import uvicorn

app = FastAPI()

ensemble = FluxEnsemble(use_schnell=False)


class SampleRequest(BaseModel):
    prompts: List[str]
    width: int = 512
    height: int = 512
    sample_steps: int = 3
    capture_double_layers: List[int] = []
    capture_single_layers: List[int] = []
    capture_timesteps: List[int] = []


@app.post("/sample")
async def generate_images(request: SampleRequest):
    if len(request.prompts) == 0:
        raise HTTPException(status_code=400, detail="At least one prompt is required.")

    images = []

    with post_single_stream.capture(*request.capture_single_layers) as reaped_single, post_double_stream.capture(
        *request.capture_double_layers) as reaped_double:
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

    return {"images": images,
            "debug_double": {k: {k2: v2.tolist() for k2, v2 in v.items()} for k, v in reaped_double.items()},
            "debug_single": {k: v.tolist() for k, v in reaped_single.items()}}


# Run the app if this file is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
