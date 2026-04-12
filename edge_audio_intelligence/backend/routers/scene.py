"""Scene generation endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from ..models.requests import SceneRequest
from ..models.responses import SceneResponse
from ..services.scene_service import generate_scene
from ..store import run_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=SceneResponse)
def generate(req: SceneRequest):
    """Generate an acoustic scene from configuration.

    Returns a scene_id that can be used with /api/pipeline/run.
    """
    try:
        scene = generate_scene(req)
        scene_id = run_store.store_scene(scene)

        return SceneResponse(
            scene_id=scene_id,
            n_sources=scene.n_sources,
            true_doas=scene.true_doas,
            transcriptions=scene.transcriptions,
            mic_positions=[list(p) for p in scene.mic_positions]
            if scene.mic_positions is not None else [],
            duration_s=req.duration_s,
            sample_rate=scene.sample_rate,
        )
    except Exception as e:
        logger.error(f"Scene generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
