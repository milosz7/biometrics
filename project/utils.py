from insightface.app import FaceAnalysis
import torch


def init_face_analysis_model(config):
    providers = (
        ["CudaExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    model = FaceAnalysis(
        name=config.FACE_ANALYSIS_MODEL,
        providers=providers,
        allowed_modules=config.FACE_ANALYSIS_MODEL_MODULES,
    )
    model.prepare(ctx_id=-1)
    return model
