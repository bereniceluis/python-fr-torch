from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

MODEL_MAPPING = {
    "r18": iresnet18,
    "r34": iresnet34,
    "r50": iresnet50,
    "r100": iresnet100,
    "r200": iresnet200,
}

def get_model(name, **kwargs):
    if name not in MODEL_MAPPING:
        raise ValueError("Invalid model name.")

    model_fn = MODEL_MAPPING[name]
    return model_fn(False, **kwargs)
