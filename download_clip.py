from CLIP.clip import load, available_models


if __name__ == '__main__':
    models = available_models()
    for model in models:
        model_loaded, preprocess = load(model)
        print(model_loaded)
        