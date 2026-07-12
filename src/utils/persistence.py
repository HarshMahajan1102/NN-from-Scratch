import pickle

def save_model(model, path):

    params = []
    for l in model.layers:
        if hasattr(l, "weight"):
            params.append({"weight": l.weight, "biases": l.biases})
        else:
            params.append(None)

    with open(path, "wb") as f:
        pickle.dump(params, f)

def load_model(model, path):

    with open(path, "rb") as f:
        params = pickle.load(f)

    for l, param in zip(model.layers, params):
        if param is not None:
            l.weight = param["weight"]
            l.biases = param["biases"]
