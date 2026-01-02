import torch
from .utils import atleast_kdim


def get_init_with_noise(model, X, y):
    init = X.clone()
    print("y: ", y)
    p = model(X).argmax(1)
    print("p: ", p)

    while any(p == y):
        init = torch.where(
            atleast_kdim(p == y, len(X.shape)), 
            (X + 0.5*torch.randn_like(X)).clip(0, 1), 
            init)
        p = model(init).argmax(1)
        print("y: ", y)
        print("p: ", p)
    return init