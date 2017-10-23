

class Translation():
    def exponential(velocity):
        return velocity

def action(translation, f):
    def translated(x):
        return f - translation

def product(f, g):
    return np.dot(f, g)

def pairing(f, g):
    return np.dot(f, g[::2])


