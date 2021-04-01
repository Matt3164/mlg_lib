import joblib

LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
               "sandal", "shirt", "sneaker", "bag", "ankle boot"]
memory = joblib.Memory("/tmp/fmnist_bench")