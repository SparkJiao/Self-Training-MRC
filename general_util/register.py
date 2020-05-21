reader_dict = {}
model_dict = {}


def register_reader(name):
    def call(cls):
        reader_dict[name] = cls

    return call


def register_model(name):
    def call(cls):
        model_dict[name] = cls

    return call
