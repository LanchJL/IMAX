# multi-path paradigm
# from model.clip_multi_path import CLIP_Multi_Path
# from model.coop_multi_path import COOP_Multi_Path
from model.models import IMAX

def get_model(config, attributes, classes, offset):
    if config.model_name == 'IMAX':
        model = IMAX(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )
    return model