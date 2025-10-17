from detectron2.config import CfgNode as CN


def add_language_cfg(cfg: CN) -> None:
    """
    Extend Detectron2 config with language-related options for LaMI-DETR.
    """
    cfg.MODEL.LANGUAGE = CN()
    cfg.MODEL.LANGUAGE.USE_TPA = False
    cfg.MODEL.LANGUAGE.TEXT_EMB_PATH = ""  # e.g. "dataset/metadata/multi_phrase_embeddings.npy"
    cfg.MODEL.LANGUAGE.NUM_PROTOTYPES = 4
    cfg.MODEL.LANGUAGE.HIDDEN_DIM = 256
    cfg.MODEL.LANGUAGE.DROPOUT = 0.1
    cfg.MODEL.LANGUAGE.TAU = 0.1
