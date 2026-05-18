from .dflash import DFlashDraftModel
from .eagle3 import Eagle3DraftModel
from .utils import (
    apply_final_logit_softcapping,
    compute_target_lm_logits,
    embed_target_input_ids,
    extract_context_feature,
    get_final_logit_softcapping,
    get_gemma4_embedding_scale,
    get_input_embeddings_module,
    get_lm_head_module,
    get_model_text_config,
    load_and_process_dataset,
    sample,
)
