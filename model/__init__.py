from .dflash import DFlashDraftModel
from .eagle3 import Eagle3DraftModel
from .quant import (
    FakeQuantize,
    compute_all_qparams,
    configure_fake_quants,
    export_qparams,
    format_quant_summary,
    load_qparams,
    quant_summary,
    set_enabled as set_activation_quant_enabled,
    set_observing as set_activation_quant_observing,
)
from .calibration import (
    CalibrationDataReader,
    calibrate_dflash_activations,
    parse_eval_tasks,
)
from .utils import (
    apply_final_logit_softcapping,
    apply_logit_processing,
    compute_target_lm_logits,
    embed_target_input_ids,
    extract_context_feature,
    get_final_logit_softcapping,
    get_gemma4_embedding_scale,
    get_input_embeddings_module,
    get_logit_scale,
    get_lm_head_module,
    get_model_text_config,
    load_and_process_dataset,
    sample,
)
