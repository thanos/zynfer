#ifndef ZYNFER_COREML_BRIDGE_H
#define ZYNFER_COREML_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ZYNFER_COREML_OK = 0,
    ZYNFER_COREML_UNAVAILABLE = 1,
    ZYNFER_COREML_LOAD_FAILED = 2,
    ZYNFER_COREML_PREDICT_FAILED = 3,
};

typedef struct ZynferCoreMlProbe {
    int framework_linked;
    int configuration_ok;
    int compute_units_all_ok;
    int compute_units_cpu_and_ane_ok;
    /** 1 only if Instruments/Core ML tools confirmed ANE placement. Always 0 here. */
    int ane_execution_verified;
    /** 1 when MLState class is present (macOS 15+ / Core ML stateful API). */
    int ml_state_available;
    /** Darwin major version (e.g. 24 for macOS 15 Sequoia, 25 for macOS 16). 0 if unknown. */
    int macos_major;
    /** 1 if a .mlmodel / .mlpackage was discovered under models/ (never auto-loaded). */
    int model_artifact_present;
    char detail[768];
} ZynferCoreMlProbe;

typedef struct ZynferCoreMlSmoke {
    int load_ok;
    int predict_ok;
    /** Requested compute units label written for the ledger (not ANE proof). */
    char compute_units[64];
    /** First few output floats from the toy model (y[0..3] for the 1×4 toy). */
    float y0;
    float y1;
    float y2;
    float y3;
    char detail[768];
} ZynferCoreMlSmoke;

/** Probe Core ML availability. Does not load a model or claim ANE execution. */
int zynfer_coreml_probe(ZynferCoreMlProbe *out);

/**
 * Load an .mlmodel / .mlpackage and run one prediction with ones input.
 * Expects the Stage M7 toy: input "x" float32 [1,8], output "y" float32 [1,4].
 * Uses MLComputeUnitsCPUAndNeuralEngine when available (placement still unverified).
 */
int zynfer_coreml_smoke(const char *path, ZynferCoreMlSmoke *out);

#ifdef __cplusplus
}
#endif

#endif
