#ifndef ZYNFER_COREML_BRIDGE_H
#define ZYNFER_COREML_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ZYNFER_COREML_OK = 0,
    ZYNFER_COREML_UNAVAILABLE = 1,
};

typedef struct ZynferCoreMlProbe {
    int framework_linked;
    int configuration_ok;
    int compute_units_all_ok;
    int compute_units_cpu_and_ane_ok;
    /** 1 only if Instruments/Core ML tools confirmed ANE placement. Always 0 here. */
    int ane_execution_verified;
    char detail[512];
} ZynferCoreMlProbe;

/** Probe Core ML availability. Does not load a model or claim ANE execution. */
int zynfer_coreml_probe(ZynferCoreMlProbe *out);

#ifdef __cplusplus
}
#endif

#endif
