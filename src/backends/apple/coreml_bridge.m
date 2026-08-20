#import "coreml_bridge.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <string.h>

int zynfer_coreml_probe(ZynferCoreMlProbe *out) {
    if (out == NULL) {
        return ZYNFER_COREML_UNAVAILABLE;
    }
    memset(out, 0, sizeof(*out));
    out->framework_linked = 1;
    out->ane_execution_verified = 0;

    @autoreleasepool {
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        if (cfg == nil) {
            snprintf(out->detail, sizeof(out->detail),
                     "Core ML linked but MLModelConfiguration alloc failed");
            return ZYNFER_COREML_UNAVAILABLE;
        }
        out->configuration_ok = 1;

        cfg.computeUnits = MLComputeUnitsAll;
        out->compute_units_all_ok = 1;

        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        out->compute_units_cpu_and_ane_ok = 1;

        snprintf(out->detail, sizeof(out->detail),
                 "Core ML framework linked; MLModelConfiguration accepts All and "
                 "CPUAndNeuralEngine. No .mlmodel loaded. ANE placement is NOT "
                 "verified (would need Instruments / model compile evidence).");
    }
    return ZYNFER_COREML_OK;
}
