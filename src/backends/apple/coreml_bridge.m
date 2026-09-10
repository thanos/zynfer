#import "coreml_bridge.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <string.h>
#include <sys/sysctl.h>

static int zynfer_darwin_major(void) {
    char buf[64];
    size_t len = sizeof(buf);
    if (sysctlbyname("kern.osrelease", buf, &len, NULL, 0) != 0) {
        return 0;
    }
    int major = 0;
    if (sscanf(buf, "%d", &major) != 1) {
        return 0;
    }
    return major;
}

int zynfer_coreml_probe(ZynferCoreMlProbe *out) {
    if (out == NULL) {
        return ZYNFER_COREML_UNAVAILABLE;
    }
    memset(out, 0, sizeof(*out));
    out->framework_linked = 1;
    out->ane_execution_verified = 0;
    out->model_artifact_present = 0;
    out->macos_major = zynfer_darwin_major();

    @autoreleasepool {
        Class ml_state = NSClassFromString(@"MLState");
        out->ml_state_available = (ml_state != Nil) ? 1 : 0;

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
                 "Core ML linked; MLModelConfiguration accepts All and "
                 "CPUAndNeuralEngine. MLState class %s (Darwin %d). "
                 "No .mlmodel auto-loaded. ANE placement is NOT verified "
                 "(needs Instruments Neural Engine / Core ML activity). "
                 "Stage M7: Qwen-scale path not retained. Toy smoke: "
                 "tools/fixtures/coreml_toy.mlpackage + `zynfer coreml-smoke`.",
                 out->ml_state_available ? "present" : "absent",
                 out->macos_major);
    }
    return ZYNFER_COREML_OK;
}

int zynfer_coreml_smoke(const char *path, ZynferCoreMlSmoke *out) {
    if (out == NULL) {
        return ZYNFER_COREML_UNAVAILABLE;
    }
    memset(out, 0, sizeof(*out));
    if (path == NULL || path[0] == '\0') {
        snprintf(out->detail, sizeof(out->detail), "empty model path");
        return ZYNFER_COREML_LOAD_FAILED;
    }

    @autoreleasepool {
        NSString *ns_path = [NSString stringWithUTF8String:path];
        NSURL *url = [NSURL fileURLWithPath:ns_path isDirectory:YES];
        /* .mlmodel is a file; .mlpackage is a directory. Try both. */
        BOOL is_dir = NO;
        [[NSFileManager defaultManager] fileExistsAtPath:ns_path isDirectory:&is_dir];
        if (!is_dir) {
            url = [NSURL fileURLWithPath:ns_path isDirectory:NO];
        }

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        if (cfg == nil) {
            snprintf(out->detail, sizeof(out->detail), "MLModelConfiguration alloc failed");
            return ZYNFER_COREML_UNAVAILABLE;
        }
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        snprintf(out->compute_units, sizeof(out->compute_units), "CPUAndNeuralEngine");

        NSError *err = nil;
        /* Uncompiled .mlpackage / .mlmodel need compileModel(at:) first. */
        NSURL *compiled = [MLModel compileModelAtURL:url error:&err];
        if (compiled == nil) {
            NSString *msg = err != nil ? [err localizedDescription] : @"unknown";
            snprintf(out->detail, sizeof(out->detail), "compile failed: %s", msg.UTF8String);
            return ZYNFER_COREML_LOAD_FAILED;
        }

        MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
        if (model == nil) {
            NSString *msg = err != nil ? [err localizedDescription] : @"unknown";
            snprintf(out->detail, sizeof(out->detail), "load failed: %s", msg.UTF8String);
            return ZYNFER_COREML_LOAD_FAILED;
        }
        out->load_ok = 1;

        MLMultiArray *x = [[MLMultiArray alloc] initWithShape:@[ @1, @8 ]
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&err];
        if (x == nil) {
            NSString *msg = err != nil ? [err localizedDescription] : @"unknown";
            snprintf(out->detail, sizeof(out->detail), "input alloc failed: %s", msg.UTF8String);
            return ZYNFER_COREML_PREDICT_FAILED;
        }
        for (NSUInteger i = 0; i < 8; i++) {
            x[@[ @0, @(i) ]] = @1.0;
        }

        id<MLFeatureProvider> in_feat =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ @"x" : x } error:&err];
        if (in_feat == nil) {
            NSString *msg = err != nil ? [err localizedDescription] : @"unknown";
            snprintf(out->detail, sizeof(out->detail), "feature provider failed: %s", msg.UTF8String);
            return ZYNFER_COREML_PREDICT_FAILED;
        }

        id<MLFeatureProvider> pred = [model predictionFromFeatures:in_feat error:&err];
        if (pred == nil) {
            NSString *msg = err != nil ? [err localizedDescription] : @"unknown";
            snprintf(out->detail, sizeof(out->detail), "predict failed: %s", msg.UTF8String);
            return ZYNFER_COREML_PREDICT_FAILED;
        }

        MLFeatureValue *yv = [pred featureValueForName:@"y"];
        MLMultiArray *y = yv != nil ? yv.multiArrayValue : nil;
        if (y == nil || y.count < 4) {
            snprintf(out->detail, sizeof(out->detail),
                     "predict ok but missing y[1,4] multiarray (got count=%lu)",
                     (unsigned long)(y != nil ? y.count : 0));
            return ZYNFER_COREML_PREDICT_FAILED;
        }
        out->predict_ok = 1;
        out->y0 = y[@[ @0, @0 ]].floatValue;
        out->y1 = y[@[ @0, @1 ]].floatValue;
        out->y2 = y[@[ @0, @2 ]].floatValue;
        out->y3 = y[@[ @0, @3 ]].floatValue;
        snprintf(out->detail, sizeof(out->detail),
                 "loaded %s; predict ok under %s; y=[%.6g, %.6g, %.6g, %.6g]. "
                 "ANE placement still UNVERIFIED without Instruments.",
                 path, out->compute_units, out->y0, out->y1, out->y2, out->y3);
    }
    return ZYNFER_COREML_OK;
}
