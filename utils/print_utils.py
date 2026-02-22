import json

def print_run_config(*, MODEL_CFG, CLUSTER_CFG, TRAIN_CFG, DATASET_CFG, CIRCUIT_SOURCE_CFG, derived=None):
    payload = {
        "MODEL_CFG": MODEL_CFG,
        "CLUSTER_CFG": CLUSTER_CFG,
        "TRAIN_CFG": TRAIN_CFG,
        "DATASET_CFG": DATASET_CFG,
        "CIRCUIT_SOURCE_CFG": CIRCUIT_SOURCE_CFG,
    }
    if derived:
        payload["DERIVED"] = derived

    print("\n" + "=" * 80)
    print("RUN CONFIG")
    print("=" * 80)
    print(json.dumps(payload, indent=2, sort_keys=False))
    print("=" * 80 + "\n")


def print_run_config_analyze(
    *,
    device,
    K,
    tech_names,
    MODEL_CFG,
    CLUSTER_CFG,
    DATASET_CFG,
    CIRCUIT_SOURCE_CFG,
):
    payload = {
        "MODEL_CFG": MODEL_CFG,
        "CLUSTER_CFG": CLUSTER_CFG,
        "DATASET_CFG": DATASET_CFG,
        "CIRCUIT_SOURCE_CFG": CIRCUIT_SOURCE_CFG,
        "DERIVED": {
            "device": str(device),
            "K": K,
            "tech_names": tech_names,
        },
    }
    print("\n" + "=" * 80)
    print("ANALYZE RUN CONFIG")
    print("=" * 80)
    print(json.dumps(payload, indent=2))
    print("=" * 80 + "\n")