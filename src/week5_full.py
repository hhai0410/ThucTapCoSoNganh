"""Compatibility entry point for the leakage-free final training workflow.
Historical charts are retained in outputs/models; new measured metrics are
written to validation_metrics.json. No test-set threshold optimization.
"""
from train_validated import main

if __name__ == "__main__":
    main()
