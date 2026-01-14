# -*- coding: utf-8 -*-
"""
Validation test script for all decision modules
"""

import sys
import os

print("="*80)
print("DECISION MODULES - VALIDATION TESTS")
print("="*80)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Test 1: CTMC Matrix
print("\n[TEST 1] CTMC Generator Matrix...")
try:
    from src.decision.ctmc_generator_matrix import main as ctmc_main
    ctmc_main()
    print("SUCCESS: CTMC matrix generated")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Dependency Analysis
print("\n[TEST 2] Route-Temperature Dependency...")
try:
    from src.decision.route_temp_dependency import main as dep_main
    dep_main()
    print("SUCCESS: Dependency analysis complete")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: MLE Estimation
print("\n[TEST 3] Maximum Likelihood Estimation...")
try:
    from src.decision.mle_estimation import main as mle_main
    mle_main()
    print("SUCCESS: MLE estimates generated")
except Exception as e:
    print(f"ERROR: {e}")

# Test 4: Monte Carlo Optimization
print("\n[TEST 4] Monte Carlo Optimization...")
try:
    from src.decision.optimized_monte_carlo import main as mc_main
    mc_main()
    print("SUCCESS: Monte Carlo optimization complete")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*80)
print("EXECUTION COMPLETE")
print("="*80)
