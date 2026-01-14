# Supply Chain Cold Chain Management System

> Probabilistic decision system for temperature-sensitive medicine distribution in crisis situations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Research-green.svg)]()

---

## 🎯 Description

This project implements a **probabilistic decision system** for cold chain supply chain management. It focuses on decision-making modules that complement agent-based modeling for distribution in uncertain environments.

### Research Context

**Problem**: Distribution of temperature-sensitive medicines (COVID-19 vaccines, insulin, etc.) in crisis situations (wars, pandemics, natural disasters) with:
- ❌ Lack of real-time data
- ❌ Limited visibility of supply chain status
- ❌ Unpredictable critical events
- ❌ Decisions under uncertainty

**Solution**: Two-part system:
1. **AnyLogic**: Multi-agent modeling + data generation
2. **Python (this project)**: Probabilistic evaluation + decision-making

---

## ⚡ Quick Start

### Installation

```powershell
pip install -r requirements.txt
```

### First Test

```powershell
python examples\complete_example.py
```

### Validation Tests

```python
from src.models import Vehicle, Medicine
from src.decision import DecisionEngine

# Create vehicle and medicine
vehicle = Vehicle("TRUCK_001", "refrigerated_truck", 1000, (2, 8), 500, 60, 5000)
medicine = Medicine("VAX_001", "COVID-19 Vaccine", "Pfizer-BioNTech", 1000, (2, 8), 0.95)

# Initialize decision engine
engine = DecisionEngine()
decision = engine.evaluate_delivery(vehicle, medicine, route_info={
    'distance': 250,
    'estimated_time': 4.5,
    'risk_level': 'medium'
})

print(f"Decision: {decision.action}")
print(f"Risk Score: {decision.risk_score:.2%}")
```

---

## 📦 Project Structure

```
SupplyChainManagement/
├── src/
│   ├── decision/              # Probabilistic decision modules
│   │   ├── ctmc_generator_matrix.py      # CTMC Q matrix (9×9)
│   │   ├── route_temp_dependency.py      # Statistical dependency analysis
│   │   ├── mle_estimation.py             # Maximum likelihood estimation
│   │   ├── optimized_monte_carlo.py      # Monte Carlo optimization
│   │   ├── markov_chain_module.py        # Markov chain implementation
│   │   ├── probabilistic_forecasting_module.py
│   │   ├── dynamic_risk_assessment_module.py
│   │   └── decision_engine.py            # Main decision engine
│   ├── models/                # Data models
│   │   ├── vehicle.py
│   │   ├── medicine.py
│   │   └── events.py
│   ├── simulation/            # Simulation engine
│   │   ├── simulation_engine.py
│   │   └── critical_events.py
│   ├── probabilistic/         # Risk models
│   │   └── risk_models.py
│   └── utils/                 # Utilities
│       └── visualization.py
├── examples/
│   └── complete_example.py    # Complete usage example
├── tests/
│   ├── test_decision.py
│   ├── test_medicine.py
│   └── test_vehicle.py
├── scripts/
│   ├── generate_data_only.py
│   └── generate_templates.py
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

---

## 🧪 Key Features

### 1. CTMC Generator Matrix
Complete 9×9 continuous-time Markov chain matrix modeling:
- **States**: (Route Condition × Temperature State)
- **Properties**: Rows sum to 0, off-diagonal ≥ 0
- **Validation**: P(t) = exp(Q×t) verified via Monte Carlo

```python
from src.decision.ctmc_generator_matrix import CTMCGeneratorMatrix

ctmc = CTMCGeneratorMatrix()
ctmc.print_matrix()
ctmc.validate_against_simulation(t=1.0, n_simulations=500)
```

### 2. Route-Temperature Dependency
Statistical validation of route-temperature coupling:
- **Chi-square test**: χ² = 74.54, p < 0.001
- **Conditional probabilities**: P(Critical|Dangerous) = 2×P(Critical|Safe)

```python
from src.decision.route_temp_dependency import RouteTempDependency

analysis = RouteTempDependency()
chi2_result = analysis.chi_square_test()
print(f"χ² = {chi2_result.statistic:.2f}, p = {chi2_result.p_value:.4f}")
```

### 3. Maximum Likelihood Estimation
Empirical rate estimation from UNHCR/WHO logs:
- **13 transition rates** with 95% confidence intervals
- **Validation**: Alignment with literature (WHO 2015, UNHCR 2022)

```python
from src.decision.mle_estimation import MLEEstimator

estimator = MLEEstimator()
rates = estimator.estimate_all_rates()
estimator.export_to_latex("results/mle_estimates.tex")
```

### 4. Optimized Monte Carlo
Performance-optimized forecasting:
- **Adaptive sampling**: 90% time reduction (23s → 2.3s)
- **Importance sampling**: Focus on critical states
- **Convergence**: MSE < 0.01 with N=1000 samples

```python
from src.decision.optimized_monte_carlo import OptimizedMonteCarlo

mc = OptimizedMonteCarlo()
results = mc.adaptive_monte_carlo(n_initial=100, target_mse=0.01)
```

---

## 📊 Dependencies

Core dependencies:
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing
- `pandas>=2.0.0` - Data manipulation
- `simpy>=4.0.1` - Discrete-event simulation
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `pyyaml>=6.0` - Configuration

See [`requirements.txt`](requirements.txt) for complete list.

---

## 🚀 Usage Examples

### Basic Decision Making
```python
from src.decision import DecisionEngine

engine = DecisionEngine()
decision = engine.evaluate_delivery(vehicle, medicine, route_info)

if decision.action == "PROCEED":
    print(f"✓ Delivery approved (risk: {decision.risk_score:.1%})")
elif decision.action == "REROUTE":
    print(f"⚠ Rerouting recommended: {decision.alternative_route}")
else:
    print(f"✗ Delivery cancelled: {decision.reason}")
```

### Run Validation Tests
```powershell
python test_validation.py
```

---

## 📄 License

This project is part of doctoral research. For academic use only.

---

## 📧 Contact

For questions or collaboration: articledev82@gmail.com

---

## 🔬 Citation

If you use this code in your research, please cite accordingly.
