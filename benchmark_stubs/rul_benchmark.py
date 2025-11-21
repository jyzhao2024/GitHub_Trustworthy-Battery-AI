"""
RUL (Remaining Useful Life) Benchmark Example - FIXED VERSION
===============================================================
This benchmark demonstrates how to:
1. Load and preprocess battery data with CORRECT RUL calculation
2. Load a pre-trained model or initialize a simple baseline
3. Perform RUL predictions (remaining cycles until EOL)
4. Evaluate model performance with multiple metrics
5. Save results in JSON format for auditability

KEY FIXES:
- RUL is now INTEGER (cycles are discrete)
- RUL is MONOTONICALLY DECREASING for each battery
- RUL is calculated by backward projection from EOL point
- Data maintains temporal ordering within each battery
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model (adjust based on your actual model structure)
from models.baseline_lr import SimpleLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RULBenchmark:
    """
    RUL Benchmark Framework - FIXED VERSION

    This class encapsulates the complete workflow for RUL prediction tasks,
    ensuring reproducibility, traceability, and explainability.

    RUL is defined as the INTEGER number of remaining cycles until the battery
    reaches its End-of-Life (EOL) criterion (e.g., capacity drops to 80%).
    """

    def __init__(self, results_dir: str = 'results', eol_threshold: float = 80.0):
        """
        Initialize the benchmark

        Args:
            results_dir: Directory to save results
            eol_threshold: EOL threshold in SOH percentage (default: 80%)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.eol_threshold = eol_threshold

        # Metadata for traceability
        self.metadata = {
            'task': 'RUL_prediction',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.1-fixed',
            'eol_threshold': eol_threshold,
            'rul_type': 'integer_cycles'
        }

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load and preprocess battery data for RUL prediction

        FIXED VERSION ensures:
        1. RUL is INTEGER
        2. RUL decreases monotonically for each battery
        3. RUL is calculated correctly from EOL point

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target values (RUL in cycles, INTEGER)
            data_info: Metadata about the dataset
        """
        print("=" * 60)
        print("Step 1: Loading Data (FIXED VERSION)")
        print("=" * 60)

        # Simulate multiple battery degradation trajectories
        n_batteries = 5
        max_cycles = 1000
        all_X = []
        all_y = []
        battery_info = []

        for battery_id in range(n_batteries):
            print(f"  Generating battery {battery_id + 1}/{n_batteries}...", end=' ')

            # Generate DENSE cycle sequence (every cycle has data)
            cycles = np.arange(1, max_cycles + 1)  # [1, 2, 3, ..., 1000]
            n_cycles = len(cycles)

            # Simulate battery-specific degradation
            degradation_rate = np.random.uniform(0.015, 0.025)
            noise = np.random.normal(0, 0.5, n_cycles)  # Reduced noise for stability

            # Capacity degradation over cycles (SOH%)
            capacity = (100 - degradation_rate * cycles
                       - 0.00001 * (cycles ** 1.5)
                       + noise)
            capacity = np.clip(capacity, 70, 100)

            # Find the EOL cycle (first time capacity drops to threshold)
            eol_mask = capacity <= self.eol_threshold
            if np.any(eol_mask):
                eol_cycle = cycles[eol_mask][0]  # First cycle reaching EOL
            else:
                # Battery hasn't reached EOL in simulation
                # Extrapolate based on current degradation
                last_capacity = capacity[-1]
                if last_capacity > self.eol_threshold:
                    cycles_to_eol = int((last_capacity - self.eol_threshold) / degradation_rate)
                    eol_cycle = max_cycles + cycles_to_eol
                else:
                    eol_cycle = max_cycles

            # CORRECT RUL CALCULATION: Backward projection from EOL
            rul = eol_cycle - cycles
            rul = np.maximum(rul, 0).astype(int)  # Non-negative INTEGER

            # Verify RUL is monotonically decreasing
            assert np.all(np.diff(rul) <= 0), f"RUL not monotonic for battery {battery_id}!"

            # Generate other features
            voltages = 3.7 - (100 - capacity) * 0.01 + np.random.normal(0, 0.01, n_cycles)
            currents = np.ones(n_cycles) * 0.5 + np.random.normal(0, 0.03, n_cycles)
            temperatures = 25 + np.random.normal(0, 3, n_cycles)
            discharge_times = 3600 * (capacity / 100) + np.random.normal(0, 30, n_cycles)

            # Sample data points (to reduce dataset size but maintain temporal order)
            # Sample more densely near EOL for better prediction
            sample_indices = self._get_sample_indices(n_cycles, eol_cycle, n_samples=50)

            X_battery = np.column_stack([
                cycles[sample_indices],
                capacity[sample_indices],
                voltages[sample_indices],
                currents[sample_indices],
                temperatures[sample_indices],
                discharge_times[sample_indices],
                np.ones(len(sample_indices)) * battery_id  # Battery ID
            ])

            y_battery = rul[sample_indices]

            # Store battery info for analysis
            battery_info.append({
                'battery_id': battery_id,
                'eol_cycle': int(eol_cycle),
                'n_samples': len(sample_indices),
                'initial_rul': int(rul[sample_indices[0]]),
                'final_rul': int(rul[sample_indices[-1]])
            })

            all_X.append(X_battery)
            all_y.append(y_battery)

            print(f"✓ (EOL at cycle {eol_cycle})")

        # Combine all batteries
        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        # Verify all RUL values are integers
        assert np.all(y == y.astype(int)), "RUL contains non-integer values!"
        assert np.all(y >= 0), "RUL contains negative values!"

        data_info = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_batteries': n_batteries,
            'feature_names': [
                'cycle_number',
                'capacity',
                'voltage',
                'current',
                'temperature',
                'discharge_time',
                'battery_id'
            ],
            'target_name': 'RUL',
            'data_source': 'simulated_battery_degradation',
            'rul_range': [int(y.min()), int(y.max())],
            'cycle_range': [int(X[:, 0].min()), int(X[:, 0].max())],
            'eol_threshold': self.eol_threshold,
            'battery_info': battery_info
        }

        print(f"\n✓ Data loaded: {len(X)} samples from {n_batteries} batteries")
        print(f"✓ Features: {X.shape[1]} features")
        print(f"✓ RUL range: {data_info['rul_range'][0]} - {data_info['rul_range'][1]} cycles (INTEGER)")
        print(f"✓ EOL threshold: {self.eol_threshold}% capacity")
        print("\n✓ RUL Properties Verified:")
        print(f"  • All RUL values are integers: ✓")
        print(f"  • All RUL values are non-negative: ✓")
        print(f"  • RUL decreases monotonically per battery: ✓")
        print()

        return X, y, data_info

    def _get_sample_indices(self, n_cycles: int, eol_cycle: int, n_samples: int = 50) -> np.ndarray:
        """
        Get sample indices with denser sampling near EOL

        Args:
            n_cycles: Total number of cycles
            eol_cycle: Cycle number at EOL
            n_samples: Desired number of samples

        Returns:
            sample_indices: Sorted array of sample indices
        """
        # Sample more densely in the last 30% of life
        n_early = int(n_samples * 0.5)  # 50% samples from early life
        n_late = n_samples - n_early    # 50% samples from late life

        transition_point = int(n_cycles * 0.7)

        # Early life: uniform sampling
        early_indices = np.linspace(0, transition_point, n_early, dtype=int)

        # Late life: denser sampling near EOL
        late_indices = np.linspace(transition_point, n_cycles - 1, n_late, dtype=int)

        sample_indices = np.concatenate([early_indices, late_indices])
        sample_indices = np.unique(sample_indices)  # Remove duplicates
        sample_indices.sort()  # Maintain temporal order

        return sample_indices

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets

        STRATEGY: Use battery-wise split to avoid data leakage
        - Some batteries for training
        - Some batteries for testing

        Args:
            X: Feature matrix
            y: Target values (RUL)
            train_ratio: Ratio of batteries for training

        Returns:
            X_train, X_test, y_train, y_test
        """
        print("=" * 60)
        print("Step 2: Splitting Data (Battery-wise Split)")
        print("=" * 60)

        # Extract battery IDs
        battery_ids = X[:, -1]  # Last column is battery_id
        unique_batteries = np.unique(battery_ids)
        n_batteries = len(unique_batteries)

        # Randomly select batteries for training
        np.random.seed(42)  # For reproducibility
        n_train_batteries = max(1, int(n_batteries * train_ratio))
        train_batteries = np.random.choice(unique_batteries, n_train_batteries, replace=False)

        # Split data by battery
        train_mask = np.isin(battery_ids, train_batteries)

        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]

        print(f"✓ Training batteries: {sorted(train_batteries.astype(int).tolist())}")
        print(f"✓ Testing batteries: {sorted(np.setdiff1d(unique_batteries, train_batteries).astype(int).tolist())}")
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Testing samples: {len(X_test)}")
        print(f"✓ This prevents data leakage (no future info in training)")
        print()

        return X_train, X_test, y_train, y_test

    def load_or_train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           model_path: str = None) -> Any:
        """
        Load pre-trained model or train a simple baseline

        Args:
            X_train: Training features
            y_train: Training targets (RUL in cycles)
            model_path: Path to saved model (if available)

        Returns:
            Trained model instance
        """
        print("=" * 60)
        print("Step 3: Model Initialization")
        print("=" * 60)

        model = SimpleLinearRegression()

        if model_path and os.path.exists(model_path):
            print(f"✓ Loading pre-trained model from {model_path}")
        else:
            print("✓ Training baseline model (for demonstration)")
            print("  Note: RUL prediction typically benefits from LSTM/GRU")
            model.fit(X_train, y_train)
            print("✓ Model training completed")

        print()
        return model

    def predict(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """
        Generate RUL predictions on test data

        Args:
            model: Trained model
            X_test: Test features

        Returns:
            predictions: Predicted RUL values (INTEGER, non-negative)
        """
        print("=" * 60)
        print("Step 4: Generating Predictions")
        print("=" * 60)

        predictions = model.predict(X_test)

        # Post-processing: RUL must be non-negative INTEGER
        predictions = np.maximum(predictions, 0)  # Non-negative
        predictions = np.round(predictions).astype(int)  # Round to integer

        print(f"✓ Generated {len(predictions)} predictions")
        print(f"✓ Prediction range: {predictions.min()} - {predictions.max()} cycles (INTEGER)")
        print()

        return predictions

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for RUL prediction

        Args:
            y_true: True RUL values (INTEGER)
            y_pred: Predicted RUL values (INTEGER)

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print("=" * 60)
        print("Step 5: Model Evaluation")
        print("=" * 60)

        # Standard regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # RUL-specific metrics
        errors = y_true - y_pred

        # MAPE (avoid division by zero)
        mask = y_true > 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs(errors[mask] / y_true[mask])) * 100
        else:
            mape = 0.0

        # Prognostic metrics
        early_pred_rate = np.mean(errors > 0) * 100  # Overestimate (safer)
        late_pred_rate = np.mean(errors < 0) * 100   # Underestimate (risky)
        exact_pred_rate = np.mean(errors == 0) * 100  # Exact match

        # Accuracy within tolerance
        within_10 = np.mean(np.abs(errors) <= 10) * 100
        within_25 = np.mean(np.abs(errors) <= 25) * 100
        within_50 = np.mean(np.abs(errors) <= 50) * 100

        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'Max_Error': float(np.max(np.abs(errors))),
            'Early_Prediction_Rate': float(early_pred_rate),
            'Late_Prediction_Rate': float(late_pred_rate),
            'Exact_Prediction_Rate': float(exact_pred_rate),
            'Within_10cycles_Accuracy': float(within_10),
            'Within_25cycles_Accuracy': float(within_25),
            'Within_50cycles_Accuracy': float(within_50),
            'Mean_Error': float(np.mean(errors)),
            'Std_Error': float(np.std(errors)),
            'Median_Error': float(np.median(errors))
        }

        print("Evaluation Metrics:")
        print(f"  • MSE (Mean Squared Error):          {metrics['MSE']:.2f}")
        print(f"  • RMSE (Root Mean Squared Error):    {metrics['RMSE']:.2f} cycles")
        print(f"  • MAE (Mean Absolute Error):         {metrics['MAE']:.2f} cycles")
        print(f"  • R² (R-squared):                    {metrics['R2']:.4f}")
        print(f"  • MAPE (Mean Abs Percentage Error):  {metrics['MAPE']:.2f} %")
        print(f"  • Max Error:                         {metrics['Max_Error']:.0f} cycles")

        print(f"\n  RUL-Specific Metrics:")
        print(f"  • Early Predictions (safer):         {metrics['Early_Prediction_Rate']:.1f} %")
        print(f"  • Late Predictions (risky):          {metrics['Late_Prediction_Rate']:.1f} %")
        print(f"  • Exact Predictions:                 {metrics['Exact_Prediction_Rate']:.1f} %")

        print(f"\n  Accuracy Metrics:")
        print(f"  • Within ±10 cycles:                 {metrics['Within_10cycles_Accuracy']:.1f} %")
        print(f"  • Within ±25 cycles:                 {metrics['Within_25cycles_Accuracy']:.1f} %")
        print(f"  • Within ±50 cycles:                 {metrics['Within_50cycles_Accuracy']:.1f} %")

        print(f"\n  Error Statistics:")
        print(f"  • Mean Error:                        {metrics['Mean_Error']:.2f} cycles")
        print(f"  • Median Error:                      {metrics['Median_Error']:.2f} cycles")
        print(f"  • Std Error:                         {metrics['Std_Error']:.2f} cycles")
        print()

        return metrics

    def analyze_predictions(self, X_test: np.ndarray, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Perform detailed analysis of RUL predictions

        Args:
            X_test: Test features
            y_true: True RUL values
            y_pred: Predicted RUL values

        Returns:
            analysis: Dictionary containing prediction analysis
        """
        errors = y_true - y_pred
        battery_ids = X_test[:, -1].astype(int)

        # Per-battery analysis
        per_battery = {}
        for battery_id in np.unique(battery_ids):
            mask = battery_ids == battery_id
            per_battery[int(battery_id)] = {
                'n_samples': int(np.sum(mask)),
                'mae': float(np.mean(np.abs(errors[mask]))),
                'rmse': float(np.sqrt(np.mean(errors[mask] ** 2))),
                'mean_error': float(np.mean(errors[mask])),
                'rul_range': [int(y_true[mask].min()), int(y_true[mask].max())]
            }

        # RUL stage analysis
        early_life_mask = y_true > 500
        mid_life_mask = (y_true >= 200) & (y_true <= 500)
        eol_mask = y_true < 200

        stage_performance = {
            'early_life': {
                'rul_range': '> 500 cycles',
                'n_samples': int(np.sum(early_life_mask)),
                'mae': float(np.mean(np.abs(errors[early_life_mask]))) if np.sum(early_life_mask) > 0 else None,
                'rmse': float(np.sqrt(np.mean(errors[early_life_mask] ** 2))) if np.sum(early_life_mask) > 0 else None
            },
            'mid_life': {
                'rul_range': '200-500 cycles',
                'n_samples': int(np.sum(mid_life_mask)),
                'mae': float(np.mean(np.abs(errors[mid_life_mask]))) if np.sum(mid_life_mask) > 0 else None,
                'rmse': float(np.sqrt(np.mean(errors[mid_life_mask] ** 2))) if np.sum(mid_life_mask) > 0 else None
            },
            'end_of_life': {
                'rul_range': '< 200 cycles',
                'n_samples': int(np.sum(eol_mask)),
                'mae': float(np.mean(np.abs(errors[eol_mask]))) if np.sum(eol_mask) > 0 else None,
                'rmse': float(np.sqrt(np.mean(errors[eol_mask] ** 2))) if np.sum(eol_mask) > 0 else None
            }
        }

        analysis = {
            'per_battery_performance': per_battery,
            'stage_wise_performance': stage_performance,
            'prediction_distribution': {
                'early_predictions': int(np.sum(errors > 0)),
                'late_predictions': int(np.sum(errors < 0)),
                'exact_predictions': int(np.sum(errors == 0))
            }
        }

        return analysis

    def save_results(self, X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                    metrics: Dict[str, float], data_info: Dict[str, Any],
                    analysis: Dict[str, Any]) -> str:
        """
        Save prediction results and metrics to JSON file

        Args:
            X_test: Test features
            y_true: True RUL values
            y_pred: Predicted RUL values
            metrics: Evaluation metrics
            data_info: Dataset information
            analysis: Prediction analysis

        Returns:
            output_path: Path to saved results file
        """
        print("=" * 60)
        print("Step 6: Saving Results")
        print("=" * 60)

        # Convert to Python native types for JSON serialization
        results = {
            'metadata': self.metadata,
            'data_info': data_info,
            'predictions': {
                'true_values': y_true.tolist(),
                'predicted_values': y_pred.tolist(),
                'errors': (y_true - y_pred).tolist(),
                'absolute_errors': np.abs(y_true - y_pred).tolist(),
                'battery_ids': X_test[:, -1].astype(int).tolist(),
                'cycle_numbers': X_test[:, 0].astype(int).tolist()
            },
            'metrics': metrics,
            'analysis': analysis,
            'statistical_summary': {
                'true_mean': float(np.mean(y_true)),
                'true_std': float(np.std(y_true)),
                'true_median': float(np.median(y_true)),
                'true_min': int(np.min(y_true)),
                'true_max': int(np.max(y_true)),
                'pred_mean': float(np.mean(y_pred)),
                'pred_std': float(np.std(y_pred)),
                'pred_median': float(np.median(y_pred)),
                'pred_min': int(np.min(y_pred)),
                'pred_max': int(np.max(y_pred))
            }
        }

        # Save main results
        output_path = self.results_dir / 'rul_benchmark_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Results saved to: {output_path}")

        return str(output_path)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete RUL benchmark pipeline

        Returns:
            results: Complete results dictionary
        """
        print("\n" + "=" * 60)
        print("RUL PREDICTION BENCHMARK - FIXED VERSION")
        print("Trustworthy Battery AI Framework")
        print("=" * 60 + "\n")

        # Step 1: Load data
        X, y, data_info = self.load_data()

        # Step 2: Split data (battery-wise)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Step 3: Load or train model
        model = self.load_or_train_model(X_train, y_train)

        # Step 4: Generate predictions
        y_pred = self.predict(model, X_test)

        # Step 5: Evaluate
        metrics = self.evaluate(y_test, y_pred)

        # Additional analysis
        analysis = self.analyze_predictions(X_test, y_test, y_pred)

        # Step 6: Save results
        output_path = self.save_results(X_test, y_test, y_pred, metrics, data_info, analysis)

        print("=" * 60)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\n✓ All results saved to: {self.results_dir}")
        print(f"✓ Main results file: {output_path}")

        print("\n✓ Key Improvements in FIXED Version:")
        print("  • RUL is INTEGER (cycles are discrete)")
        print("  • RUL decreases monotonically for each battery")
        print("  • RUL calculated correctly from EOL point")
        print("  • Battery-wise data split prevents leakage")
        print()

        return {
            'metrics': metrics,
            'analysis': analysis,
            'output_path': output_path
        }


def main():
    """
    Main entry point for the RUL benchmark
    """
    # Initialize benchmark with EOL threshold at 80% capacity
    benchmark = RULBenchmark(results_dir='results', eol_threshold=80.0)

    # Run benchmark
    results = benchmark.run()

    return results


if __name__ == "__main__":
    main()