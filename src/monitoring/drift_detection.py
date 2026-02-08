"""
Drift Detection Module - Pure SciPy Version

This module detects drift using scipy statistical tests.
No Evidently dependency required.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Drift detection system for ML models using scipy statistical tests.
    
    Detects data drift and target drift.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.5,
        stattest: str = "ks"
    ):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Threshold for drift detection (0-1)
            stattest: Statistical test ('ks' for Kolmogorov-Smirnov)
        """
        self.drift_threshold = drift_threshold
        self.stattest = stattest
        
    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Detect data drift between reference and current datasets.
        
        Args:
            reference_data: Baseline dataset (e.g., training data)
            current_data: New dataset to compare
            
        Returns:
            has_drift: True if significant drift detected
            metrics: Dictionary of drift metrics
        """
        logger.info("Detecting data drift...")
        
        # Calculate drift for each column
        n_features = len([col for col in reference_data.columns if col != 'target'])
        drifted_features = 0
        column_drift = {}
        
        for col in reference_data.columns:
            if col == 'target':
                continue
                
            # Get values
            ref_values = reference_data[col].values
            curr_values = current_data[col].values
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_values, curr_values)
            
            # Drift detected if p-value < 0.05 (95% confidence)
            drift_detected = p_value < 0.05
            
            if drift_detected:
                drifted_features += 1
            
            column_drift[col] = {
                'drift_detected': drift_detected,
                'drift_score': statistic,
                'p_value': p_value
            }
        
        # Calculate drift share
        drift_share = drifted_features / n_features if n_features > 0 else 0
        
        # Dataset drift if more than threshold of features drift
        has_drift = drift_share > self.drift_threshold
        
        metrics = {
            'dataset_drift': has_drift,
            'drift_share': drift_share,
            'n_features': n_features,
            'n_drifted_features': drifted_features,
            'column_drift': column_drift
        }
        
        if has_drift:
            logger.warning(f"⚠️  Data drift detected!")
            logger.warning(f"   Drift share: {drift_share:.2%}")
            logger.warning(f"   Drifted features: {drifted_features}/{n_features}")
        else:
            logger.info(f"✅ No significant data drift detected")
            logger.info(f"   Drift share: {drift_share:.2%}")
        
        return has_drift, metrics
    
    def detect_target_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = 'target'
    ) -> Tuple[bool, Dict]:
        """
        Detect drift in target variable distribution.
        
        Args:
            reference_data: Baseline dataset with targets
            current_data: New dataset with targets
            target_column: Name of target column
            
        Returns:
            has_drift: True if target drift detected
            metrics: Dictionary of drift metrics
        """
        logger.info("Detecting target drift...")
        
        ref_targets = reference_data[target_column].values
        curr_targets = current_data[target_column].values
        
        # For categorical targets, use chi-square test
        ref_counts = pd.Series(ref_targets).value_counts(normalize=True).sort_index()
        curr_counts = pd.Series(curr_targets).value_counts(normalize=True).sort_index()
        
        # Align indices
        all_classes = sorted(set(ref_counts.index) | set(curr_counts.index))
        ref_props = [ref_counts.get(c, 0) for c in all_classes]
        curr_props = [curr_counts.get(c, 0) for c in all_classes]
        
        # Chi-square test
        chi2_stat, p_value = stats.chisquare(
            [c * len(curr_targets) for c in curr_props],
            [r * len(curr_targets) for r in ref_props]
        )
        
        has_drift = p_value < 0.05
        
        target_drift_info = {
            'drift_detected': has_drift,
            'drift_score': chi2_stat,
            'p_value': p_value,
            'reference_distribution': dict(zip(all_classes, ref_props)),
            'current_distribution': dict(zip(all_classes, curr_props))
        }
        
        if has_drift:
            logger.warning(f"⚠️  Target drift detected!")
            logger.warning(f"   Chi-square statistic: {chi2_stat:.4f}")
            logger.warning(f"   p-value: {p_value:.6f}")
        else:
            logger.info(f"✅ No target drift detected")
            logger.info(f"   p-value: {p_value:.6f}")
        
        return has_drift, target_drift_info
    
    def save_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: Path,
        report_name: str = "drift_report"
    ) -> Path:
        """
        Save drift report as HTML file.
        
        Args:
            reference_data: Baseline dataset
            current_data: Current dataset
            output_path: Directory to save report
            report_name: Name of report file (without extension)
            
        Returns:
            Path to saved report
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"{report_name}.html"
        
        # Create a simple HTML report
        html_content = self._generate_html_report(reference_data, current_data)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Drift report saved to: {report_file}")
        
        return report_file
    
    def _generate_html_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> str:
        """Generate a simple HTML drift report."""
        
        # Detect drift
        data_drift, data_metrics = self.detect_data_drift(reference_data, current_data)
        
        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                .metric {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
                .warning {{ border-left-color: #ff9800; }}
                .alert {{ border-left-color: #f44336; }}
                .drift-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-box {{ background: #e3f2fd; padding: 20px; border-radius: 4px; text-align: center; }}
                .stat-value {{ font-size: 32px; font-weight: bold; color: #1976d2; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #4CAF50; color: white; }}
                .drift-yes {{ color: #f44336; font-weight: bold; }}
                .drift-no {{ color: #4CAF50; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Data Drift Detection Report</h1>
                
                <div class="drift-stats">
                    <div class="stat-box">
                        <div class="stat-value">{'YES' if data_drift else 'NO'}</div>
                        <div class="stat-label">Dataset Drift</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{data_metrics['drift_share']:.1%}</div>
                        <div class="stat-label">Drift Share</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{data_metrics['n_drifted_features']}/{data_metrics['n_features']}</div>
                        <div class="stat-label">Drifted Features</div>
                    </div>
                </div>
                
                <h2>Summary</h2>
                <div class="metric {'alert' if data_drift else ''}">
                    <strong>Dataset Drift:</strong> {'⚠️ DETECTED' if data_drift else '✅ NOT DETECTED'}<br>
                    <strong>Drift Share:</strong> {data_metrics['drift_share']:.2%} of features show significant drift<br>
                    <strong>Threshold:</strong> {self.drift_threshold:.0%}
                </div>
                
                <h2>Feature-Level Drift Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Drift Detected</th>
                            <th>Drift Score (KS Statistic)</th>
                            <th>P-Value</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add table rows for each feature (top 20 by drift score)
        sorted_features = sorted(data_metrics['column_drift'].items(), 
                                key=lambda x: x[1]['drift_score'], 
                                reverse=True)[:20]
        
        for col, metrics in sorted_features:
            drift_class = 'drift-yes' if metrics['drift_detected'] else 'drift-no'
            html += f"""
                        <tr>
                            <td>{col}</td>
                            <td class="{drift_class}">{'YES' if metrics['drift_detected'] else 'NO'}</td>
                            <td>{metrics['drift_score']:.4f}</td>
                            <td>{metrics['p_value']:.6f}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
                
                <h2>Recommendations</h2>
                <div class="metric warning">
        """
        
        if data_drift:
            html += """
                    <strong>⚠️ Action Required:</strong><br>
                    • Investigate the drifted features listed above<br>
                    • Consider retraining the model with recent data<br>
                    • Monitor model performance closely<br>
                    • Update data validation rules if needed
            """
        else:
            html += """
                    <strong>✅ No Action Needed:</strong><br>
                    • Data distribution remains stable<br>
                    • Continue normal monitoring schedule<br>
                    • Model performance should remain consistent
            """
        
        html += """
                </div>
                
                <h2>About This Report</h2>
                <div class="metric">
                    <strong>Statistical Test:</strong> Kolmogorov-Smirnov (KS) two-sample test<br>
                    <strong>Significance Level:</strong> 0.05 (95% confidence)<br>
                    <strong>Drift Threshold:</strong> {threshold:.0%} of features must drift to trigger dataset-level drift<br>
                    <strong>Generated:</strong> {timestamp}
                </div>
            </div>
        </body>
        </html>
        """.format(
            threshold=self.drift_threshold,
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return html
    
    def should_retrain(
        self,
        data_drift: bool,
        target_drift: bool,
        prediction_drift: bool,
        drift_share: float,
        retrain_threshold: float = 0.3
    ) -> Tuple[bool, str]:
        """
        Determine if model should be retrained based on drift detection.
        
        Args:
            data_drift: Data drift detected
            target_drift: Target drift detected
            prediction_drift: Prediction drift detected
            drift_share: Share of features with drift
            retrain_threshold: Threshold for triggering retraining
            
        Returns:
            should_retrain: True if retraining recommended
            reason: Explanation for decision
        """
        reasons = []
        
        if data_drift and drift_share > retrain_threshold:
            reasons.append(f"Data drift detected ({drift_share:.1%} of features)")
        
        if target_drift:
            reasons.append("Target distribution has changed")
        
        if prediction_drift:
            reasons.append("Prediction distribution has changed")
        
        if reasons:
            reason = "Retraining recommended: " + "; ".join(reasons)
            logger.warning(f"⚠️  {reason}")
            return True, reason
        else:
            reason = "No significant drift detected - model is stable"
            logger.info(f"✅ {reason}")
            return False, reason


if __name__ == "__main__":
    print("=" * 70)
    print("Drift Detection Module - Pure SciPy Version")
    print("=" * 70)
    print()
    print("✅ Ready to use!")
    print()
    print("Run:")
    print("  python scripts/simulate_drift.py   # Test with simulated drift")
    print("  python scripts/check_drift.py      # Check real data drift")
    print()
