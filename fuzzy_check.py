#!/usr/bin/env python3
"""
Comprehensive parameter sweep analysis for fuzzy community detection and report structuring.
"""

import os
import pandas as pd
import numpy as np
from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.clustering import detect_communities_fuzzy_cmeans
from src.visualization.network import create_network_graph, visualize_fuzzy_communities
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import config
import time
from collections import Counter, defaultdict
import itertools
import json

class ParameterSweepAnalysis:
    """Class to manage parameter sweep analysis of fuzzy clustering and report structuring."""
    
    def __init__(self, output_dir=None):
        """Initialize with output directory structure."""
        self.output_dir = output_dir or os.path.join(config.OUTPUT_DIR, "parameter_sweep")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.viz_dir = os.path.join(self.output_dir, "visualizations")
        self.report_dir = os.path.join(self.output_dir, "report_analysis")
        
        # Create directories
        for directory in [self.output_dir, self.results_dir, self.viz_dir, self.report_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Store results
        self.results = []
        self.df_exploded = None
        self.G = None
        self.cooccurrence_matrix = None
    
    def save_output(self, data, filename, directory=None, index=False, message=None):
        """Save output data to file with error handling."""
        directory = directory or self.results_dir
        filepath = os.path.join(directory, filename)
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=index)
            elif isinstance(data, dict):
                if all(isinstance(v, dict) for v in data.values()):
                    pd.DataFrame.from_dict(data, orient='index').to_csv(filepath, index=True)
                else:
                    pd.DataFrame([data]).to_csv(filepath, index=False)
            elif isinstance(data, list):
                pd.DataFrame(data).to_csv(filepath, index=False)
            else:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            if message:
                print(message)
            return True
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return False
    
    def prepare_data(self):
        """Load and prepare data for analysis."""
        print("Loading and preparing data...")
        
        # Load data
        df = load_data(config.INPUT_FILE)
        
        # Create exploded dataframe
        self.df_exploded = create_exploded_dataframe(df)
        self.save_output(self.df_exploded, "exploded_data.csv", 
                       message=f"Processed {len(self.df_exploded)} report-element pairs")
        
        # Calculate co-occurrence
        co_occurrence = calculate_cooccurrence(self.df_exploded)
        
        # Create co-occurrence matrix
        self.cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
        
        # Create network graph
        self.G = create_network_graph(co_occurrence, all_elements)
        print(f"Created network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
    
    def generate_parameter_grid(self, cluster_range=None, fuzziness_range=None):
        """Generate parameter combinations to test."""
        cluster_range = cluster_range or range(3, 13)
        
        # If fuzziness_range is a tuple of (min, max), create a range of values
        if fuzziness_range and isinstance(fuzziness_range, tuple) and len(fuzziness_range) == 2:
            fuzziness_values = np.linspace(fuzziness_range[0], fuzziness_range[1], 5)
        else:
            fuzziness_values = fuzziness_range or [1.5, 1.8, 2.0, 2.2, 2.5]
        
        # Create all combinations
        param_grid = list(itertools.product(cluster_range, fuzziness_values))
        
        print(f"Generated {len(param_grid)} parameter combinations to test")
        return param_grid
    
    def analyze_community_balance(self, primary_communities):
        """Analyze if communities are sufficiently balanced."""
        community_sizes = Counter(primary_communities.values())
        total_elements = len(primary_communities)
        
        sizes = pd.Series(community_sizes)
        
        # Calculate balance metrics
        if len(sizes) <= 1:
            return {
                "is_balanced": False,
                "size_variation": 0,
                "max_community_percent": 1.0,
                "min_community_percent": 1.0,
                "num_communities": 1
            }
        
        size_cv = sizes.std() / sizes.mean()  # Coefficient of variation
        max_size_percent = sizes.max() / total_elements
        min_size_percent = sizes.min() / total_elements
        
        # Check if balance is acceptable
        is_balanced = True
        
        if size_cv > 0.8:
            is_balanced = False
        
        if max_size_percent > 0.4:
            is_balanced = False
        
        if min_size_percent < 0.05 and len(community_sizes) > 3:
            is_balanced = False
        
        return {
            "is_balanced": is_balanced,
            "size_variation": size_cv,
            "max_community_percent": max_size_percent,
            "min_community_percent": min_size_percent,
            "num_communities": len(community_sizes),
            "sizes": dict(community_sizes)
        }
    
    def map_reports_to_communities(self, primary_communities):
        """Map reports to the communities they use elements from."""
        # Create a mapping of reports to their elements
        report_elements = self.df_exploded.groupby('Report')['data_element'].apply(list).to_dict()
        
        # Analyze each report's community alignment
        report_analysis = {}
        
        for report, elements in report_elements.items():
            # Map elements to communities
            element_communities = [primary_communities.get(elem) for elem in elements if elem in primary_communities]
            
            # Count community occurrences
            community_counts = Counter(element_communities)
            total_elements = len(element_communities)
            
            if total_elements == 0:
                continue
            
            # Get primary community (most elements)
            primary_community = community_counts.most_common(1)[0][0] if community_counts else None
            primary_community_percent = community_counts.most_common(1)[0][1] / total_elements if community_counts else 0
            
            # Calculate report fragmentation (entropy)
            community_proportions = [count/total_elements for count in community_counts.values()]
            entropy = -sum(p * np.log2(p) for p in community_proportions if p > 0)
            
            report_analysis[report] = {
                'element_count': total_elements,
                'community_count': len(community_counts),
                'primary_community': primary_community,
                'primary_community_percent': primary_community_percent,
                'entropy': entropy,
                'community_distribution': dict(community_counts),
                'is_fragmented': len(community_counts) > 1 and primary_community_percent < 0.8
            }
        
        return report_analysis
    
    def identify_report_opportunities(self, report_analysis):
        """Identify report consolidation and splitting opportunities."""
        # Identify consolidation opportunities
        consolidation = self._identify_report_consolidation(report_analysis)
        
        # Identify splitting opportunities
        splitting = self._identify_report_splitting(report_analysis)
        
        # Calculate opportunity metrics
        consolidation_count = len(consolidation)
        splitting_count = len(splitting)
        aligned_reports = sum(1 for data in report_analysis.values() 
                            if data['community_count'] == 1 or data['primary_community_percent'] >= 0.8)
        fragmented_reports = sum(1 for data in report_analysis.values() if data['is_fragmented'])
        
        total_reports = len(report_analysis)
        alignment_score = aligned_reports / total_reports if total_reports > 0 else 0
        
        # Calculate opportunity score (higher is better)
        opportunity_score = alignment_score + (consolidation_count * 0.1) / max(1, total_reports/5)
        
        return {
            'consolidation_opportunities': consolidation,
            'splitting_opportunities': splitting,
            'consolidation_count': consolidation_count,
            'splitting_count': splitting_count,
            'aligned_reports': aligned_reports,
            'fragmented_reports': fragmented_reports,
            'alignment_score': alignment_score,
            'opportunity_score': opportunity_score
        }
    
    def _identify_report_consolidation(self, report_analysis):
        """Find consolidation opportunities."""
        # Group reports by primary community
        community_reports = defaultdict(list)
        
        for report, data in report_analysis.items():
            if data['primary_community'] is not None and data['primary_community_percent'] >= 0.7:
                community_reports[data['primary_community']].append((report, data))
        
        # Find consolidation opportunities (communities with multiple reports)
        opportunities = []
        
        for community_id, reports in community_reports.items():
            if len(reports) <= 1:
                continue
                
            # For larger groups, suggest consolidation
            if len(reports) >= 3:
                report_names = [r[0] for r in reports]
                total_elements = sum(r[1]['element_count'] for r in reports)
                
                opportunities.append({
                    'community_id': community_id,
                    'report_count': len(reports),
                    'reports': report_names,
                    'total_elements': total_elements,
                    'opportunity_type': 'consolidation',
                    'impact_score': len(reports) * total_elements / 100  # Simple impact score
                })
        
        return opportunities
    
    def _identify_report_splitting(self, report_analysis):
        """Find splitting opportunities."""
        opportunities = []
        
        for report, data in report_analysis.items():
            # Check if the report is significantly fragmented
            if data['community_count'] >= 3 and data['primary_community_percent'] < 0.6:
                # This report draws from multiple communities with no strong alignment
                communities = data['community_distribution']
                top_communities = sorted(communities.items(), key=lambda x: x[1], reverse=True)[:3]
                
                opportunities.append({
                    'report': report,
                    'element_count': data['element_count'],
                    'community_count': data['community_count'],
                    'entropy': data['entropy'],
                    'opportunity_type': 'splitting',
                    'top_communities': dict(top_communities),
                    'impact_score': data['entropy'] * data['element_count'] / 10  # Simple impact score
                })
        
        return opportunities
    
    def run_parameter_sweep(self, param_grid=None):
        """Run analysis with different parameter combinations."""
        if self.G is None:
            self.prepare_data()
        
        # Generate parameter grid if not provided
        if param_grid is None:
            param_grid = self.generate_parameter_grid()
        
        print(f"Starting parameter sweep with {len(param_grid)} combinations...")
        
        # Run analysis for each parameter combination
        for i, (num_clusters, fuzziness) in enumerate(param_grid):
            run_id = f"clusters_{num_clusters}_fuzziness_{fuzziness:.1f}"
            print(f"\nRunning analysis {i+1}/{len(param_grid)}: {run_id}")
            
            # Create run directory
            run_dir = os.path.join(self.results_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            try:
                # Run fuzzy clustering
                primary_communities, membership_values = detect_communities_fuzzy_cmeans(
                    self.G, num_clusters=num_clusters, fuzziness=fuzziness
                )
                
                # Analyze community balance
                balance = self.analyze_community_balance(primary_communities)
                
                # Map reports to communities
                report_analysis = self.map_reports_to_communities(primary_communities)
                
                # Identify opportunities
                opportunities = self.identify_report_opportunities(report_analysis)
                
                # Store results
                result = {
                    'run_id': run_id,
                    'params': {
                        'num_clusters': num_clusters,
                        'fuzziness': fuzziness
                    },
                    'community_balance': balance,
                    'report_metrics': {
                        'total_reports': len(report_analysis),
                        'aligned_reports': opportunities['aligned_reports'],
                        'fragmented_reports': opportunities['fragmented_reports'],
                        'alignment_score': opportunities['alignment_score']
                    },
                    'opportunity_metrics': {
                        'consolidation_count': opportunities['consolidation_count'],
                        'splitting_count': opportunities['splitting_count'],
                        'opportunity_score': opportunities['opportunity_score']
                    }
                }
                
                self.results.append(result)
                
                # Save detailed results for this run
                self.save_output(result, "summary.json", directory=run_dir)
                
                # Save top opportunities
                if opportunities['consolidation_opportunities']:
                    top_consolidation = sorted(
                        opportunities['consolidation_opportunities'], 
                        key=lambda x: x['impact_score'], 
                        reverse=True
                    )[:10]
                    self.save_output(top_consolidation, "top_consolidation_opportunities.csv", directory=run_dir)
                
                if opportunities['splitting_opportunities']:
                    top_splitting = sorted(
                        opportunities['splitting_opportunities'], 
                        key=lambda x: x['impact_score'], 
                        reverse=True
                    )[:10]
                    self.save_output(top_splitting, "top_splitting_opportunities.csv", directory=run_dir)
                
                # Save visualization for interesting parameter combinations
                if i < 5 or opportunities['opportunity_score'] > 0.7:
                    viz_file = os.path.join(self.viz_dir, f"{run_id}_communities.png")
                    visualize_fuzzy_communities(self.G, membership_values, output_file=viz_file)
                
                print(f"  Completed analysis for {run_id}")
                print(f"  Opportunity score: {opportunities['opportunity_score']:.2f}")
                print(f"  Consolidation opportunities: {opportunities['consolidation_count']}")
                print(f"  Splitting opportunities: {opportunities['splitting_count']}")
                
            except Exception as e:
                print(f"Error analyzing parameters {run_id}: {e}")
                continue
        
        # Save all results
        self.save_output(self.results, "all_parameter_results.csv")
        
        return self.results
    
    def find_best_parameters(self, criteria='opportunity_score'):
        """Find best parameter combination based on a specific criterion."""
        if not self.results:
            print("No results available. Run parameter sweep first.")
            return None
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(self.results)
        
        # Extract nested metrics
        for metric_type in ['community_balance', 'report_metrics', 'opportunity_metrics']:
            if metric_type in results_df.columns:
                metrics = pd.json_normalize(results_df[metric_type])
                metrics.columns = [f"{metric_type}.{col}" for col in metrics.columns]
                results_df = pd.concat([results_df.drop(columns=[metric_type]), metrics], axis=1)
        
        # Extract parameters
        if 'params' in results_df.columns:
            params = pd.json_normalize(results_df['params'])
            results_df = pd.concat([results_df.drop(columns=['params']), params], axis=1)
        
        # Define criteria mapping
        criteria_map = {
            'opportunity_score': 'opportunity_metrics.opportunity_score',
            'alignment_score': 'report_metrics.alignment_score',
            'consolidation': 'opportunity_metrics.consolidation_count',
            'balance': 'community_balance.is_balanced'
        }
        
        # Get criterion to sort by
        criterion = criteria_map.get(criteria, criteria)
        
        if criterion not in results_df.columns:
            print(f"Criterion '{criterion}' not found in results. Available columns: {results_df.columns.tolist()}")
            criterion = 'opportunity_metrics.opportunity_score'
            print(f"Using '{criterion}' instead")
        
        # Sort by criterion
        sorted_results = results_df.sort_values(by=criterion, ascending=False)
        
        # Get best parameters
        best_row = sorted_results.iloc[0]
        best_params = {
            'num_clusters': best_row['num_clusters'],
            'fuzziness': best_row['fuzziness'],
            'run_id': best_row['run_id'],
            'score': best_row[criterion]
        }
        
        return best_params, sorted_results
    
    def visualize_parameter_impact(self):
        """Create visualizations showing how parameters affect results."""
        if not self.results:
            print("No results available. Run parameter sweep first.")
            return
        
        # Convert results to DataFrame
        _, results_df = self.find_best_parameters()
        
        # Create visualization directory
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 1. Heatmap of opportunity score by parameters
        plt.figure(figsize=(12, 8))
        pivot = results_df.pivot_table(
            index='fuzziness', 
            columns='num_clusters', 
            values='opportunity_metrics.opportunity_score'
        )
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Opportunity Score by Parameters")
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "opportunity_score_heatmap.png"))
        plt.close()
        
        # 2. Line plot of scores by number of clusters
        plt.figure(figsize=(12, 8))
        cluster_scores = results_df.groupby('num_clusters').agg({
            'opportunity_metrics.opportunity_score': 'mean',
            'report_metrics.alignment_score': 'mean',
            'opportunity_metrics.consolidation_count': 'mean'
        })
        cluster_scores.columns = ['Opportunity Score', 'Alignment Score', 'Consolidation Opportunities']
        
        # Normalize columns for easier comparison
        for col in cluster_scores.columns:
            if cluster_scores[col].max() > 0:
                cluster_scores[col] = cluster_scores[col] / cluster_scores[col].max()
        
        cluster_scores.plot(marker='o')
        plt.title("Normalized Scores by Number of Clusters")
        plt.ylabel("Normalized Score (higher is better)")
        plt.xlabel("Number of Clusters")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "scores_by_clusters.png"))
        plt.close()
        
        # 3. Community balance by parameters
        plt.figure(figsize=(12, 8))
        balance_cols = [col for col in results_df.columns if col.startswith('community_balance.')]
        
        for col in ['community_balance.size_variation', 'community_balance.max_community_percent']:
            if col in results_df.columns:
                plt.figure(figsize=(12, 6))
                for clusters, group in results_df.groupby('num_clusters'):
                    plt.plot(group['fuzziness'], group[col], marker='o', label=f"Clusters={clusters}")
                
                plt.title(f"{col.split('.')[-1].replace('_', ' ').title()} by Parameters")
                plt.xlabel("Fuzziness")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.viz_dir, f"{col.split('.')[-1]}_by_parameters.png"))
                plt.close()
        
        # 4. Top 5 parameter combinations
        top5 = results_df.sort_values('opportunity_metrics.opportunity_score', ascending=False).head(5)
        
        plt.figure(figsize=(12, 6))
        x = range(len(top5))
        width = 0.2
        
        plt.bar([i-width for i in x], top5['report_metrics.alignment_score'], width=width, label='Alignment Score')
        plt.bar([i for i in x], top5['opportunity_metrics.opportunity_score'], width=width, label='Opportunity Score')
        plt.bar([i+width for i in x], top5['opportunity_metrics.consolidation_count']/10, width=width, label='Consolidation Count (/10)')
        
        plt.xticks(x, [f"C={row['num_clusters']}, F={row['fuzziness']:.1f}" for _, row in top5.iterrows()])
        plt.title("Top 5 Parameter Combinations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "top5_parameters.png"))
        plt.close()
    
    def generate_final_report(self):
        """Generate final report with best parameters and opportunities."""
        if not self.results:
            print("No results available. Run parameter sweep first.")
            return
        
        # Find best parameters for different criteria
        best_params = {}
        for criterion in ['opportunity_score', 'alignment_score', 'consolidation']:
            best_params[criterion], _ = self.find_best_parameters(criterion)
        
        # Create report
        report = {
            'best_parameters': best_params,
            'parameter_combinations_tested': len(self.results),
            'run_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate detailed report for the overall best parameters
        best_run_id = best_params['opportunity_score']['run_id']
        best_run_dir = os.path.join(self.results_dir, best_run_id)
        
        try:
            # Load detailed results for best run
            with open(os.path.join(best_run_dir, "summary.json"), 'r') as f:
                best_run_summary = json.load(f)
            
            report['best_run_summary'] = best_run_summary
            
            # Load top opportunities
            consolidation_file = os.path.join(best_run_dir, "top_consolidation_opportunities.csv")
            if os.path.exists(consolidation_file):
                top_consolidation = pd.read_csv(consolidation_file)
                report['top_consolidation_opportunities'] = top_consolidation.to_dict(orient='records')
            
            splitting_file = os.path.join(best_run_dir, "top_splitting_opportunities.csv")
            if os.path.exists(splitting_file):
                top_splitting = pd.read_csv(splitting_file)
                report['top_splitting_opportunities'] = top_splitting.to_dict(orient='records')
        
        except Exception as e:
            print(f"Error loading detailed results for best run: {e}")
        
        # Save report
        self.save_output(report, "parameter_sweep_report.json", 
                       message="Generated final parameter sweep report")
        
        # Print summary
        print("\n=== PARAMETER SWEEP ANALYSIS SUMMARY ===")
        print(f"Tested {len(self.results)} parameter combinations")
        
        print("\nBest parameters by criterion:")
        for criterion, params in best_params.items():
            print(f"  {criterion.title()}: Clusters={params['num_clusters']}, Fuzziness={params['fuzziness']:.1f} (Score: {params['score']:.2f})")
        
        # Recommended parameters
        recommended = best_params['opportunity_score']
        print(f"\nRECOMMENDED PARAMETERS: Clusters={recommended['num_clusters']}, Fuzziness={recommended['fuzziness']:.1f}")
        
        # Report location
        print(f"\nDetailed report saved to: {os.path.join(self.output_dir, 'parameter_sweep_report.json')}")
        print(f"Visualizations saved to: {self.viz_dir}")
        
        return report

def main():
    """Run the parameter sweep analysis."""
    start_time = time.time()
    print("Starting comprehensive parameter sweep analysis...")
    
    # Create analyzer
    analyzer = ParameterSweepAnalysis()
    
    # Define parameter grid
    cluster_range = range(3, 4)  # 3 to 14 clusters
    fuzziness_range = (1.3, 1.5)  # Range of fuzziness values
    param_grid = analyzer.generate_parameter_grid(cluster_range, fuzziness_range)
    
    # Run parameter sweep
    analyzer.run_parameter_sweep(param_grid)
    
    # Create visualizations
    analyzer.visualize_parameter_impact()
    
    # Generate final report
    analyzer.generate_final_report()
    
    elapsed_time = time.time() - start_time
    print(f"\nParameter sweep analysis complete! ({elapsed_time:.2f} seconds)")
    
    return 0

if __name__ == "__main__":
    exit(main())