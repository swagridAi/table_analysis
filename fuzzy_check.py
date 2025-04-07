#!/usr/bin/env python3
"""
Data Co-occurrence Analysis with report-community alignment analysis.
"""

import os
import pandas as pd
import numpy as np
from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.clustering import detect_communities_fuzzy_cmeans
from src.visualization.heatmap import create_heatmap
from src.visualization.network import create_network_graph, visualize_fuzzy_communities
import networkx as nx
import config
import time
from collections import Counter, defaultdict

def ensure_directories_exist():
    """Create output directories if they don't exist."""
    directories = [
        config.PROCESSED_DATA_DIR,
        config.VISUALIZATIONS_DIR,
        config.EXPORTS_DIR,
        config.COMMUNITY_VIZ_DIR,
        os.path.join(config.OUTPUT_DIR, "report_analysis")  # New directory for report analysis
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_output(data, filename, directory=None, index=False, message=None):
    """Centralized function to save outputs with proper error handling."""
    directory = directory or config.PROCESSED_DATA_DIR
    filepath = os.path.join(directory, filename)
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=index)
        elif isinstance(data, dict):
            pd.DataFrame.from_dict(data, orient='index').to_csv(filepath, index=True)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if message:
            print(message)
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def analyze_community_balance(primary_communities):
    """Analyze if communities are sufficiently balanced."""
    community_sizes = Counter(primary_communities.values())
    total_elements = len(primary_communities)
    
    sizes = pd.Series(community_sizes)
    
    # Calculate balance metrics
    size_cv = sizes.std() / sizes.mean()  # Coefficient of variation
    max_size_percent = sizes.max() / total_elements
    min_size_percent = sizes.min() / total_elements
    
    # Check if balance is acceptable
    is_balanced = True
    reasons = []
    
    if size_cv > 0.8:
        is_balanced = False
        reasons.append(f"High size variation (CV={size_cv:.2f})")
    
    if max_size_percent > 0.4:
        is_balanced = False
        reasons.append(f"Dominant community ({max_size_percent:.1%} of elements)")
    
    if min_size_percent < 0.05 and len(community_sizes) > 3:
        is_balanced = False
        reasons.append(f"Very small communities ({min_size_percent:.1%} of elements)")
    
    result = {
        "is_balanced": is_balanced,
        "size_variation": size_cv,
        "max_community_percent": max_size_percent,
        "min_community_percent": min_size_percent,
        "num_communities": len(community_sizes),
        "reasons": reasons
    }
    
    # Print analysis
    status = "✓ ACCEPTABLE" if is_balanced else "❌ NEEDS IMPROVEMENT"
    print(f"\nCommunity Balance Analysis: {status}")
    print(f"Communities: {len(community_sizes)}")
    print(f"Size variation (CV): {size_cv:.2f}")
    print(f"Largest community: {sizes.max()} elements ({max_size_percent:.1%})")
    print(f"Smallest community: {sizes.min()} elements ({min_size_percent:.1%})")
    
    if reasons:
        print("Issues found:")
        for reason in reasons:
            print(f"  • {reason}")
    
    return result

def map_reports_to_communities(df_exploded, primary_communities):
    """
    Map reports to the communities they use elements from.
    
    Args:
        df_exploded: DataFrame with report-element relationships
        primary_communities: Dictionary mapping elements to community IDs
        
    Returns:
        Dictionary with report analysis
    """
    # Create a mapping of reports to their elements
    report_elements = df_exploded.groupby('Report')['data_element'].apply(list).to_dict()
    
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

def generate_community_report_assignments(report_analysis, min_elements=3):
    """
    Generate recommended report assignments based on communities.
    
    Args:
        report_analysis: Dictionary with report community analysis
        min_elements: Minimum elements to include in a report
        
    Returns:
        DataFrame with recommended report assignments
    """
    # Group elements by community
    community_elements = defaultdict(list)
    
    # Extract the primary community for each element from the report analysis
    for report, data in report_analysis.items():
        if data['element_count'] < min_elements:
            continue
            
        primary_community = data['primary_community']
        if primary_community is not None:
            community_elements[primary_community].append(report)
    
    # Create recommendations for each community
    recommendations = []
    
    for community_id, reports in community_elements.items():
        # Sort reports by how well they align with the community
        reports_sorted = sorted(
            [(report, report_analysis[report]['primary_community_percent']) 
             for report in reports],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Group strongly aligned reports
        core_reports = [r[0] for r in reports_sorted if r[1] >= 0.8]
        mixed_reports = [r[0] for r in reports_sorted if r[1] < 0.8]
        
        recommendations.append({
            'community_id': community_id,
            'report_count': len(reports),
            'core_report_count': len(core_reports),
            'mixed_report_count': len(mixed_reports),
            'core_reports': ', '.join(core_reports[:5]) + ('...' if len(core_reports) > 5 else ''),
            'mixed_reports': ', '.join(mixed_reports[:5]) + ('...' if len(mixed_reports) > 5 else ''),
            'consolidation_potential': len(core_reports) > 1,
            'fragmentation_issues': len(mixed_reports) > 0
        })
    
    return pd.DataFrame(recommendations)

def identify_report_consolidation_opportunities(report_analysis):
    """
    Identify opportunities to consolidate reports based on community alignment.
    
    Args:
        report_analysis: Dictionary with report community analysis
        
    Returns:
        List of consolidation opportunities
    """
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
                'description': f"Consider consolidating {len(reports)} reports from Community {community_id}"
            })
    
    return opportunities

def identify_report_splitting_opportunities(report_analysis):
    """
    Identify opportunities to split reports based on community fragmentation.
    
    Args:
        report_analysis: Dictionary with report community analysis
        
    Returns:
        List of splitting opportunities
    """
    opportunities = []
    
    for report, data in report_analysis.items():
        # Check if the report is significantly fragmented
        if data['community_count'] >= 3 and data['primary_community_percent'] < 0.6:
            # This report draws from multiple communities with no strong alignment
            communities = data['community_distribution']
            top_communities = sorted(communities.items(), key=lambda x: x[1], reverse=True)[:3]
            
            description = f"Consider splitting into {len(top_communities)} reports by community:"
            for comm_id, count in top_communities:
                description += f"\n- Community {comm_id}: {count} elements ({count/data['element_count']:.0%})"
            
            opportunities.append({
                'report': report,
                'element_count': data['element_count'],
                'community_count': data['community_count'],
                'entropy': data['entropy'],
                'opportunity_type': 'splitting',
                'description': description
            })
    
    return opportunities

def create_report_community_matrix(df_exploded, primary_communities):
    """
    Create a matrix showing which reports use elements from which communities.
    
    Args:
        df_exploded: DataFrame with report-element relationships
        primary_communities: Dictionary mapping elements to community IDs
    
    Returns:
        DataFrame with reports as rows and communities as columns
    """
    # Map each element to its community
    element_to_community = primary_communities
    
    # Create a new column with the community ID
    df_with_community = df_exploded.copy()
    df_with_community['community_id'] = df_with_community['data_element'].map(element_to_community)
    
    # Filter out rows where we couldn't assign a community
    df_with_community = df_with_community.dropna(subset=['community_id'])
    
    # Count elements per report and community
    report_community_counts = df_with_community.groupby(['Report', 'community_id']).size().unstack(fill_value=0)
    
    # Calculate the percentage of elements from each community
    report_community_pct = report_community_counts.div(report_community_counts.sum(axis=1), axis=0)
    
    return report_community_counts, report_community_pct

def main():
    """Main function to orchestrate the analysis workflow with report alignment."""
    start_time = time.time()
    print("Starting data co-occurrence analysis with report-community alignment...")
    
    # Ensure all required directories exist
    ensure_directories_exist()
    
    try:
        # Load the data
        print(f"Loading data from {config.INPUT_FILE}...")
        df = load_data(config.INPUT_FILE)
        
        # Create the exploded dataframe
        print("Processing data elements and reports...")
        df_exploded = create_exploded_dataframe(df)
        save_output(df_exploded, "exploded_data.csv", 
                   message=f"Processed {len(df_exploded)} report-element pairs")
        
        # Calculate co-occurrence
        print("Building co-occurrence relationships...")
        co_occurrence = calculate_cooccurrence(df_exploded)
        
        # Create co-occurrence matrix
        cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
        save_output(cooccurrence_matrix, "cooccurrence_matrix.csv", index=True)
        
        # Create network graph 
        print("Building network graph...")
        G = create_network_graph(co_occurrence, all_elements)
        print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Run fuzzy clustering with default parameters
        num_clusters = getattr(config, 'FUZZY_CLUSTERS', 8)
        fuzziness = getattr(config, 'FUZZY_FUZZINESS', 2.0)
        print(f"Running fuzzy c-means clustering with {num_clusters} clusters and fuzziness={fuzziness}...")
        
        primary_communities, membership_values = detect_communities_fuzzy_cmeans(
            G, num_clusters=num_clusters, fuzziness=fuzziness
        )
        
        # Analyze if communities are well-balanced
        balance_analysis = analyze_community_balance(primary_communities)
        save_output(pd.DataFrame([balance_analysis]), "community_balance_analysis.csv")
        
        if not balance_analysis["is_balanced"]:
            print("\nWARNING: Community balance is not ideal. Consider adjusting parameters.")
            print("Continuing analysis with current communities...")
        
        # Save primary communities
        primary_df = pd.DataFrame({
            'data_element': list(primary_communities.keys()),
            'primary_community_id': list(primary_communities.values())
        })
        save_output(primary_df, "fuzzy_primary_communities.csv")
        
        # REPORT ALIGNMENT ANALYSIS
        print("\n=== REPORT-COMMUNITY ALIGNMENT ANALYSIS ===")
        
        # Map reports to communities
        print("Analyzing how reports align with communities...")
        report_analysis = map_reports_to_communities(df_exploded, primary_communities)
        
        # Save report analysis
        report_analysis_df = pd.DataFrame.from_dict(report_analysis, orient='index')
        save_output(report_analysis_df, "report_community_analysis.csv", 
                   directory=os.path.join(config.OUTPUT_DIR, "report_analysis"),
                   index=True,
                   message=f"Saved community analysis for {len(report_analysis)} reports")
        
        # Generate community-based report assignments
        print("Generating recommended report assignments based on communities...")
        assignments = generate_community_report_assignments(report_analysis)
        save_output(assignments, "community_report_assignments.csv",
                   directory=os.path.join(config.OUTPUT_DIR, "report_analysis"),
                   message=f"Generated report assignments for {len(assignments)} communities")
        
        # Identify consolidation opportunities
        print("Identifying report consolidation opportunities...")
        consolidation = identify_report_consolidation_opportunities(report_analysis)
        if consolidation:
            save_output(pd.DataFrame(consolidation), "report_consolidation_opportunities.csv",
                       directory=os.path.join(config.OUTPUT_DIR, "report_analysis"),
                       message=f"Identified {len(consolidation)} report consolidation opportunities")
        
        # Identify splitting opportunities
        print("Identifying report splitting opportunities...")
        splitting = identify_report_splitting_opportunities(report_analysis)
        if splitting:
            save_output(pd.DataFrame(splitting), "report_splitting_opportunities.csv",
                       directory=os.path.join(config.OUTPUT_DIR, "report_analysis"),
                       message=f"Identified {len(splitting)} report splitting opportunities")
        
        # Create report-community matrix
        print("Creating report-community matrix...")
        report_community_counts, report_community_pct = create_report_community_matrix(df_exploded, primary_communities)
        save_output(report_community_counts, "report_community_matrix.csv",
                   directory=os.path.join(config.OUTPUT_DIR, "report_analysis"),
                   index=True,
                   message="Saved report-community element count matrix")
        save_output(report_community_pct, "report_community_percent_matrix.csv",
                   directory=os.path.join(config.OUTPUT_DIR, "report_analysis"),
                   index=True,
                   message="Saved report-community percentage matrix")
        
        # Print summary statistics
        print("\n=== REPORT ALIGNMENT SUMMARY ===")
        aligned_reports = sum(1 for data in report_analysis.values() 
                            if data['community_count'] == 1 or data['primary_community_percent'] >= 0.8)
        fragmented_reports = sum(1 for data in report_analysis.values() if data['is_fragmented'])
        
        print(f"Total reports analyzed: {len(report_analysis)}")
        print(f"Well-aligned reports: {aligned_reports} ({aligned_reports/len(report_analysis):.1%})")
        print(f"Fragmented reports: {fragmented_reports} ({fragmented_reports/len(report_analysis):.1%})")
        
        if consolidation:
            print(f"Consolidation opportunities: {len(consolidation)}")
        if splitting:
            print(f"Splitting opportunities: {len(splitting)}")
        
        print("\nTop report consolidation opportunities:")
        if consolidation:
            top_opportunities = sorted(consolidation, key=lambda x: x['report_count'], reverse=True)[:3]
            for i, opp in enumerate(top_opportunities, 1):
                print(f"{i}. Community {opp['community_id']}: {opp['report_count']} reports could be consolidated")
        else:
            print("None identified")
            
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis complete! All results saved. ({elapsed_time:.2f} seconds)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())