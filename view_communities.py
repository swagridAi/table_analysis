#!/usr/bin/env python3
"""
Element-Community Viewer Script

This script creates visualizations and interactive browsers to explore 
the element-community assignments from a fuzzy clustering analysis.

Usage:
  python view_communities.py --dir /path/to/results [--run run_id] [--top n]
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import webbrowser
import time

class ElementCommunityViewer:
    """Tool for viewing and exploring element-community assignments."""
    
    def __init__(self, results_dir, output_dir=None):
        """
        Initialize with path to results directory.
        
        Args:
            results_dir: Directory containing parameter sweep results
            output_dir: Directory to save visualizations and browser files
        """
        self.results_dir = results_dir
        self.output_dir = output_dir or os.path.join(results_dir, "element_viewer")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find available runs
        self.runs = self._find_available_runs()
        print(f"Found {len(self.runs)} parameter runs to analyze")
    
    def _find_available_runs(self):
        """Find available parameter runs in the results directory."""
        runs = []
        
        # Check if this is a parameter sweep directory structure
        if os.path.exists(os.path.join(self.results_dir, "all_parameter_results.csv")):
            # This is a parameter sweep results directory
            for item in os.listdir(self.results_dir):
                if os.path.isdir(os.path.join(self.results_dir, item)) and item.startswith("clusters_"):
                    run_dir = os.path.join(self.results_dir, item)
                    membership_file = os.path.join(run_dir, "fuzzy_membership_values.csv")
                    if os.path.exists(membership_file):
                        runs.append({
                            'id': item,
                            'path': run_dir,
                            'membership_file': membership_file
                        })
        else:
            # Try single run structure
            membership_file = os.path.join(self.results_dir, "fuzzy_membership_values.csv")
            if os.path.exists(membership_file):
                run_id = os.path.basename(self.results_dir)
                runs.append({
                    'id': run_id,
                    'path': self.results_dir,
                    'membership_file': membership_file
                })
                
        return runs
    
    def find_best_run(self):
        """Find the best run from a parameter sweep."""
        # Try to load parameter sweep report
        report_file = os.path.join(self.results_dir, "parameter_sweep_report.json")
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                # Get best run for opportunity score
                if 'best_parameters' in report and 'opportunity_score' in report['best_parameters']:
                    return report['best_parameters']['opportunity_score']['run_id']
            except:
                pass
        
        # If no report or can't parse, return first run
        return self.runs[0]['id'] if self.runs else None
    
    def create_element_community_browser(self, run_id=None, open_browser=True):
        """
        Create an interactive HTML browser for exploring element-community assignments.
        
        Args:
            run_id: Specific parameter run to use, or None for best run
            open_browser: Whether to automatically open the browser
            
        Returns:
            Path to the generated HTML file
        """
        if not self.runs:
            print("No runs available to analyze.")
            return None
        
        # Find run to use
        if run_id is None:
            run_id = self.find_best_run()
        
        run_info = next((r for r in self.runs if r['id'] == run_id), None)
        if not run_info:
            print(f"Run {run_id} not found. Available runs: {[r['id'] for r in self.runs]}")
            return None
        
        # Load membership values
        try:
            membership_df = pd.read_csv(run_info['membership_file'], index_col=0)
            print(f"Loaded membership data with {len(membership_df)} elements and {len(membership_df.columns)} communities")
        except Exception as e:
            print(f"Error loading membership values: {e}")
            return None
        
        # Generate HTML content
        html_content = []
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Element-Community Browser</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; }
                .filters { margin-bottom: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .community-bar { height: 20px; background: #ddd; position: relative; border-radius: 3px; overflow: hidden; }
                .community-segment { height: 100%; position: absolute; }
                .search-box { padding: 8px; width: 300px; }
                select { padding: 8px; }
                .filter-group { margin-bottom: 10px; }
                .filter-label { font-weight: bold; margin-right: 10px; }
                #tableSummary { margin-top: 30px; }
                #runInfo { margin-bottom: 20px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Element-Community Browser</h1>
            
            <div id="runInfo">
                <strong>Parameter Run:</strong> """ + run_id + """<br>
                <strong>Elements:</strong> """ + str(len(membership_df)) + """<br>
                <strong>Communities:</strong> """ + str(len(membership_df.columns)) + """
            </div>
            
            <div class="filters">
                <div class="filter-group">
                    <span class="filter-label">Filter by Table:</span>
                    <select id="tableFilter" onchange="filterElements()">
                        <option value="">All Tables</option>
        """)
        
        # Extract table names from elements
        tables = defaultdict(list)
        for element in membership_df.index:
            if isinstance(element, str) and '.' in element:
                parts = element.split('.')
                table = parts[1] if len(parts) > 1 else parts[0]
                tables[table].append(element)
        
        # Add table options
        for table in sorted(tables.keys()):
            element_count = len(tables[table])
            html_content.append(f'<option value="{table}">{table} ({element_count} elements)</option>')
        
        html_content.append("""
                    </select>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Filter by Community:</span>
                    <select id="communityFilter" onchange="filterElements()">
                        <option value="">All Communities</option>
        """)
        
        # Add community options
        for i, col in enumerate(membership_df.columns):
            comm_size = (membership_df[col] == membership_df.max(axis=1)).sum()
            html_content.append(f'<option value="{col}">Community {i} ({comm_size} primary elements)</option>')
        
        html_content.append("""
                    </select>
                    <span class="filter-label" style="margin-left: 20px;">Min Membership:</span>
                    <input type="range" id="membershipFilter" min="0" max="100" value="20" onchange="filterElements()" oninput="document.getElementById('membershipValue').textContent = this.value + '%'">
                    <span id="membershipValue">20%</span>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Search:</span>
                    <input type="text" id="searchBox" class="search-box" onkeyup="filterElements()" placeholder="Search for elements...">
                </div>
            </div>
            
            <table id="elementTable">
                <thead>
                    <tr>
                        <th>Element</th>
                        <th>Table</th>
                        <th>Primary Community</th>
                        <th>Community Distribution</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        """)
        
        # Generate table rows
        for element in membership_df.index:
            # Get primary community
            primary_community = membership_df.loc[element].idxmax()
            primary_value = membership_df.loc[element, primary_community]
            primary_comm_id = primary_community.split('_')[1]
            
            # Format element
            display_element = element
            if isinstance(element, str) and '.' in element:
                parts = element.split('.')
                if len(parts) >= 3:
                    display_element = parts[2]
                    table = parts[1]
                else:
                    table = parts[0]
                    display_element = parts[-1]
            else:
                table = ""
            
            # Create community distribution bar
            bar_html = '<div class="community-bar">'
            
            # Sort memberships by strength
            memberships = [(col, membership_df.loc[element, col]) for col in membership_df.columns]
            memberships.sort(key=lambda x: x[1], reverse=True)
            
            # Add segments to bar
            left = 0
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#d35400', '#c0392b', '#2980b9']
            
            community_data = []  # For data attributes
            
            for i, (comm, value) in enumerate(memberships):
                if value > 0.05:  # Only show significant memberships
                    width = value * 100
                    comm_id = int(comm.split('_')[1])
                    color = colors[comm_id % len(colors)]
                    bar_html += f'<div class="community-segment" style="left: {left}%; width: {width}%; background-color: {color};" title="{comm}: {value:.2f}"></div>'
                    left += width
                    
                    if value >= 0.2:
                        community_data.append(f"{comm}:{value:.2f}")
            
            bar_html += '</div>'
            
            # Create details with all membership values
            details = []
            for comm, value in memberships:
                if value >= 0.1:  # Only show significant memberships
                    comm_id = comm.split('_')[1]
                    details.append(f"{comm} ({comm_id}): {value:.2f}")
            
            details_html = ', '.join(details)
            
            # Create row
            html_content.append(f"""
                <tr data-table="{table}" data-communities="{','.join([c for c, v in memberships if v >= 0.2])}" data-community-values="{','.join(community_data)}">
                    <td>{display_element}</td>
                    <td>{table}</td>
                    <td>{primary_comm_id} ({primary_value:.2f})</td>
                    <td>{bar_html}</td>
                    <td>{details_html}</td>
                </tr>
            """)
        
        # Add table summary section
        html_content.append("""
                </tbody>
            </table>
            
            <div id="tableSummary">
                <h2>Table Community Summary</h2>
                <table id="tableCommunitySummary">
                    <thead>
                        <tr>
                            <th>Table</th>
                            <th>Elements</th>
                            <th>Primary Communities</th>
                            <th>Community Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
        """)
        
        # Generate table summary
        for table, elements in sorted(tables.items(), key=lambda x: len(x[1]), reverse=True):
            table_df = membership_df.loc[elements]
            
            # Count primary communities
            primary_communities = [table_df.loc[element].idxmax() for element in elements]
            community_counts = Counter(primary_communities)
            
            # Format primary communities
            primary_comm_text = ', '.join([f"{comm.split('_')[1]}: {count}" for comm, count in community_counts.most_common(3)])
            if len(community_counts) > 3:
                primary_comm_text += f", +{len(community_counts)-3} more"
            
            # Create community distribution bar
            bar_html = '<div class="community-bar">'
            
            # Calculate average memberships for this table
            avg_memberships = table_df.mean()
            memberships = [(col, avg_memberships[col]) for col in avg_memberships.index]
            memberships.sort(key=lambda x: x[1], reverse=True)
            
            # Add segments to bar
            left = 0
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#d35400', '#c0392b', '#2980b9']
            
            for i, (comm, value) in enumerate(memberships):
                if value > 0.05:  # Only show significant memberships
                    width = value * 100
                    comm_id = int(comm.split('_')[1])
                    color = colors[comm_id % len(colors)]
                    bar_html += f'<div class="community-segment" style="left: {left}%; width: {width}%; background-color: {color};" title="{comm}: {value:.2f}"></div>'
                    left += width
            
            bar_html += '</div>'
            
            # Add table row
            html_content.append(f"""
                <tr>
                    <td>{table}</td>
                    <td>{len(elements)}</td>
                    <td>{primary_comm_text}</td>
                    <td>{bar_html}</td>
                </tr>
            """)
        
        # Add JavaScript for filtering
        html_content.append("""
                </tbody>
            </table>
            </div>
            
            <script>
                function filterElements() {
                    // Get filter values
                    const tableFilter = document.getElementById('tableFilter').value;
                    const communityFilter = document.getElementById('communityFilter').value;
                    const membershipFilter = document.getElementById('membershipFilter').value / 100;
                    const searchText = document.getElementById('searchBox').value.toLowerCase();
                    
                    // Get all rows
                    const rows = document.getElementById('elementTable').getElementsByTagName('tbody')[0].rows;
                    
                    // Filter rows
                    let visibleCount = 0;
                    
                    for (let i = 0; i < rows.length; i++) {
                        let row = rows[i];
                        let showRow = true;
                        
                        // Filter by table
                        if (tableFilter && row.getAttribute('data-table') !== tableFilter) {
                            showRow = false;
                        }
                        
                        // Filter by community and membership threshold
                        if (communityFilter) {
                            const communityValues = row.getAttribute('data-community-values');
                            if (communityValues) {
                                const communityData = {};
                                communityValues.split(',').forEach(item => {
                                    const [comm, value] = item.split(':');
                                    communityData[comm] = parseFloat(value);
                                });
                                
                                // Check if this community exists and meets threshold
                                if (!communityData[communityFilter] || communityData[communityFilter] < membershipFilter) {
                                    showRow = false;
                                }
                            } else {
                                showRow = false;
                            }
                        }
                        
                        // Filter by search text
                        if (searchText) {
                            const text = row.textContent.toLowerCase();
                            if (!text.includes(searchText)) {
                                showRow = false;
                            }
                        }
                        
                        // Show/hide row
                        row.style.display = showRow ? '' : 'none';
                        if (showRow) {
                            visibleCount++;
                        }
                    }
                    
                    // Update visible count in runInfo
                    const runInfo = document.getElementById('runInfo');
                    const countText = runInfo.innerHTML.split('<br>')[2];
                    if (countText) {
                        runInfo.innerHTML = runInfo.innerHTML.replace(
                            countText, 
                            `<strong>Elements:</strong> ${visibleCount} shown (${rows.length} total)`
                        );
                    }
                }
                
                // Initial filtering
                filterElements();
            </script>
        </body>
        </html>
        """)
        
        # Write HTML file
        html_path = os.path.join(self.output_dir, f"element_community_browser_{run_id}.html")
        with open(html_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        print(f"Created element-community browser at {html_path}")
        
        # Open in browser if requested
        if open_browser:
            webbrowser.open('file://' + os.path.abspath(html_path))
        
        return html_path
    
    def create_community_visualizations(self, run_id=None, top_elements=20):
        """
        Create visualizations showing element-community memberships.
        
        Args:
            run_id: Specific parameter run to use, or None for best run
            top_elements: Number of top elements to show per community
        """
        if not self.runs:
            print("No runs available to analyze.")
            return None
        
        # Find run to use
        if run_id is None:
            run_id = self.find_best_run()
        
        run_info = next((r for r in self.runs if r['id'] == run_id), None)
        if not run_info:
            print(f"Run {run_id} not found. Available runs: {[r['id'] for r in self.runs]}")
            return None
        
        # Load membership values
        try:
            membership_df = pd.read_csv(run_info['membership_file'], index_col=0)
            print(f"Loaded membership data with {len(membership_df)} elements and {len(membership_df.columns)} communities")
            
            # Check if we need to transpose the DataFrame (more communities than elements)
            if len(membership_df.columns) > len(membership_df):
                print("Transposing membership matrix (more communities than elements detected)")
                membership_df = membership_df.T
                print(f"After transposing: {len(membership_df)} elements and {len(membership_df.columns)} communities")
            
            # Ensure index is string type to prevent iteration errors
            membership_df.index = membership_df.index.astype(str)
        except Exception as e:
            print(f"Error loading membership values: {e}")
            return None
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create table-community heatmap
        self._create_table_community_heatmap(membership_df, run_id, viz_dir)
        
        # Create top elements per community
        self._create_top_elements_per_community(membership_df, run_id, viz_dir, top_n=top_elements)
        
        # Create community size comparison
        self._create_community_size_comparison(membership_df, run_id, viz_dir)
        
        print(f"Created community visualizations in {viz_dir}")
        return viz_dir
    
    def _create_table_community_heatmap(self, membership_df, run_id, viz_dir):
        """Create heatmap showing table-community relationships."""
        # Ensure all data is numeric
        for col in membership_df.columns:
            membership_df[col] = pd.to_numeric(membership_df[col], errors='coerce')
        
        # Group by table
        tables = {}
        for element in membership_df.index:
            element_str = str(element)  # Ensure element is string
            if '.' in element_str:
                parts = element_str.split('.')
                table = parts[1] if len(parts) > 1 else parts[0]
                if table not in tables:
                    tables[table] = []
                tables[table].append(element)
        
        # If no tables were identified, use another grouping approach
        if not tables:
            print("No table structure detected in element names, using prefix grouping")
            for element in membership_df.index:
                element_str = str(element)
                # Try to extract a prefix (first few characters)
                prefix = element_str[:3] if len(element_str) > 3 else element_str
                if prefix not in tables:
                    tables[prefix] = []
                tables[prefix].append(element)
        
        # Create table-level heatmap
        table_memberships = pd.DataFrame(index=tables.keys(), columns=membership_df.columns)
        for table, elements in tables.items():
            try:
                table_df = membership_df.loc[elements]
                table_memberships.loc[table] = table_df.mean()
            except Exception as e:
                print(f"Error processing table {table}: {e}")
        
        # Ensure all values are numeric
        table_memberships = table_memberships.astype(float)
        
        # Sort tables by their primary community
        try:
            primary_communities = table_memberships.idxmax(axis=1)
            table_memberships = table_memberships.loc[table_memberships.index.sort_values(
                key=lambda x: primary_communities.get(x, 0))]
        except Exception as e:
            print(f"Error sorting tables: {e}")
        
        # Create heatmap
        plt.figure(figsize=(12, max(10, len(tables) * 0.3)))
        try:
            ax = sns.heatmap(table_memberships, cmap="YlGnBu", annot=True, fmt=".2f",
                        cbar_kws={'label': 'Average Membership'})
            
            # Format community labels safely
            try:
                # Check if columns are like 'community_0' or just numbers
                if all(isinstance(col, str) and '_' in col for col in table_memberships.columns):
                    ax.set_xticklabels([f"C{col.split('_')[1]}" for col in table_memberships.columns])
                else:
                    ax.set_xticklabels([f"C{col}" for col in table_memberships.columns])
            except Exception as e:
                print(f"Warning: Could not format community labels: {e}")
            
            plt.title(f"Table-Community Membership Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{run_id}_table_community_heatmap.png"), dpi=300)
        except Exception as e:
            print(f"Error creating heatmap: {e}")
        finally:
            plt.close()
    
    def _create_top_elements_per_community(self, membership_df, run_id, viz_dir, top_n=20):
        """Create plots showing top elements in each community."""
        # Ensure all data is numeric
        for col in membership_df.columns:
            membership_df[col] = pd.to_numeric(membership_df[col], errors='coerce')
        
        # For each community
        for i, community in enumerate(membership_df.columns):
            try:
                # Handle both string and numeric community identifiers
                if isinstance(community, str) and '_' in community:
                    comm_id = community.split('_')[1]
                else:
                    comm_id = str(community)  # Convert numeric community ID to string
                
                # Get elements sorted by membership
                top_elements = membership_df[community].sort_values(ascending=False).head(top_n)
                
                # Skip if no elements
                if len(top_elements) == 0:
                    continue
                
                # Clean up element names for display
                display_names = []
                tables = []
                for element in top_elements.index:
                    element_str = str(element)
                    if '.' in element_str:
                        parts = element_str.split('.')
                        if len(parts) >= 3:
                            name = parts[2]
                            table = parts[1]
                        else:
                            name = parts[-1]
                            table = parts[0]
                    else:
                        name = element_str
                        table = ""
                    
                    display_names.append(name)
                    tables.append(table)
                
                # Create bar chart
                plt.figure(figsize=(12, max(8, len(top_elements) * 0.4)))
                bars = plt.barh(display_names, top_elements.values)
                
                # Color by table
                unique_tables = list(set(tables))
                table_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_tables)))
                table_color_map = {table: table_colors[i] for i, table in enumerate(unique_tables)}
                
                for i, table in enumerate(tables):
                    bars[i].set_color(table_color_map[table])
                
                # Add legend
                legend_patches = [plt.Rectangle((0,0),1,1, color=table_color_map[table]) for table in unique_tables]
                plt.legend(legend_patches, unique_tables, loc='lower right')
                
                plt.title(f"Top {top_n} Elements in Community {comm_id}")
                plt.xlabel("Membership Value")
                plt.xlim(0, 1)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{run_id}_community_{comm_id}_top_elements.png"), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error creating visualization for community {community}: {e}")
    
    def _create_community_size_comparison(self, membership_df, run_id, viz_dir):
        """Create visualization comparing community sizes."""
        # Count primary elements per community
        primary_counts = {}
        for i, community in enumerate(membership_df.columns):
            # Handle both string and numeric community identifiers
            if isinstance(community, str) and '_' in community:
                comm_id = int(community.split('_')[1])
            else:
                comm_id = i  # Use the column index as the community ID
            
            count = (membership_df[community] == membership_df.max(axis=1)).sum()
            primary_counts[comm_id] = count
        
        # Sort by community ID
        sorted_counts = {k: primary_counts[k] for k in sorted(primary_counts.keys())}
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_counts.keys(), sorted_counts.values())
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}', ha='center', va='bottom')
        
        plt.title(f"Community Sizes (Primary Elements)")
        plt.xlabel("Community ID")
        plt.ylabel("Number of Elements")
        plt.xticks(list(sorted_counts.keys()))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{run_id}_community_sizes.png"), dpi=300)
        plt.close()
    
    def create_all(self, run_id=None, top_elements=20, open_browser=True):
        """Create all visualizations and browser for a run."""
        if not self.runs:
            print("No runs available to analyze.")
            return None
        
        # Find run to use
        if run_id is None:
            run_id = self.find_best_run()
            if run_id is None and self.runs:
                run_id = self.runs[0]['id']
        
        print(f"Creating visualizations and browser for run: {run_id}")
        
        # Create browser
        browser_path = self.create_element_community_browser(run_id, open_browser=open_browser)
        
        # Create visualizations
        viz_dir = self.create_community_visualizations(run_id, top_elements=top_elements)
        
        return {
            'browser_path': browser_path,
            'viz_dir': viz_dir,
            'run_id': run_id
        }
    
    def list_runs(self):
        """List available parameter runs."""
        print("\nAvailable parameter runs:")
        print("-" * 50)
        for i, run in enumerate(self.runs):
            print(f"{i+1}. {run['id']}")
        print("-" * 50)

def main():
    """Run the element community viewer tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='View and explore element-community assignments')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing parameter sweep results')
    parser.add_argument('--run', type=str, help='Specific run ID to analyze (default: best run)')
    parser.add_argument('--top', type=int, default=20, help='Number of top elements to show per community (default: 20)')
    parser.add_argument('--output', type=str, help='Output directory (default: inside results directory)')
    parser.add_argument('--list', action='store_true', help='List available runs and exit')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t automatically open browser')
    
    args = parser.parse_args()
    
    # Create viewer
    viewer = ElementCommunityViewer(args.dir, args.output)
    
    # List runs if requested
    if args.list:
        viewer.list_runs()
        return 0
    
    # Create visualizations and browser
    viewer.create_all(args.run, args.top, not args.no_browser)
    
    return 0

if __name__ == "__main__":
    exit(main())