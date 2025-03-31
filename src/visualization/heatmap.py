"""
Functions for creating heatmap visualizations of co-occurrence data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import config

def create_heatmap(cooccurrence_matrix, output_file=None, cmap=None, dpi=None):
    """
    Create a heatmap visualization of the co-occurrence matrix.
    
    Args:
        cooccurrence_matrix (pandas.DataFrame): Co-occurrence matrix
        output_file (str, optional): Path to save the heatmap image. 
                                    If None, uses default path.
        cmap (str, optional): Colormap to use for the heatmap.
                             If None, uses config default.
        dpi (int, optional): DPI for the saved image.
                            If None, uses config default.
                            
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Use default values from config if not specified
    cmap = cmap or config.HEATMAP_CMAP
    dpi = dpi or config.HEATMAP_DPI
    
    # Create the figure and heatmap
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(cooccurrence_matrix, annot=True, cmap=cmap, linewidths=0.5)
    plt.title("Data Element Co-occurrence Matrix")
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        return None
    
    # Return the figure if not saving
    return plt.gcf()