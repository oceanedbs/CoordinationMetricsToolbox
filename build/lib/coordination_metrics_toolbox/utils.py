import matplotlib.pyplot as plt

def generate_palette(n):
    """
    Generate a palette with n colors using the viridis colormap.
    
    Parameters:
    n (int): Number of colors to generate.
    
    Returns:
    list: List of colors in RGB format.
    """
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / n) for i in range(n)]
    return colors
