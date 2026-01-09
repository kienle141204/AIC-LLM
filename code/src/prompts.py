
import numpy as np
import torch

def get_statistics(x):
    """
    Calculate statistics for the input data.
    x: Input tensor or numpy array (B, T, N, F) or similar.
    Returns: A formatted string containing statistics.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    mean_val = np.mean(x)
    std_val = np.std(x)
    min_val = np.min(x)
    max_val = np.max(x)
    
    q25 = np.percentile(x, 25)
    median_val = np.median(x)
    q75 = np.percentile(x, 75)
    
    return f"- Mean: {mean_val:.4f}\n- Std: {std_val:.4f}\n- Min: {min_val:.4f}\n- Max: {max_val:.4f}\n- 25%: {q25:.4f}\n- 50%: {median_val:.4f}\n- 75%: {q75:.4f}"

PROMPTS = {
    "PEMS03": """PEMS03 dataset contains traffic flow data collected from Caltrans Performance Measurement System (PeMS) District 3, covering the North Central Valley area of California. This dataset includes data from hundreds of sensors aggregated into 5-minute intervals.
### Domain: Traffic flow forecasting requires capturing both spatial dependencies (road network topology) and temporal dependencies (daily and weekly patterns) to predict future traffic conditions accurately.
### Instruction: Given the historical traffic flow data and the sensor graph structure, predict the future traffic flow for all nodes in the network over the next horizon.
### Input statistics:
{statistics}""",

    "PEMS04": """PEMS04 dataset consists of traffic flow data from Caltrans Performance Measurement System (PeMS) District 4, representing the San Francisco Bay Area. It is a widely used benchmark for spatial-temporal traffic forecasting models.
### Domain: Traffic forecasting in dense urban areas like the Bay Area involves modeling complex interactions between different road segments and accounting for rush hour patterns and non-linear temporal dynamics.
### Instruction: Analyze the input historical traffic data to generating accurate forecasts for the future traffic speed and volume across the sensor network.
### Input statistics:
{statistics}""",

    "PEMS07": """PEMS07 dataset comprises traffic data from Caltrans Performance Measurement System (PeMS) District 7, covering the Los Angeles area. This dataset is characterized by a large and dense network of sensors monitoring one of the busiest traffic regions.
### Domain: Forecasting traffic in Los Angeles requires robust modeling of high-volume traffic flows, recurring congestion patterns, and the propagation of traffic waves across a large spatial graph.
### Instruction: Utilize the historical time-series data and spatial correlations to predict the traffic flow values for the upcoming time steps.
### Input statistics:
{statistics}""",

    "PEMS08": """PEMS08 dataset collects traffic flow information from Caltrans Performance Measurement System (PeMS) District 8, spanning the San Bernardino and Riverside counties. It provides granular traffic data useful for analyzing suburban and highway traffic patterns.
### Domain: Traffic prediction in this region involves understanding highway flow dynamics, identifying bottlenecks, and capturing long-term temporal trends in traffic movement.
### Instruction: Based on the sequence of historical traffic observations, predict the future traffic state for the entire network.
### Input statistics:
{statistics}"""
}
