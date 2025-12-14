/**
 * Graph Visualization Module
 * 
 * This module handles the visualization of training metrics using Chart.js
 * Displays two metrics on a dual-axis line chart:
 * - Cost (Loss) on the left Y-axis - should decrease during training
 * - Accuracy (%) on the right Y-axis - should increase during training
 * 
 * @module graph
 */

/** Reference to the Chart.js instance (stored for cleanup/update purposes) */
let chart;

/**
 * Generates and updates the training metrics chart
 * 
 * Parses the graphdata string containing training logs and extracts:
 * - Cost values from lines like "the cost is X.XXX"
 * - Accuracy values from lines like "The accuracy is XX"
 * 
 * Creates a dual-axis line chart showing training progress over iterations
 * Destroys the previous chart instance before creating a new one to prevent
 * memory leaks and rendering issues
 */
function graphgenerater() {
    // Regex patterns to extract metrics from the graphdata string
    const costPattern = /the cost is (\d+\.\d+)/g;      // Matches: "the cost is 2.345"
    const accuracyPattern = /The accuracy is (\d+)/g;   // Matches: "The accuracy is 85"

    let costMatches;
    let accuracyMatches;

    // Arrays to store extracted metric values
    const graphcost = [];        // Cost values for each training iteration
    const graphaccuracies = [];  // Accuracy values for each training iteration

    // Extract all cost values from the graphdata string
    while ((costMatches = costPattern.exec(graphdata)) !== null) {
        graphcost.push(parseFloat(costMatches[1]));
    }
  
    // Extract all accuracy values from the graphdata string
    while ((accuracyMatches = accuracyPattern.exec(graphdata)) !== null) {
        graphaccuracies.push(parseInt(accuracyMatches[1]));
    }
  
    // Get the 2D rendering context for the canvas element
    const ctx = document.getElementById("graphcanvas").getContext("2d");
    
    // Destroy existing chart to prevent memory leaks and overlapping charts
    if (chart) {
        chart.destroy();
    }
    
    // Create new Chart.js line chart with dual Y-axes
    chart = new Chart(ctx, {
        type: "line",  // Line chart for showing trends over time
        data: {
            // X-axis labels: iteration numbers (1, 2, 3, ...)
            labels: Array.from({ length: graphcost.length }, (_, i) => i + 1),
            datasets: [
                {
                    // Dataset 1: Cost (Loss) - should decrease during successful training
                    label: "Cost",
                    data: graphcost,
                    borderColor: "blue",
                    backgroundColor: "rgba(0, 0, 255, 0.2)",
                    yAxisID: "y-axis-1",  // Use left Y-axis
                },
                {
                    // Dataset 2: Accuracy - should increase during successful training
                    label: "Accuracy (%)",
                    data: graphaccuracies,
                    borderColor: "red",
                    backgroundColor: "rgba(255, 0, 0, 0.2)",
                    yAxisID: "y-axis-2",  // Use right Y-axis
                },
            ],
        },
        options: {
            responsive: true,  // Chart resizes with container
            scales: {
                yAxes: [
                    {
                        // Left Y-axis: Cost values (dynamic scale)
                        id: "y-axis-1",
                        type: "linear",
                        position: "left",
                        scaleLabel: {
                            display: true,
                            labelString: "Cost",  // Axis label
                        },
                    },
                    {
                        // Right Y-axis: Accuracy percentage (fixed 0-100 scale)
                        id: "y-axis-2",
                        type: "linear",
                        position: "right",
                        ticks: {
                            max: 100,  // Maximum accuracy is 100%
                            min: 0,    // Minimum accuracy is 0%
                        },
                        scaleLabel: {
                            display: true,
                            labelString: "Accuracy (%)",  // Axis label
                        },
                    },
                ],
            },
        },
    });
}