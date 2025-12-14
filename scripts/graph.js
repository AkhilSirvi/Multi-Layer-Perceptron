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
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99, 102, 241, 0.2)",
                    yAxisID: "y1",  // Use left Y-axis
                    tension: 0.3,
                    fill: true,
                },
                {
                    // Dataset 2: Accuracy - should increase during successful training
                    label: "Accuracy (%)",
                    data: graphaccuracies,
                    borderColor: "#10b981",
                    backgroundColor: "rgba(16, 185, 129, 0.2)",
                    yAxisID: "y2",  // Use right Y-axis
                    tension: 0.3,
                    fill: true,
                },
            ],
        },
        options: {
            responsive: true,  // Chart resizes with container
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f8fafc',  // Light text for dark theme
                        font: {
                            family: "'Poppins', sans-serif",
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Training Iteration',
                        color: '#94a3b8',
                        font: {
                            family: "'Poppins', sans-serif",
                        }
                    },
                    ticks: {
                        color: '#94a3b8',
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                    }
                },
                y1: {
                    // Left Y-axis: Cost values (dynamic scale)
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Cost',
                        color: '#6366f1',
                        font: {
                            family: "'Poppins', sans-serif",
                        }
                    },
                    ticks: {
                        color: '#6366f1',
                    },
                    grid: {
                        color: 'rgba(99, 102, 241, 0.1)',
                    }
                },
                y2: {
                    // Right Y-axis: Accuracy percentage (fixed 0-100 scale)
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: '#10b981',
                        font: {
                            family: "'Poppins', sans-serif",
                        }
                    },
                    ticks: {
                        color: '#10b981',
                    },
                    grid: {
                        drawOnChartArea: false,  // Don't draw grid lines for right axis
                    }
                },
            },
        },
    });
}