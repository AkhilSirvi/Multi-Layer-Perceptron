/**
 * @fileoverview Graph Visualization Module for Training Metrics
 * 
 * Provides real-time visualization of neural network training progress
 * using Chart.js. Displays dual-axis charts with:
 *   - Cost (Loss) on the left Y-axis - should decrease during training
 *   - Accuracy (%) on the right Y-axis - should increase during training
 * 
 * Features:
 *   - Multiple scale types: Linear, Logarithmic
 *   - Multiple chart types: Line, Bar
 *   - Configurable fill/area display
 *   - Real-time updates during training
 * 
 * @module graph
 * @author Akhil Sirvi
 * @version 2.0.0
 * @requires Chart.js
 */

'use strict';

/* =============================================================================
 * CHART CONFIGURATION
 * ============================================================================= */

/**
 * Chart color scheme (matches the UI theme)
 * @constant {Object}
 */
const CHART_COLORS = Object.freeze({
    COST: {
        border: '#6366f1',           // Indigo
        background: 'rgba(99, 102, 241, 0.2)'
    },
    ACCURACY: {
        border: '#10b981',           // Emerald
        background: 'rgba(16, 185, 129, 0.2)'
    },
    TEXT: {
        primary: '#f8fafc',          // Light text
        muted: '#94a3b8'             // Muted text
    },
    GRID: 'rgba(255, 255, 255, 0.1)'
});

/** Chart.js instance reference (for cleanup/update) */
let chart = null;

/** Current graph configuration */
let graphConfig = {
    scaleType: 'linear',      // 'linear' | 'logarithmic'
    chartType: 'line',        // 'line' | 'bar'
    showFill: true            // Area fill under lines
};

/** Stored metrics for re-rendering with different options */
let storedGraphCost = [];
let storedGraphAccuracies = [];

/* =============================================================================
 * GRAPH CONTROL FUNCTIONS
 * ============================================================================= */

/**
 * Sets the Y-axis scale type and re-renders the chart
 * @param {string} scaleType - 'linear' or 'logarithmic'
 */
function setGraphScale(scaleType) {
    graphConfig.scaleType = scaleType;
    
    // Update button states
    document.querySelectorAll('#scaleTypeGroup .btn-chip').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.scale === scaleType);
    });
    
    // Re-render if data exists
    if (storedGraphCost.length > 0) {
        renderChart();
    }
}

/**
 * Sets the chart type and re-renders
 * @param {string} chartType - 'line' or 'bar'
 */
function setChartType(chartType) {
    graphConfig.chartType = chartType;
    
    // Update button states
    document.querySelectorAll('#chartTypeGroup .btn-chip').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.chart === chartType);
    });
    
    // Regenerate chart with new type
    if (storedGraphCost.length > 0) {
        renderChart();
    }
}

/**
 * Toggles the fill/area display on the chart
 */
function toggleFill() {
    graphConfig.showFill = !graphConfig.showFill;
    
    // Update button state
    const fillBtn = document.getElementById('fillToggle');
    if (fillBtn) {
        fillBtn.classList.toggle('active', graphConfig.showFill);
    }
    
    // Regenerate chart
    if (storedGraphCost.length > 0) {
        renderChart();
    }
}

/**
 * Resets the graph data and clears the chart
 */
function resetGraphData() {
    graphdata = '';
    storedGraphCost = [];
    storedGraphAccuracies = [];
    
    if (chart) {
        chart.destroy();
        chart = null;
    }
    
    console.log('Graph data reset');
}

/**
 * Renders the chart with current configuration
 * Called internally after data or settings change
 */
function renderChart() {
    // Get the 2D rendering context for the canvas element
    const ctx = document.getElementById("graphcanvas").getContext("2d");
    
    // Destroy existing chart to prevent memory leaks
    if (chart) {
        chart.destroy();
    }
    
    // For logarithmic scale, filter out zero/negative values and add small offset
    let costData = [...storedGraphCost];
    let accuracyData = [...storedGraphAccuracies];
    
    if (graphConfig.scaleType === 'logarithmic') {
        // Add small offset to prevent log(0) issues
        costData = costData.map(v => v <= 0 ? 0.001 : v);
        accuracyData = accuracyData.map(v => v <= 0 ? 0.1 : v);
    }
    
    // Create new Chart.js chart
    chart = new Chart(ctx, {
        type: graphConfig.chartType,
        data: {
            labels: Array.from({ length: costData.length }, (_, i) => i + 1),
            datasets: [
                {
                    label: "Cost",
                    data: costData,
                    borderColor: "#6366f1",
                    backgroundColor: graphConfig.showFill ? "rgba(99, 102, 241, 0.2)" : "transparent",
                    yAxisID: "y1",
                    tension: graphConfig.chartType === 'line' ? 0.3 : 0,
                    fill: graphConfig.showFill && graphConfig.chartType === 'line',
                    borderWidth: 2,
                    pointRadius: graphConfig.chartType === 'line' ? 3 : 0,
                    pointHoverRadius: 6,
                },
                {
                    label: "Accuracy (%)",
                    data: accuracyData,
                    borderColor: "#10b981",
                    backgroundColor: graphConfig.showFill ? "rgba(16, 185, 129, 0.2)" : "rgba(16, 185, 129, 0.6)",
                    yAxisID: "y2",
                    tension: graphConfig.chartType === 'line' ? 0.3 : 0,
                    fill: graphConfig.showFill && graphConfig.chartType === 'line',
                    borderWidth: 2,
                    pointRadius: graphConfig.chartType === 'line' ? 3 : 0,
                    pointHoverRadius: 6,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f8fafc',
                        font: {
                            family: "'Poppins', sans-serif",
                        },
                        usePointStyle: true,
                        pointStyle: 'circle',
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.95)',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                if (context.dataset.yAxisID === 'y2') {
                                    label += context.parsed.y.toFixed(1) + '%';
                                } else {
                                    label += context.parsed.y.toFixed(4);
                                }
                            }
                            return label;
                        }
                    }
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
                    type: graphConfig.scaleType,
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: graphConfig.scaleType === 'logarithmic' ? 'Cost (log scale)' : 'Cost',
                        color: '#6366f1',
                        font: {
                            family: "'Poppins', sans-serif",
                        }
                    },
                    ticks: {
                        color: '#6366f1',
                        callback: function(value) {
                            if (graphConfig.scaleType === 'logarithmic') {
                                return value.toExponential(1);
                            }
                            return value.toFixed(2);
                        }
                    },
                    grid: {
                        color: 'rgba(99, 102, 241, 0.1)',
                    }
                },
                y2: {
                    type: graphConfig.scaleType === 'logarithmic' ? 'logarithmic' : 'linear',
                    display: true,
                    position: 'right',
                    min: graphConfig.scaleType === 'logarithmic' ? 0.1 : 0,
                    max: 100,
                    title: {
                        display: true,
                        text: graphConfig.scaleType === 'logarithmic' ? 'Accuracy (log scale)' : 'Accuracy (%)',
                        color: '#10b981',
                        font: {
                            family: "'Poppins', sans-serif",
                        }
                    },
                    ticks: {
                        color: '#10b981',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        drawOnChartArea: false,
                    }
                },
            },
        },
    });
}

/**
 * Generates and updates the training metrics chart
 * 
 * Parses the graphdata string containing training logs and extracts:
 * - Cost values from lines like "the cost is X.XXX"
 * - Accuracy values from lines like "The accuracy is XX"
 * 
 * Creates a dual-axis chart showing training progress over iterations
 */
function graphgenerater() {
    // Regex patterns to extract metrics from the graphdata string
    const costPattern = /the cost is (\d+\.\d+)/g;
    const accuracyPattern = /The accuracy is (\d+)/g;

    let costMatches;
    let accuracyMatches;

    // Arrays to store extracted metric values
    storedGraphCost = [];
    storedGraphAccuracies = [];

    // Extract all cost values from the graphdata string
    while ((costMatches = costPattern.exec(graphdata)) !== null) {
        storedGraphCost.push(parseFloat(costMatches[1]));
    }
  
    // Extract all accuracy values from the graphdata string
    while ((accuracyMatches = accuracyPattern.exec(graphdata)) !== null) {
        storedGraphAccuracies.push(parseInt(accuracyMatches[1]));
    }
  
    // Render the chart with current configuration
    renderChart();
}