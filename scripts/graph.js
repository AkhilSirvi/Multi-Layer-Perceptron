'use strict';

const CHART_COLORS = Object.freeze({
    COST: { border: '#6366f1', background: 'rgba(99,102,241,0.2)' },
    ACCURACY: { border: '#10b981', background: 'rgba(16,185,129,0.2)' },
    TEXT: { primary: '#f8fafc', muted: '#94a3b8' },
    GRID: 'rgba(255,255,255,0.1)'
});

let chart = null, graphConfig = { scaleType: 'linear', chartType: 'line', showFill: true };
let storedGraphCost = [], storedGraphAccuracies = [];

function setGraphScale(scaleType) {
    graphConfig.scaleType = scaleType;
    document.querySelectorAll('#scaleTypeGroup .btn-chip').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.scale === scaleType));
    if (storedGraphCost.length) renderChart();
}

function setChartType(chartType) {
    graphConfig.chartType = chartType;
    document.querySelectorAll('#chartTypeGroup .btn-chip').forEach(btn =>
        btn.classList.toggle('active', btn.dataset.chart === chartType));
    if (storedGraphCost.length) renderChart();
}

function toggleFill() {
    graphConfig.showFill = !graphConfig.showFill;
    const fillBtn = document.getElementById('fillToggle');
    if (fillBtn) fillBtn.classList.toggle('active', graphConfig.showFill);
    if (storedGraphCost.length) renderChart();
}

function resetGraphData() {
    graphdata = '';
    storedGraphCost = [];
    storedGraphAccuracies = [];
    if (chart) { chart.destroy(); chart = null; }
    console.log('Graph data reset');
}

function renderChart() {
    const ctx = document.getElementById("graphcanvas").getContext("2d");
    if (chart) chart.destroy();

    let costData = [...storedGraphCost], accuracyData = [...storedGraphAccuracies];
    if (graphConfig.scaleType === 'logarithmic') {
        costData = costData.map(v => v <= 0 ? 0.001 : v);
        accuracyData = accuracyData.map(v => v <= 0 ? 0.1 : v);
    }

    chart = new Chart(ctx, {
        type: graphConfig.chartType,
        data: {
            labels: costData.map((_, i) => i + 1),
            datasets: [
                {
                    label: "Cost",
                    data: costData,
                    borderColor: CHART_COLORS.COST.border,
                    backgroundColor: graphConfig.showFill ? CHART_COLORS.COST.background : "transparent",
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
                    borderColor: CHART_COLORS.ACCURACY.border,
                    backgroundColor: graphConfig.showFill ? CHART_COLORS.ACCURACY.background : "rgba(16,185,129,0.6)",
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
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: {
                        color: CHART_COLORS.TEXT.primary,
                        font: { family: "'Poppins', sans-serif" },
                        usePointStyle: true, pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(30,41,59,0.95)',
                    titleColor: CHART_COLORS.TEXT.primary,
                    bodyColor: CHART_COLORS.TEXT.muted,
                    borderColor: 'rgba(255,255,255,0.2)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: ctx => {
                            let l = ctx.dataset.label || '';
                            if (l) l += ': ';
                            if (ctx.parsed.y != null)
                                l += ctx.dataset.yAxisID === 'y2'
                                    ? ctx.parsed.y.toFixed(1) + '%'
                                    : ctx.parsed.y.toFixed(4);
                            return l;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Training Iteration', color: CHART_COLORS.TEXT.muted, font: { family: "'Poppins', sans-serif" } },
                    ticks: { color: CHART_COLORS.TEXT.muted },
                    grid: { color: CHART_COLORS.GRID }
                },
                y1: {
                    type: graphConfig.scaleType,
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: graphConfig.scaleType === 'logarithmic' ? 'Cost (log scale)' : 'Cost',
                        color: CHART_COLORS.COST.border,
                        font: { family: "'Poppins', sans-serif" }
                    },
                    ticks: {
                        color: CHART_COLORS.COST.border,
                        callback: v => graphConfig.scaleType === 'logarithmic' ? v.toExponential(1) : v.toFixed(2)
                    },
                    grid: { color: 'rgba(99,102,241,0.1)' }
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
                        color: CHART_COLORS.ACCURACY.border,
                        font: { family: "'Poppins', sans-serif" }
                    },
                    ticks: { color: CHART_COLORS.ACCURACY.border, callback: v => v + '%' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

function graphgenerater() {
    storedGraphCost = [...graphdata.matchAll(/the cost is (\d+\.\d+)/g)].map(m => parseFloat(m[1]));
    storedGraphAccuracies = [...graphdata.matchAll(/The accuracy is (\d+)/g)].map(m => parseInt(m[1]));
    renderChart();
}
