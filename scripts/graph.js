let chart;
function graphgenerater() {
    
    const costPattern = /the cost is (\d+\.\d+)/g;
    const accuracyPattern = /The accuracy is (\d+)/g;

let costMatches;
let accuracyMatches;

const graphcost = [];
const graphaccuracies = [];

while ((costMatches = costPattern.exec(graphdata)) !== null) {
    graphcost.push(parseFloat(costMatches[1]));
}
  
  // Extract accuracy values using regex
  while ((accuracyMatches = accuracyPattern.exec(graphdata)) !== null) {
    graphaccuracies.push(parseInt(accuracyMatches[1]));
}
  
// Chart.js
  const ctx = document.getElementById("graphcanvas").getContext("2d");
  if (chart) {
    chart.destroy();
  }
  // Create the chart
chart = new Chart(ctx, {
      type: "line",
    data: {
      labels: Array.from({ length: graphcost.length }, (_, i) => i + 1),
      datasets: [
        {
          label: "Cost",
          data: graphcost,
          borderColor: "blue",
          backgroundColor: "rgba(0, 0, 255, 0.2)",
          yAxisID: "y-axis-1",
        },
        {
          label: "Accuracy (%)",
          data: graphaccuracies,
          borderColor: "red",
          backgroundColor: "rgba(255, 0, 0, 0.2)",
          yAxisID: "y-axis-2",
        },
    ],
},
options: {
    responsive: true,
    scales: {
        yAxes: [
            {
                id: "y-axis-1",
                type: "linear",
                position: "left",
            scaleLabel: {
              display: true,
              labelString: "Cost",
            },
          },
          {
            id: "y-axis-2",
            type: "linear",
            position: "right",
            ticks: {
              max: 100,
              min: 0,
            },
            scaleLabel: {
              display: true,
              labelString: "Accuracy (%)",
            },
        },
    ],
},
},
});
}