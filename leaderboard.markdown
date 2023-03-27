---
layout: leaderboard
title: Leaderboard
permalink: /leaderboard/
---

<div style="padding-bottom: 50px">
  <canvas id="myChart"></canvas>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

<script>
  const ctx = document.getElementById('myChart');
  var chartData = new Chart(ctx, {
    type: 'line',
    options: {
        plugins: {
            title: {
              display: true,
              text: 'Robustness Evolvement on Typical Datasets and Settings',
              font: {size: 20}
            },
            legend: {
              position: 'bottom',
              labels: {
                font: {size: 14},
                padding: 20
              }
            },
            tooltip: {
              callbacks: {
                afterLabel: function (context) {
                  return context.dataset.data[context.dataIndex].title + "\n[Click for detail]";
                },
                label: function(context) {
                    label = context.dataset.label || '';
                    if (label) {
                        label += ': ';
                    }
                    if (context.parsed.y !== null) {
                        label += parseFloat(context.parsed.y * 100.).toFixed(2);
                        label += '%';
                    }
                    return label;
                }
              },
            }
        },
        scales: {
            y: {
                max: 1,
                min: 0,
                title: {
                  text: "SOTA Reported Robust Accuracy",
                  display: true,
                  font: {size: 14}
                },
                ticks: {
                    format: {
                        style: 'percent'
                    },
                    precision: 100
                }
            },
            x: {
                type: 'time',
                title: {
                  text: "Year-Month",
                  display: true,
                  font: {size: 14}
                },
                time: {
                  unit: "quarter",
                  parser: "yyyy-MM",
                  displayFormats: {
                      quarter: 'yyyy-MM'
                  },
                  tooltipFormat: "yyyy-MMM"
                }
            }
        },
        interaction: {
            mode: 'point'
        }
    },
    data: {
      datasets: [
        {% for group in site.data.sota_trend %}
        {
          label: "{{ group.groupname }}",
          data: [
              {% for dataitem in group.data %}
              {
                x: "{{ dataitem.date }}", 
                y: {{ dataitem.racc }},
                title: "{{ dataitem.title }}",
                venue: "{{ dataitem.venue }}",
                url: "{{ dataitem.url }}",
                {% if dataitem.comment != null %}
                comment: "{{ dataitem.comment }}",
                {% endif %}
                {% if group.empirical != null and group.empirical %}
                empirical: true,
                {% else %}
                empirical: false,
                {% endif %}
              },
              {% endfor %}
            ],
            tension: 0.2,
            {% if group.color != null %}
            borderColor: "{{ group.color }}",
            {% endif %}
            {% if group.dash != null %}
            borderDash: [10, 10],
            {% endif %}
        },
        {% endfor %}
      ]
    },
  });
</script>


<div class="modal fade" id="staticPaperDetailModal" tabindex="-1" role="dialog" aria-labelledby="staticPaperDetailModal" aria-hidden="true">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Detail Publication Info</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <center><span id="certifiedSpan">Certified accuracy</span><span id="empiricalSpan">Empirical accuracy (under strong attacks)</span> = <span id="paperAccuracy"></span></center>
        <hr>
        <p><b><span id="paperTitle"></span></b>. <i><span id="paperVenue"></span></i>. <a id="paperLink" href="#" target="_blank"></a> </p>
        <p id="paperComment"></p>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
      </div>
    </div>
  </div>
</div>

<script>
  chartData.options.onClick = function (event, elements) {
        if (elements.length) {
          currentDataPoint = chartData.data.datasets[elements[0].datasetIndex].data[elements[0].index];
          
          $("#paperAccuracy").text(parseFloat(currentDataPoint.y * 100.0).toFixed(2) + "%");
          if (currentDataPoint.empirical) {
            $("#empiricalSpan").show();
            $("#certifiedSpan").hide();
          } else {
            $("#empiricalSpan").hide();
            $("#certifiedSpan").show();
          }
          $("#paperTitle").text(currentDataPoint.title);
          $("#paperVenue").text(currentDataPoint.venue);
          $("#paperLink").text(currentDataPoint.url);
          $("#paperLink").attr("href", currentDataPoint.url);
          if (currentDataPoint.comment) {
            $("#paperComment").text(currentDataPoint.comment);
          } else {
            $("#paperComment").text("");
          }

          $("#staticPaperDetailModal").modal('show');
        }
      };
</script>


This page keeps track of the highest certified accuracy reported by existing papers.

The papers that are not published on conferences or journals, such as preprints, are in gray text.

For probabilistic certification, we only take the results into account if certification confidence \\(\ge 99.9\%\\).

<div class="accordion" id="accordion_leaderboard">
{% for group in site.data.board %}
  <div class="accordion-item">
    <h2 class="accordion-header" id="heading{{ forloop.index }}">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{{ forloop.index }}" aria-expanded="false" aria-controls="collapse{{ forloop.index }}">
        {{ group.setting }}
      </button>
    </h2>
    <div id="collapse{{ forloop.index }}" class="accordion-collapse collapse" aria-labelledby="heading{{ forloop.index }}">
      <div class="accordion-body">
        <table id="table{{ forloop.index }}" class='table table-striped'>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th style='min-width: 300px'>Paper Name</th>
                    <th>Reported Certified Accuracy</th>
                    <th>Certification Type</th>
                    <th>Venue</th>
                    <th>Comment</th>
                </tr>
            </thead>
            <tbody>
                {% for item in group.records %}
                <tr {% if item.venue == null %}class="preprint"{% endif %}>
                    <td>{{ forloop.index }}</td>
                    <td><a href="{{ item.link }}" target="_blank">{{ item.title }}</a></td>
                    <td>{{ item.score }}</td>
                    <td>{% if item.prob != null and item.prob %} <span class="prob-span">Probabilistic</span> {% else %} <span class="deter-span">Deterministic</span> {% endif %}
                    </td>
                    <td>{% if item.venue %} {{item.venue}} {% else %} <span style='font-style: italic;'>*preprint</span> {% endif %}
                    </td>
                    <td style='font-size:12px'>{% if item.comment %} {{item.comment}} {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% if group.text %}
        <div class="card">
        <div class="card-body">
        {{ group.text }}
        </div>
        </div>
        {% endif %}
      </div>
    </div>
  </div>
{% endfor %}
</div>

<hr>

- Want to announce your awesome SOTA result, add new leaderboard settings, or report a bug? 

Feel free to directly edit ``_data/board.yml`` in the [website repo](https://github.com/sokcertifiedrobustness/sokcertifiedrobustness.github.io/blob/master/_data/board.yml) and send a pull request.


