---
layout: leaderboard
title: Leaderboard
permalink: /leaderboard/
---

This page keeps track of highest certified accuracy reported by existing papers.

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

Feel free to directly edit ``_data/board.yml`` in the website repo and send a pull request.


