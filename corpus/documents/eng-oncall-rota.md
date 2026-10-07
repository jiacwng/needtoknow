+++
id = "eng-oncall-rota"
title = "On-call rota and escalation"
readers = ["group:engineering"]
fact = "#inc-bridge-lyon"
question = "Which chat channel is used to escalate an incident?"
allowed_user = "julie"
denied_user = "ben"
+++
One engineer is on call each week, from Monday 09:00 to the next Monday 09:00. The rota is in
the shared engineering calendar.

When an alert fires, the on-call engineer acknowledges it within 15 minutes. If the problem
affects customers, open the channel #inc-bridge-lyon and post a first update there. The
engineering lead joins any incident that lasts more than 30 minutes.

After the incident, the on-call engineer writes a report within five working days.

On-call shifts are paid as an allowance on top of salary.
