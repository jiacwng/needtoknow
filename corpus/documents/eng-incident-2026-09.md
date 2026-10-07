+++
id = "eng-incident-2026-09"
title = "Incident report: September 2026 outage"
readers = ["group:engineering"]
fact = "edge-proxy-02"
question = "Which host caused the September 2026 outage?"
allowed_user = "oskar"
denied_user = "sofia"
+++
On 14 September 2026 the public API was unavailable for 47 minutes, from 09:12 to 09:59.

Root cause: the TLS certificate on edge-proxy-02 expired. The renewal job had failed silently
since August because a permission change blocked it from writing the new certificate. The
other proxy kept serving traffic, but the load balancer sent half of the requests to the
broken host.

Actions: alert on certificate expiry 21 days ahead, alert when the renewal job fails, and
remove a host from the load balancer when its health check fails on TLS.

Oskar Nilsson led the response. Customers received a status page update at 09:25.
