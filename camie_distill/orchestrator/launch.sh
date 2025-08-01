#!/usr/bin/env bash
# One‑liner that survives reboots via systemd/cron @reboot
cd /home/ubuntu/camie_distill
exec nohup python -m camie_distill.orchestrator.scheduler \
     >> /var/log/camie_orchestrator.log 2>&1 &
