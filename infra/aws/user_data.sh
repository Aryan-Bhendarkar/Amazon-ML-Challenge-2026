#!/bin/bash
# EC2 bootstrap for the AMLC dev box (runs once as root at first boot). Log: /var/log/amlc-bootstrap.log
exec > >(tee -a /var/log/amlc-bootstrap.log) 2>&1
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
BUCKET="__AMLC_BUCKET__"
U=ubuntu

apt-get update -y
apt-get install -y git git-lfs unzip zip htop tmux jq build-essential python-is-python3 python3-venv python3-pip nvtop pigz

# AWS CLI v2 (the DLAMI already has one; reinstall/update is harmless)
curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q -o /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update

# GitHub CLI (for cloning the private repo: `gh auth login`)
mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /etc/apt/keyrings/githubcli-archive-keyring.gpg
chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
apt-get update -y && apt-get install -y gh

# uv (Python/venv manager) + Claude Code (native installer) for the ubuntu user
sudo -u $U bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u $U bash -lc 'curl -fsSL https://claude.ai/install.sh | bash' || echo "claude install failed - rerun manually"

# 16 GB swap: a safety net against OOM kills during big joins
if [ ! -f /swapfile ]; then fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile && echo '/swapfile none swap sw 0 0' >> /etc/fstab; fi

# Environment for every login shell
cat > /etc/profile.d/amlc.sh <<ENV
export AMLC_BUCKET=$BUCKET
export PYTHONUTF8=1
export PATH="\$HOME/.local/bin:\$PATH"
ENV

# ---------------------------------------------------------------- auto-stop (protects credits)
# Stops the instance after 20 min with: no SSH session AND 5-min load < 0.3 AND no ~/.keepalive file.
cat > /usr/local/bin/amlc-autostop.sh <<'AS'
#!/bin/bash
STATE=/var/tmp/amlc-idle-count
[ -f /home/ubuntu/.keepalive ] && { echo 0 > $STATE; exit 0; }
SSH=$(ss -Htn state established '( sport = :22 )' | wc -l)
LOAD=$(cut -d' ' -f2 /proc/loadavg)
if [ "$SSH" -eq 0 ] && awk "BEGIN{exit !($LOAD < 0.3)}"; then
  N=$(( $(cat $STATE 2>/dev/null || echo 0) + 1 )); echo $N > $STATE
  if [ "$N" -ge 4 ]; then logger "amlc-autostop: idle 20 min -> shutdown"; echo 0 > $STATE; /sbin/shutdown -h now; fi
else
  echo 0 > $STATE
fi
AS
chmod +x /usr/local/bin/amlc-autostop.sh
cat > /etc/systemd/system/amlc-autostop.service <<'SV'
[Unit]
Description=AMLC idle auto-stop
[Service]
Type=oneshot
ExecStart=/usr/local/bin/amlc-autostop.sh
SV
cat > /etc/systemd/system/amlc-autostop.timer <<'TM'
[Unit]
Description=Run AMLC idle check every 5 minutes
[Timer]
OnBootSec=15min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
TM
systemctl daemon-reload && systemctl enable --now amlc-autostop.timer

# project setup script (interactive part) for the ubuntu user
cat > /home/ubuntu/box_setup.sh <<'BOXSETUP'
__BOX_SETUP__
BOXSETUP
chown ubuntu:ubuntu /home/ubuntu/box_setup.sh && chmod +x /home/ubuntu/box_setup.sh

cat > /etc/motd <<'MOTD'
 ===== Amazon ML Challenge 2026 dev box =====
 First time:  bash ~/box_setup.sh      (GitHub login, venv, data, caches, tests)
 Claude Code: cd ~/amlc && claude
 Long job?    tmux new -s job   |  keep box alive while you're away: touch ~/.keepalive (rm it after!)
 Auto-stop:   after 20 min with no SSH + idle CPU (credits protection)
MOTD
touch /var/log/amlc-bootstrap.done
echo "bootstrap complete"
