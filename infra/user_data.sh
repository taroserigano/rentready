#!/bin/bash
# Base provisioning for the RentReady demo instance (Amazon Linux 2023).
# Installs the runtime + reverse proxy; it does NOT deploy the app itself —
# that's a separate step (rsync/git the repo to /opt/rentready, then run
# `sudo systemctl start rentready-backend` and build the frontend), and TLS
# is a further separate step, run once the app is up and nginx is serving it
# on port 80:
#   pulumi stack output tlsSetupCommand   # prints the exact command to run
#   ssh -i rentready-key.pem ec2-user@<publicIp>
#   <paste the printed command>           # certbot gets a cert + adds the
#                                          # HTTP->HTTPS redirect to nginx
# (certbot is installed below; it isn't run automatically here because the
# hostname it needs -- <eip-with-dashes>.sslip.io -- only exists once the
# Elastic IP is allocated, which happens after this script already ran once.)
set -euxo pipefail

dnf update -y

# A 1 GB-RAM instance needs swap headroom for `npm run build` / pip installs.
if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

dnf install -y git nginx python3.11 python3.11-pip nodejs npm certbot python3-certbot-nginx

# uv: fast Python package/venv manager, used by the backend. Installed
# straight into /usr/local/bin (not symlinked from /root/.local/bin/) --
# /root isn't traversable by ec2-user, which would make a symlink there
# a dead link for the app's own service user.
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

mkdir -p /opt/rentready
chown ec2-user:ec2-user /opt/rentready

# systemd unit for the FastAPI backend (uvicorn). Deploy the app to
# /opt/rentready first, then: sudo systemctl enable --now rentready-backend
#
# CORS: nginx serves the built frontend AND proxies /api/ on this SAME
# origin (see the server block below), so the browser's own calls back to
# the API are same-origin and need no CORS allowance at all. The backend's
# CORS_ALLOWED_ORIGINS (default "*", see settings.py) only matters if this
# ever changes -- e.g. the frontend gets split out to S3/CloudFront instead
# of being served by this nginx. If that happens, set it in
# /opt/rentready/backend/.env as part of the app-deploy step:
#   CORS_ALLOWED_ORIGINS=https://<that-frontend-origin>
cat > /etc/systemd/system/rentready-backend.service <<'EOF'
[Unit]
Description=RentReady FastAPI backend
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/rentready/backend
Environment=PATH=/opt/rentready/backend/.venv/bin:/usr/local/bin:/usr/bin
ExecStart=/opt/rentready/backend/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload

# nginx: serves the built frontend + proxies /api to the backend. Points at
# /opt/rentready/frontend/dist, which won't exist until the app is deployed
# and built -- nginx will 502/404 until then, which is expected.
cat > /etc/nginx/conf.d/rentready.conf <<'EOF'
server {
    listen 80;
    server_name _;

    root /opt/rentready/frontend/dist;
    index index.html;

    # The app's own cap is settings.max_upload_mb (20). nginx's default here
    # is 1MB, which would reject a real multi-page scan with its own HTML 413
    # before the app ever sees it -- making the app-level cap dead code. Set
    # slightly above the app's so the app remains the authority and returns
    # its JSON 413.
    client_max_body_size 21m;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;

        # Both of these must be SET (not add_header'd) so a client-supplied
        # value is OVERWRITTEN rather than appended/passed through.
        # X-Forwarded-For matters specifically because uvicorn's
        # ProxyHeadersMiddleware trusts it from 127.0.0.1 and uses it to
        # rewrite request.client -- leaving it client-controlled let anyone
        # forge their identity and bypass the rate limiter entirely.
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        try_files $uri /index.html;
    }
}
EOF
rm -f /etc/nginx/conf.d/default.conf 2>/dev/null || true
systemctl enable --now nginx

# certbot + AUTO-RENEWAL.
#
# Certbot prints "Certbot has set up a scheduled task to automatically renew
# this certificate" -- on Amazon Linux 2023 that is NOT true. The dnf-packaged
# certbot ships no systemd timer (only the snap/pip installs do) and crond is
# inactive by default, so a cert issued here would simply expire after 90 days.
# Install the timer explicitly.
dnf install -y certbot python3-certbot-nginx

cat > /etc/systemd/system/certbot-renew.service <<'EOF'
[Unit]
Description=Renew Let's Encrypt certificates
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
# --deploy-hook only reloads nginx when a cert was actually renewed.
ExecStart=/usr/bin/certbot renew --quiet --deploy-hook "systemctl reload nginx"
EOF

cat > /etc/systemd/system/certbot-renew.timer <<'EOF'
[Unit]
Description=Run certbot renew twice daily (Let's Encrypt's recommendation)

[Timer]
OnCalendar=*-*-* 03,15:00:00
# Spread load on Let's Encrypt and survive a boot that missed a window.
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now certbot-renew.timer
