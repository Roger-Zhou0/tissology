#!/bin/sh
# Provision Docker on a dedicated Debian host. Application/model setup follows
# separately so credentials and private artifacts never enter instance metadata.
set -eu
test "$(id -u)" = 0
. /etc/os-release
test "$ID" = debian
case "$VERSION_CODENAME" in bookworm|trixie) ;; *) exit 1 ;; esac
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y --no-install-recommends ca-certificates curl git python3
install -d -m 0755 /etc/apt/keyrings
curl --fail --silent --show-error --location --retry 3 \
  https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
chmod 0644 /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: $VERSION_CODENAME
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt-get update -q
apt-get install -y --no-install-recommends \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
install -d -m 0755 /srv/tissology
touch /srv/tissology/host-ready
