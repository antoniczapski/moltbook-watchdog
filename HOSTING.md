# HOSTING.md — MoltBook Watchdog (GCP VM + Namecheap DNS + HTTPS)

This document explains how `moltbook-watchdog.com` was hosted on a Google Cloud VM using Nginx, with DNS managed in Namecheap, and HTTPS provided by Let’s Encrypt (Certbot).

---

## Overview

We hosted a **static HTML dashboard** (`dashboard_full.html`) on a **Compute Engine VM** and served it with **Nginx**.

High-level steps:

1. Create a VM on GCP with HTTP/HTTPS allowed
2. Upload `dashboard_full.html` to the VM
3. Install and configure Nginx to serve it
4. Point the domain DNS (Namecheap) to the VM external IP
5. Enable HTTPS using Certbot (Let’s Encrypt)

---

## Prerequisites

- A GCP project with Compute Engine enabled
- A domain registered in Namecheap: `moltbook-watchdog.com`
- Local file: `dashboard_full.html` (in our case ~47MB)

---

## 1) GCP — Create and configure the VM

### 1.1 Create VM instance

In GCP Console:

- Compute Engine → VM instances → **Create instance**
- OS: Debian (Debian 12 used here)
- Machine type: any small instance is fine for static hosting
- **Firewall** (important):
  - ✅ Allow HTTP traffic
  - ✅ Allow HTTPS traffic
  - ⬜ Allow load balancer health checks (leave unchecked)

These checkboxes add network tags (`http-server`, `https-server`) and open ports **80** and **443**.

### 1.2 Ensure the VM has an External IP

In the VM details page, confirm there is an **External IPv4** address.
Example used in our setup:

- External IP: `34.118.104.29`

### 1.3 Make the External IP static (recommended)

If the external IP is *ephemeral*, it can change when the VM is stopped/restarted, breaking your domain.

In GCP Console:
- VPC network → **IP addresses**
- Find the VM external IP
- Choose **Promote to static** / **Reserve static address**
- Attach it to the VM

---

## 2) GCP — SSH into the VM and install Nginx

Open the VM → click **SSH**.

Run:

```bash
sudo apt-get update
sudo apt-get install -y nginx
sudo systemctl enable --now nginx
