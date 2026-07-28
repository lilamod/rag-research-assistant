# Deploying on Oracle Cloud (Always Free) — free-forever, unlimited local embeddings

This guide sets up the backend on Oracle Cloud's Always Free tier, using
**local embeddings** (`EMBEDDING_PROVIDER=local`) so there's no external
embeddings API, no rate limits, and no per-request cost — ever, at any
scale. The frontend stays on Vercel as before.

Oracle's Always Free tier gives you a real, permanent VM (not a trial) with
up to 4 OCPUs and 24GB RAM (Ampere A1 / ARM), which is comfortably enough to
run `sentence-transformers` + `torch`.

**Card note:** Oracle requires a valid card at signup for identity
verification (a temporary authorization, not a charge). Always Free
resources are never billed unless you manually upgrade your account to a
paid tier yourself — this setup only uses Always Free resources.

---

## 1. Create the Oracle Cloud account + VM

1. Sign up at https://signup.oraclecloud.com (choose the "Always Free" tier
   option; you'll need to verify identity with a card as noted above).
2. Once in the console: **Compute → Instances → Create Instance**.
3. Name it (e.g. `rag-backend`).
4. **Image and shape:**
   - Image: **Ubuntu 22.04** (or latest Ubuntu LTS available)
   - Shape: click "Change Shape" → **Ampere → VM.Standard.A1.Flex** → set
     to the max Always Free allocation (typically 4 OCPUs / 24GB RAM, but
     Oracle may show your specific free limits — use whatever it offers
     under "Always Free eligible").
5. **Networking:** use the default VCN it offers to create, and make sure
   **"Assign a public IPv4 address"** is checked — this gives you a stable
   public IP that doesn't change on reboot.
6. **Add SSH keys:** let it generate a key pair for you, or upload your own
   public key. Download the private key if generated — you'll need it to
   SSH in.
7. Click **Create**. Wait a couple of minutes for it to boot, then note
   its **public IP address** from the instance details page.

## 2. Open the firewall (two layers — both need doing)

Oracle Cloud has **two separate firewalls** you must open, or nothing will
be reachable even if your app is running fine:

**A. Oracle's Security List (cloud-level firewall)**
1. Go to your instance's **VCN → Security Lists → Default Security List**.
2. Add **Ingress Rules** for:
   - Source `0.0.0.0/0`, TCP, destination port `80` (HTTP, for Let's
     Encrypt's certificate challenge)
   - Source `0.0.0.0/0`, TCP, destination port `443` (HTTPS)
   - (Port 22/SSH is usually already open by default)

**B. The VM's own OS-level firewall (iptables)**
Ubuntu images on Oracle ship with fairly strict `iptables` rules by default.
SSH in first:
```bash
ssh -i /path/to/your-key.pem ubuntu@<your-vm-public-ip>
```
Then open the same ports at the OS level:
```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save   # persist across reboots (Ubuntu)
```

## 3. Get a free hostname (needed for HTTPS)

Let's Encrypt (used automatically by Caddy in this setup) issues
certificates for domain names, not bare IP addresses. If you don't already
own a domain, use a free dynamic DNS service:

1. Go to https://www.duckdns.org, sign in (GitHub/Google login), and create
   a subdomain, e.g. `myragapp` → gives you `myragapp.duckdns.org`.
2. Point it at your Oracle VM's public IP (paste the IP into DuckDNS's
   dashboard and hit "update" — it's a static IP, so you only need to do
   this once).

## 4. Install Docker on the VM

Still SSH'd into the VM:
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Let your user run docker without sudo, and make sure Docker starts on boot
sudo usermod -aG docker $USER
sudo systemctl enable docker
newgrp docker   # or log out/in
```

## 5. Get the code onto the VM

```bash
git clone https://github.com/lilamod/rag-research-assistant.git
cd rag-research-assistant
```

Create your real `.env` from the example:
```bash
cp .env.example .env
nano .env
```
Set at minimum:
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CORS_ORIGINS=https://your-app.vercel.app
```
(No `OPENAI_API_KEY` or `VOYAGE_API_KEY` needed at all with local embeddings.)

Edit `Caddyfile` and replace the placeholder with your real DuckDNS hostname:
```bash
nano Caddyfile
# change "yourdomain.duckdns.org" to e.g. "myragapp.duckdns.org"
```

## 6. Build and run

```bash
docker compose up -d --build
```
First build takes a few minutes (installing torch + downloading the
embedding model). Check it's healthy:
```bash
docker compose logs -f backend
```
You should see Uvicorn come up with no errors. Caddy will automatically
fetch a Let's Encrypt certificate for your domain the first time it gets a
request — check its logs too if HTTPS doesn't work immediately:
```bash
docker compose logs -f caddy
```

Test it:
```bash
curl https://myragapp.duckdns.org/api/health
# should return {"status":"ok"}
```

## 7. Point the frontend at it

In Vercel → your frontend project → Settings → Environment Variables:
```
VITE_API_URL=https://myragapp.duckdns.org
```
Redeploy the frontend (env var changes need a rebuild to take effect).

## 8. Confirm it survives a reboot

This is the whole point of `restart: unless-stopped` + `systemctl enable
docker` — test it once so you're confident:
```bash
sudo reboot
```
Wait a minute, then from your own machine:
```bash
curl https://myragapp.duckdns.org/api/health
```
It should come back on its own with no manual steps.

---

## Updating the app later

```bash
cd rag-research-assistant
git pull
docker compose up -d --build
```

## Troubleshooting

- **"Failed to fetch" from the frontend, but `curl` works from your own
  terminal:** almost always CORS — double check `CORS_ORIGINS` in `.env`
  matches your Vercel URL exactly, then `docker compose restart backend`.
- **Caddy can't get a certificate:** usually means port 80 isn't actually
  reachable from the internet — re-check both firewall layers in step 2.
  `curl http://myragapp.duckdns.org` (plain HTTP, no S) from your own
  machine should at least get *some* response if port 80 is open.
- **Backend container keeps restarting:** `docker compose logs backend` to
  see the actual Python traceback — same debugging approach as the Render
  deploy, just read from Docker logs instead of the Render dashboard.
