# Cloud setup: EC2 dev box + Claude Code on the box

**Decision (25 Sep):** each member gets an **EC2 `r7i.xlarge`** (4 vCPU, 32 GB RAM, about $0.27/h) in **ap-south-1 (Mumbai)**, in their own AWS account.
- Claude Code runs **on the box**, next to the data. VS Code (Remote-SSH) or the Claude desktop app's SSH connection is the window into it.
- The laptop is only a thin client.
- GPU work (bi-/cross-encoder) uses a `g6.xlarge` (NVIDIA L4, 24 GB) once the GPU quota is approved. Kaggle's free 2×T4 is the fallback.

Why not SageMaker:
- The free trial is m5.xlarge (16 GB) / t3.medium.
- Each instance-hour is about 30% pricier.
- New accounts often have a 0 quota for ml.* instances.
- Studio adds setup overhead.

We need big RAM and a fast edit→run loop, which EC2 gives directly.

Why the Paid plan is required: the Free plan only allows < xlarge instances (max 2 vCPU / 8 GB). Upgrading keeps the credits. Budget alerts plus auto-stop protect the card.

| instance | vCPU / RAM / GPU | ≈ $/h | use | quota needed |
|---|---|---|---|---|
| r7i.xlarge | 4 / 32 GB | 0.27 | default dev box (fits the new-account 5-vCPU default) | none |
| r7i.2xlarge | 8 / 64 GB | 0.53 | full test runs, big LightGBM | standard ≥ 8 |
| r7i.4xlarge | 16 / 128 GB | 1.06 | final full pipeline | standard ≥ 16 |
| g6.xlarge | 4 / 16 GB / L4 24 GB | ~0.8–1.0 | encoders (train + inference) | G/VT ≥ 4 |

Also budget ~$0.45/day for the 150 GB disk while stopped. With $100–200 per member, money is not the constraint; quotas and time are.

## Step 1: AWS console, each member, ~15 min (do it NOW: quota approvals take hours)
1. Sign in and set the region (top-right) to **Asia Pacific (Mumbai)**.
2. Billing → Free Tier → **Upgrade plan → Upgrade account** (the credits carry over).
3. Console Home → "Explore AWS" widget. Do the 5 tasks for **+$100**:
   - launch a t3.micro EC2 instance (terminate it after)
   - one prompt in the Bedrock playground
   - create a budget
   - a Lambda web app
   - an RDS database (delete it after)
4. Note your 12-digit account ID (top-right menu) and send it to Aryan (for data sharing).

## Step 2: laptop, once (Windows PowerShell, repo root)
```powershell
winget install -e --id Amazon.AWSCLI        # skip if `aws --version` works; reopen the terminal after
aws login --region ap-south-1               # browser sign-in with your console user (AWS CLI >= 2.32)
#   (older CLI: aws configure  -> an access key of an IAM user with AdministratorAccess)
$env:AMLC_MEMBER = "aryan"                  # your short name (used in resource names)
powershell -ExecutionPolicy Bypass -File infra\aws\setup_account.ps1 -Email you@example.com
```
`setup_account.ps1` sets up:
- budget e-mails at 25/50/80/100% of $100
- quota requests: 32 standard vCPU and 8 GPU vCPU
- a private bucket `amlc26-<account>`
- an EC2 role
- an SSH key in `~/.ssh`
- a security group that allows SSH only from your current IP

## Step 3: data, once (Aryan)
```powershell
powershell -ExecutionPolicy Bypass -File infra\aws\upload_data.ps1
powershell -ExecutionPolicy Bypass -File infra\aws\share_bucket.ps1 -Accounts <teammate1>,<teammate2>,<teammate3>
```

## Step 4: launch the box
```powershell
powershell -ExecutionPolicy Bypass -File infra\aws\devbox.ps1 up            # r7i.xlarge
```
Bootstrap takes ~4 min: CLI tools, uv, Claude Code, gh, 16 GB swap, auto-stop.
The script writes `Host amlc-box` into `~/.ssh/config`.

## Step 5: connect and finish setup on the box
- **VS Code:** Remote Explorer → `amlc-box` → Connect. Open the terminal and run:
  ```bash
  bash ~/box_setup.sh            # GitHub login (device code), venv, data, caches, tests (~10 min)
  cd ~/amlc && claude            # first run prints a login URL -> open it on the laptop
  ```
  Then open the folder `/home/ubuntu/amlc` in VS Code.
  **Run Claude Code as the CLI in the VS Code terminal.** The VS Code *extension* has had Remote-SSH bugs, while the CLI works reliably over SSH.
- **Alternative:** Claude desktop app → Code → add an SSH connection `ubuntu@<ip>` with the key file. It installs Claude Code on the box automatically.
- **Teammates** reading Aryan's data: `AMLC_DATA_BUCKET=amlc26-<aryan-account-id> bash ~/box_setup.sh`

## Daily operations
| what | command (laptop) |
|---|---|
| start the box in the morning (IP changes → ssh config updated) | `infra\aws\devbox.ps1 start` |
| your home IP changed / SSH times out | `infra\aws\devbox.ps1 ip` |
| stop now | `infra\aws\devbox.ps1 stop` |
| bigger box after quota approval | `infra\aws\devbox.ps1 resize -Type r7i.2xlarge` |
| GPU box | `infra\aws\devbox.ps1 up -Gpu -Name gpu` → host `amlc-gpu` |
| check credits | Console → Billing → Credits (plus budget e-mails) |

On the box:
- **Long jobs** go in `tmux new -s run` (they survive a VS Code disconnect).
- **Auto-stop** stops the instance after 20 min with no SSH session and idle CPU. To keep a long job running with no SSH session attached, `touch ~/.keepalive`. **Remove it afterwards.**
- **Share artifacts:** `aws s3 sync artifacts/<run_id> s3://$AMLC_BUCKET/artifacts/<run_id>`
- **Commit run records:** `git add -A && git commit -m "<run_id>: …" && git push`

## Safety rules
- Never make the bucket public. Competition data must stay private.
- Stop GPU boxes as soon as a job ends.
- If a budget alert arrives at 80%, tell the team and switch the heavy jobs to another member's account.
