# Shared settings for the AWS scripts. Dot-sourced by the other .ps1 files.
# Every teammate runs the scripts in THEIR OWN AWS account (per-member credits).
$ErrorActionPreference = "Stop"
$Region      = if ($env:AMLC_REGION) { $env:AMLC_REGION } else { "ap-south-1" }   # Mumbai: low latency from India
$env:AWS_DEFAULT_REGION = $Region   # also for commands that do not pass --region
$Member      = if ($env:AMLC_MEMBER) { $env:AMLC_MEMBER } else { ($env:USERNAME -split " ")[0].ToLower() }
$Prefix      = "amlc26"
$RoleName    = "$Prefix-ec2-role"
$ProfileName = "$Prefix-ec2-profile"
$SgName      = "$Prefix-ssh"
$KeyName     = "$Prefix-$Member"
$KeyFile     = Join-Path $env:USERPROFILE ".ssh\$KeyName.pem"
$SshConfig   = Join-Path $env:USERPROFILE ".ssh\config"
$BudgetUsd   = 100
$DiskGb      = 150

function Invoke-Aws {
    # Run the AWS CLI, throw on failure, return trimmed stdout.
    $out = & aws @args --region $Region 2>&1
    if ($LASTEXITCODE -ne 0) { throw "aws $($args -join ' ') failed:`n$out" }
    return ($out | Out-String).Trim()
}

function Get-AccountId { (Invoke-Aws sts get-caller-identity --query Account --output text) }
function Get-BucketName { "$Prefix-$(Get-AccountId)" }
