<#
One-time account setup (run on your laptop after `aws login`). Idempotent: safe to re-run.
  - budget alerts (gross usage, credits excluded, so alerts actually fire)
  - EC2 quota increase requests (standard vCPU 32, GPU G/VT 8) in ap-south-1
  - private S3 bucket  amlc26-<account-id>
  - IAM role + instance profile for the dev box (S3 access to amlc26-* buckets only)
  - SSH key pair (saved to ~/.ssh) and a security group allowing SSH from YOUR current IP only
Usage:  powershell -ExecutionPolicy Bypass -File infra\aws\setup_account.ps1 -Email you@example.com
#>
param([Parameter(Mandatory = $true)][string]$Email)
. "$PSScriptRoot\config.ps1"

$acct = Get-AccountId
$bucket = Get-BucketName
Write-Host "Account $acct | region $Region | member '$Member'" -ForegroundColor Cyan

# 1) Budget with e-mail alerts at 25/50/80/100% of $BudgetUsd (credits NOT netted out)
$budget = @{ BudgetName = "$Prefix-budget"; BudgetType = "COST"; TimeUnit = "MONTHLY";
             BudgetLimit = @{ Amount = "$BudgetUsd"; Unit = "USD" };
             CostTypes = @{ IncludeCredit = $false; IncludeRefund = $false } } | ConvertTo-Json -Depth 5
$tmpB = New-TemporaryFile; Set-Content $tmpB $budget -Encoding ascii
$notes = @(25, 50, 80, 100) | ForEach-Object {
    @{ Notification = @{ NotificationType = "ACTUAL"; ComparisonOperator = "GREATER_THAN"; Threshold = $_; ThresholdType = "PERCENTAGE" };
       Subscribers = @(@{ SubscriptionType = "EMAIL"; Address = $Email }) } }
$tmpN = New-TemporaryFile; Set-Content $tmpN (ConvertTo-Json @($notes) -Depth 6) -Encoding ascii
$existing = & aws budgets describe-budgets --account-id $acct --query "Budgets[?BudgetName=='$Prefix-budget'].BudgetName" --output text 2>$null
if (-not $existing) {
    & aws budgets create-budget --account-id $acct --budget "file://$tmpB" --notifications-with-subscribers "file://$tmpN"
    if ($LASTEXITCODE -eq 0) { Write-Host "budget alerts -> $Email" -ForegroundColor Green }
} else { Write-Host "budget exists" }

# 2) Quota increase requests (approval can take hours -> request early)
$quotas = @(@{ Code = "L-1216C47A"; Name = "On-Demand Standard vCPU"; Want = 32 },
            @{ Code = "L-DB2E81BA"; Name = "On-Demand G and VT vCPU (GPU)"; Want = 8 })
foreach ($q in $quotas) {
    $cur = & aws service-quotas get-service-quota --service-code ec2 --quota-code $q.Code --region $Region --query "Quota.Value" --output text 2>$null
    Write-Host ("quota {0}: current {1}" -f $q.Name, $cur)
    if ([double]$cur -lt $q.Want) {
        & aws service-quotas request-service-quota-increase --service-code ec2 --quota-code $q.Code --desired-value $q.Want --region $Region --query "RequestedQuota.Status" --output text 2>&1 | Write-Host
    }
}

# 3) S3 bucket (private, default SSE-S3 encryption: works cross-account)
& aws s3api head-bucket --bucket $bucket --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-Aws s3api create-bucket --bucket $bucket --create-bucket-configuration "LocationConstraint=$Region" | Out-Null
    Invoke-Aws s3api put-public-access-block --bucket $bucket --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" | Out-Null
    Write-Host "bucket s3://$bucket created" -ForegroundColor Green
} else { Write-Host "bucket s3://$bucket exists" }

# 4) IAM role + instance profile (S3 on amlc26-* buckets only; also teammates' shared buckets)
& aws iam get-role --role-name $RoleName 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    $trust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    $tmpT = New-TemporaryFile; Set-Content $tmpT $trust -Encoding ascii
    & aws iam create-role --role-name $RoleName --assume-role-policy-document "file://$tmpT" | Out-Null
    $pol = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:ListBucket","s3:GetObject","s3:PutObject","s3:DeleteObject","s3:GetBucketLocation"],"Resource":["arn:aws:s3:::amlc26-*","arn:aws:s3:::amlc26-*/*"]}]}'
    $tmpP = New-TemporaryFile; Set-Content $tmpP $pol -Encoding ascii
    & aws iam put-role-policy --role-name $RoleName --policy-name "$Prefix-s3" --policy-document "file://$tmpP"
    & aws iam create-instance-profile --instance-profile-name $ProfileName | Out-Null
    & aws iam add-role-to-instance-profile --instance-profile-name $ProfileName --role-name $RoleName
    Write-Host "IAM role/profile created (propagation takes ~10 s)" -ForegroundColor Green
    Start-Sleep -Seconds 10
} else { Write-Host "IAM role exists" }

# 5) SSH key pair (private key stays on this laptop)
if (-not (Test-Path $KeyFile)) {
    New-Item -ItemType Directory -Force (Split-Path $KeyFile) | Out-Null
    & aws ec2 delete-key-pair --key-name $KeyName --region $Region 2>$null | Out-Null
    $mat = Invoke-Aws ec2 create-key-pair --key-name $KeyName --key-type ed25519 --query KeyMaterial --output text
    [IO.File]::WriteAllText($KeyFile, ($mat -replace "`r", "") + "`n")
    icacls $KeyFile /inheritance:r | Out-Null
    icacls $KeyFile /grant:r "$($env:USERNAME):(R)" | Out-Null
    Write-Host "key saved to $KeyFile" -ForegroundColor Green
} else { Write-Host "key exists: $KeyFile" }

# 6) Security group: SSH from current public IP only
$vpc = Invoke-Aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text
$sg = Invoke-Aws ec2 describe-security-groups --filters "Name=group-name,Values=$SgName" "Name=vpc-id,Values=$vpc" --query "SecurityGroups[0].GroupId" --output text
if ($sg -eq "None" -or -not $sg) {
    $sg = Invoke-Aws ec2 create-security-group --group-name $SgName --description "AMLC SSH" --vpc-id $vpc --query GroupId --output text
}
$ip = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com").Trim()
& aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr "$ip/32" --region $Region 2>$null | Out-Null
Write-Host "security group $sg allows SSH from $ip/32" -ForegroundColor Green

Write-Host "`nDone. Next: infra\aws\upload_data.ps1 (once, Aryan) then infra\aws\devbox.ps1 up" -ForegroundColor Cyan
